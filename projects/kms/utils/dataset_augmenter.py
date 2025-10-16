import asyncio
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Union

import click
import tenacity
import torch
import tqdm
import vllm
from datasets import Dataset, DatasetDict, DatasetInfo, load_dataset
from tqdm import tqdm as ttqdm
from tqdm.asyncio import tqdm as tqdm_async
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from mttl.logging import logger
from mttl.registrable import Registrable

global client


@dataclass
class OAIGenerations:
    outputs: List = None


@dataclass
class OAIGeneration:
    text: str = None


@tenacity.retry(
    wait=tenacity.wait_random_exponential(min=10, max=60),
    stop=tenacity.stop_after_attempt(100),
)
async def oai_get_completions(
    prompt, model, num_completions=1, top_p=0.8, max_tokens=768
):
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
        top_p=top_p,
        max_tokens=max_tokens,
        n=num_completions,
    )
    # make the results to be in the same format as VLLM
    return OAIGenerations(
        outputs=[
            OAIGeneration(text=choice.message.content) for choice in response.choices
        ]
    )


async def oai_get_completions_batched(
    prompts, model, num_completions, top_p, max_tokens
):
    results = []
    for i in range(0, len(prompts), 100):
        batch_prompts = prompts[i : i + 100]
        batch = await tqdm_async.gather(
            *[
                oai_get_completions(p, model, num_completions, top_p, max_tokens)
                for p in batch_prompts
            ]
        )
        results.extend(batch)
    return results


def chunk_text(
    text: str,
    tokenizer: AutoTokenizer,
    block_size: int = 2048,
    chunk_overlap: Union[float, int] = 0.1,
):
    if isinstance(text, list):
        assert len(text) == 1
        text = text[0]

    if isinstance(chunk_overlap, float):
        assert 0.0 <= chunk_overlap < 1.0
        chunk_overlap = int(block_size * chunk_overlap)

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=block_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_text(text)
    for chunk in chunks:
        yield chunk


class GenerationTask(Registrable):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def get_prompt(self, text):
        return text

    def extract_score(self, text):
        import re

        # Find all question tags
        try:
            scores = re.findall(r"Score:\s(\d+)", text, re.DOTALL | re.IGNORECASE)
        except:
            return -1
        if scores:
            return scores[0]
        return -1

    def get_filter_prompt(self, prompt, text):
        return None

    def create_task(self, text, add_chat_template=True):
        task = self.get_prompt(text)
        if add_chat_template:
            task = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": task}],
                add_generation_prompt=True,
                tokenize=False,
            )
        return task

    def postprocess_generation(self, text):
        return text

    def process_task(self, chunks, rests, tokenizer, llm, generation_params):
        prompts = [self.create_task(chunk) for chunk in chunks]

        outputs = llm.generate(prompts, generation_params)
        results = []

        for chunk, generation_output, rest in zip(chunks, outputs, rests):
            section = {
                "input": chunk,
                "type": self.__class__.registered_name,
                "outputs": [],
            }

            for response in generation_output.outputs:
                processed = self.postprocess_generation(response.text)
                if isinstance(processed, list):
                    section["outputs"].extend(processed)
                else:
                    section["outputs"].append(processed)

            if section["outputs"]:
                section.update(rest)
                results.append(section)
        return results


@dataclass
class SubtopicGenerationTask(Registrable):
    """
    First-pass: Extract subtopics (e.g. distinct points, events, or themes).
    """

    def get_prompt(self, text: str) -> str:
        return (
            "Read the following text and list the distinct subtopics or key points.\n"
            "Just provide a clear bullet list. Example format:\n"
            "- Subtopic 1\n- Subtopic 2\n\n"
            "********** Text **********\n"
            f"{text}\n"
            "********************\n"
            "Please list the subtopics:"
        )

    def postprocess_generation(self, raw_text: str) -> List[str]:
        # Naive bullet-based splitting
        import re

        lines = re.split(r"[\r\n]+", raw_text)
        subtopics = []
        for line in lines:
            line = line.strip("-• \n\r\t")
            if line:
                subtopics.append(line)
        return subtopics


@dataclass
class SubtopicQAGenerationTask(Registrable):
    """
    Second-pass: For each subtopic, generate multiple Q&A pairs.
    """

    questions_per_subtopic: int = 2

    def get_prompt(self, text: str, subtopic: str) -> str:
        return (
            f"Subtopic: {subtopic}\n\n"
            f"Based on this subtopic, generate {self.questions_per_subtopic} question/answer pairs that "
            "require careful reading of the relevant text. The Q&A should not repeat the same info. "
            "Format:\n<question>Q</question>\n<answer>A</answer>\n...\n\n"
            "********** Original Text Chunk **********\n"
            f"{text}"
        )

    def postprocess_generation(self, raw_text: str) -> List[Dict[str, str]]:
        import re

        question_pattern = r"<question>\s*(.*?)\s*</question>"
        answer_pattern = r"<answer>\s*(.*?)\s*</answer>"

        questions = re.findall(question_pattern, raw_text, re.DOTALL | re.IGNORECASE)
        answers = re.findall(answer_pattern, raw_text, re.DOTALL | re.IGNORECASE)

        out = []
        for q, a in zip(questions, answers):
            out.append({"question": q.strip(), "answer": a.strip()})
        return out


@GenerationTask.register("qa")
class QATask(GenerationTask):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.num_questions = 5

    def get_prompt(self, text):
        return (
            f"Create {self.num_questions} questions that can be answerable from the following text, along with their answers.\n"
            "Strive to generate challenging questions that require aggregating information across the provided text.\n"
            "Focus on different sections of the text to increase diversity of the generated questions. Format your answer as follows:\n"
            + "<question id='1'>QUESTION 1 HERE</question>\n"
            + "<answer id='1'>ANSWER 1 HERE</answer>\n"
            + "<question id='2'>QUESTION 2 HERE</question>\n"
            + "<answer id='2'>ANSWER 2 HERE</answer>\n"
            + "\n"
            + "********** Text **********\n"
            + text
            + f"\n********************"
        )

    def get_filter_prompt(self, prompt, qa):
        return (
            f"Below is a question/answer pair written given the text passage.\n"
            + "Evaluate whether or not the question and the answer are correct and make sense.\n"
            + "Please assign a score using the following 5-point scale:\n"
            + "1: The question and answer are both incorrect or nonsensical.\n"
            + "2: The question is correct but the answer is incorrect.\n"
            + "3: The question is correct but trivial, the answer is correct.\n"
            + "4: The question is correct and challenging, the answer is correct.\n"
            + "5: The question is correct and challenging, the answer is correct, insightful and elaborate.\n"
            + "\nPlease derive the rating score, finally write 'Score: <rating>' in the last line.\n"
            + "\n"
            + "********** Text **********\n"
            + prompt
            + f"\n********************\n"
            + f"\n********** Question/Answer **********\n"
            + f"Question: {qa['question']}\n"
            + f"Answer: {qa['answer']}\n"
            + f"\n********************"
        )

    def postprocess_generation(self, text):
        import re

        # Regular expression patterns (more forgiving)
        question_pattern = r"<question\s+id=['\"]?(\d+)['\"]?>\s*(.*?)\s*</question>"
        answer_pattern = r"<answer\s+id=['\"]?(\d+)['\"]?>\s*(.*?)\s*</answer>"

        # Find all question tags
        try:
            questions = re.findall(question_pattern, text, re.DOTALL | re.IGNORECASE)
        except:
            questions = []
            return []

        qa_dict = {}

        for q_id, question in questions:
            qa_dict[q_id] = {"question": question.strip(), "answer": ""}

        # Find all answer tags
        try:
            answers = re.findall(answer_pattern, text, re.DOTALL | re.IGNORECASE)
        except:
            answers = []
            return []

        for q_id, answer in answers:
            if q_id in qa_dict:
                qa_dict[q_id]["answer"] = answer.strip()
            else:
                # Handle case where answer ID doesn't match any question
                print(f"Warning: Answer ID {q_id} has no matching question.")
                qa_dict[q_id] = {"question": "", "answer": answer.strip()}

        # Prepare the final list
        qa_list = []
        for q_id, qa in qa_dict.items():
            if qa["question"] and qa["answer"]:
                qa_list.append(
                    {
                        "question": qa["question"],
                        "answer": qa["answer"],
                    }
                )
            else:
                print(f"Warning: Incomplete data for ID {q_id}.")

        return qa_list


@GenerationTask.register("summary")
class SummaryTask(GenerationTask):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def get_filter_prompt(self, prompt, summary):
        return (
            f"Below is a summary written given the text passage. Evaluate the quality of the summary.\n"
            + "Please assign a score using the following 5-point scale:\n"
            + "1: The summary reports incorrect facts, is badly written or contains grammatical errors.\n"
            + "2: The summary is mostly correct, but it is too short or misses important information.\n"
            + "3: The summary is mostly correct, but it misses some information.\n"
            + "4: The summary is correct, it covers most of the important information in the text.\n"
            + "5: The summary is correct, it covers all the important information, is insightful and elaborate.\n"
            + "\nPlease derive the rating score, finally write 'Score: <rating>' in the last line.\n"
            + "\n"
            + "********** Text **********\n"
            + prompt
            + f"\n********************\n"
            + f"\n********** Summary **********\n"
            + summary["summary"]
            + f"\n********************"
        )

    def get_prompt(self, text):
        return (
            f"Summarize the following text in around {int(len(text) / 4)} words without omitting any important details.\n"
            "The summary should be grammatically correct and summarize all the different sections in the text.\n"
            + "********** Text **********\n"
            + text
            + "\n********************"
        )

    def postprocess_generation(self, text):
        return {"summary": text}


class DatasetAugmenter:
    def __init__(
        self,
        model,
        block_size,
        max_continuation_length,
        num_generations,
        generation_top_p,
        model_type="oai",
        num_devices=-1,
        do_filtering=True,
    ):
        import os

        self.tasks = {}
        self.model = model
        self.block_size = block_size
        self.max_continuation_length = max_continuation_length
        self.num_generations = num_generations
        self.generation_top_p = generation_top_p
        self.do_filtering = do_filtering

        self.oai = model_type in ["oai", "azure_oai"]
        if not self.oai:
            # Set the multiprocessing method to spawn to avoid issues with torch.multiprocessing
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

            self.tokenizer = AutoTokenizer.from_pretrained(model)
            self.sampling_params = SamplingParams(
                n=num_generations,
                top_p=generation_top_p,
                stop_token_ids=[self.tokenizer.eos_token_id],
                max_tokens=max_continuation_length,
            )
            self.llm = LLM(
                model=model,
                trust_remote_code=True,
                gpu_memory_utilization=0.9,
                dtype="bfloat16",
                tensor_parallel_size=int(
                    os.environ.get(
                        "VLLM_TP_SIZE",
                        torch.cuda.device_count() if num_devices == -1 else num_devices,
                    )
                ),
                max_num_seqs=64,
                max_model_len=max(4096, self.tokenizer.model_max_length),
            )
            logger.warning(
                f"DatasetAugmenter: Setting max_model_len to {self.tokenizer.model_max_length}."
            )
        else:
            from openai import AsyncAzureOpenAI, AsyncOpenAI, OpenAI

            global client

            if model_type == "oai":
                client = AsyncOpenAI(
                    api_key=os.environ.get("OPENAI_API_KEY"),
                    base_url=os.environ.get("OPENAI_BASE_URL"),
                )
            else:
                client = AsyncAzureOpenAI(
                    api_key=os.environ.get("OPENAI_API_KEY"),
                    azure_endpoint=os.environ.get("OPENAI_BASE_URL"),
                    api_version=os.environ.get("OPENAI_API_VERSION"),
                )

            self.tokenizer = AutoTokenizer.from_pretrained(
                "microsoft/Phi-3-medium-4k-instruct"
            )
            self.llm = None

    def add_task(self, task):
        task_gen = GenerationTask.get_class_by_name(task)(self.tokenizer)
        self.tasks[task] = task_gen
        return self

    def filter(
        self,
        dataset: Dataset,
    ):
        """Filters data produced by the augment method."""
        indices, prompts = [], []
        for i, example in enumerate(dataset):
            for output in example["outputs"]:
                prompt = self.tasks[example["type"]].get_filter_prompt(
                    example["input"], output
                )
                if prompt:
                    prompts.append(prompt)
                    indices.append(i)

        if not self.oai:
            outputs = self.llm.generate(prompts, self.sampling_params)
        else:
            outputs = asyncio.run(
                oai_get_completions_batched(
                    prompts,
                    self.model,
                    num_completions=1,
                    top_p=self.generation_top_p,
                    max_tokens=self.max_continuation_length,
                )
            )

        scores = [[] for _ in range(len(dataset))]
        for index, generation_output in zip(indices, outputs):
            example = dataset[index]
            type = example["type"]

            text = generation_output.outputs[0].text
            text = self.tasks[type].extract_score(text)
            scores[index].append(text)

        dataset = dataset.add_column("scores", scores)
        return dataset

    def augment(
        self,
        dataset: Dataset,
        carry_columns: str = "all",
    ) -> Dataset:
        import tqdm

        prompts, chunks, types, output_dataset, rests = [], [], [], [], []

        for doc_idx in tqdm.tqdm(range(len(dataset))):
            text = dataset[doc_idx]["text"]
            chunks_iterator = chunk_text(text, self.tokenizer, self.block_size)

            for cid, chunk in enumerate(chunks_iterator):
                print(
                    f"Chunking {doc_idx} chunk {cid + 1} with {len(self.tokenizer.encode(chunk))} tokens"
                )
                chunks.append(chunk)
                # append the rest of the columns
                rest = {}
                for column in dataset.column_names:
                    if column != "text" and (
                        carry_columns == "all" or column in carry_columns
                    ):
                        rest[column] = dataset[doc_idx][column]
                rests.append(rest)
            print(f"Chunking {doc_idx} done with {cid + 1} chunks")

        # Process each task separately
        output_dataset = []
        for task_name, task in tqdm.tqdm(self.tasks.items(), desc="Processing tasks"):
            logger.info(f"Processing task: {task_name}")

            # For local models, use synchronous processing
            generation_params = {
                "num_completions": self.num_generations,
                "top_p": self.generation_top_p,
                "max_tokens": self.max_continuation_length,
            }

            # NOTE: let's change to something VLLM can understand
            # TODO: check with AS why `generation_params` was rebuilt here
            generation_params = self.sampling_params

            results = task.process_task(
                chunks,
                rests,
                self.tokenizer,
                self.llm,
                generation_params=generation_params,
            )
            output_dataset.extend(results)

            # Print some examples for debugging
            for n, result in enumerate(results[:5]):
                print("********************")
                print("Type: ", result["type"])
                print("Input: ", result["input"])
                print("Output: ", result["outputs"][0])
                print("********************")

        d = Dataset.from_list(output_dataset)

        if self.do_filtering:
            d = self.filter(d)

        return d
