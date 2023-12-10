from collections import defaultdict
from dataclasses import dataclass
import numpy as np
import os

from datasets import load_dataset
import tqdm
import re

from mttl.dataloader.platypus_dataset_reader import (
    InversePlatypusTemplate,
    PlatypusTemplate,
)
from mttl.utils import logger

from src.data_transforms.base import (
    TransformConfig,
    DataTransformTemplate,
    TransformModel,
)
from mttl.vllm_engines.engines import free_memory
from src.data_transforms.engines import (
    OpenAI,
    AutoEngine,
)
from src import mmlu_subject_configs
from src.data_transforms.utils import (
    INVALID_RESPONSE,
    read_jsonl_dataset,
    dump_jsonl_dataset,
    upload_to_hf_,
    reject_output,
    count_repeating_sentences,
)


class QAPlatyInstructionGenerationTemplate(DataTransformTemplate):
    @classmethod
    def apply(cls, output, context=None, icl_examples=None, **kwargs):
        """
        We provide the context as output (response) if and only if a response is not
        present. Otherwise, we provide the context as context in addition to the previously generated response.
        """
        return InversePlatypusTemplate.apply(
            output if output is not None else context,
            input if output is not None else None,
            icl_examples,
        )

    @classmethod
    def post_process_generation(cls, output):
        return {"instruction": output}


class QAPlatyResponseGenerationTemplate(PlatypusTemplate):
    @classmethod
    def apply(cls, instruction, context=None):
        return PlatypusTemplate.apply(instruction, input=context)

    @classmethod
    def post_process_generation(cls, output):
        return {"response": output}


class OAITemplate:
    @classmethod
    def post_process_generation(cls, output):
        try:
            if "### Response:" in output:
                instruction, _, response = output.split("### Response:")
                instruction = instruction.split("### Instruction:")[1]
                instruction = instruction.strip()
                response = response.strip()
            else:
                raise

            if not instruction or not response:
                raise

            # this is very likely an instruction
            if response.startswith("Please"):
                raise

            if instruction[-1] in [";", ",", ":"]:
                instruction = instruction[:-1]
            if instruction[-1] not in [".", "?", "!"]:
                instruction += "."
            if response[-1] in [";", ",", ":"]:
                response = response[:-1]
            if response[-1] not in [".", "?", "!"]:
                response[-1] += "."

            data = {"instruction": instruction, "response": response}
        except:
            data = {"instruction": INVALID_RESPONSE, "response": INVALID_RESPONSE}
        return data

    @classmethod
    def apply(cls, context, output, icl_examples=None, **kwargs):
        task_description = "Your task is to generate a clear instruction and corresponding response. To formulate your instruction, you will be given a context.\
Please use the context to formulate an instruction that is clear and comprehensive.\
Your instruction must be complete, in the sense that it must not need to have access to the context in order to be followed."

        if icl_examples is not None:
            task_description += f"\nHere are examples of some good instructions and responses formulated under different contexts. Strive to match the style, tone, and length of these examples:\n"
            for icl_input, icl_output in icl_examples:
                task_description += f"\n\n### Instruction:\n{icl_input}"
                task_description += f"\n\n### Response:\n{icl_output}"
            task_description += "\n\n"

        task_description += "This is the context.\n\n## Context:\n" + context

        task_description += "\n\nRemember, your should generate one instruction reponse pair. \
Format your output as follows:\
\n\n### Instruction:\n<your instruction>\
\n\n### Response:\n<your response>"
        return task_description


class OAITemplate_Batched:
    """
    Creates multiple instruction - response pairs per context
    """

    @classmethod
    def transform_to_valid_list(cls, output):
        output = output + ">" if output[-1] != ">" else output
        output = output.replace("\n", " ")
        output = output.replace('"', "'")
        output = f"[{output}]"

        output = re.sub(r"\d?\.?\s*<\s*#", '"#', output)
        output = re.sub(r"\d?\.?\s*<\s*Instruction", '"Instruction', output)
        output = re.sub(r"<\s*\d?\.?\s*Instruction", '"Instruction', output)

        output = re.sub(r">,?\.?\s*\"#", '","#', output)
        output = re.sub(r">,?\.?\s*\"Instruction", '","Instruction', output)

        output = re.sub(r">,?\.?\s*]", '"]', output)  # end of the string
        return output

    @classmethod
    def clean_response(cls, response):
        response = response.strip()
        if not response:
            return INVALID_RESPONSE
        if response[0] == "[" and response[-1] == "]":
            response = response[1:-1]
        if response[-1] in [";", ",", ":"]:
            response = response[:-1]
        if response[-1] not in [".", "?", "!"]:
            response += "."
        # this is very likely an instruction
        if response.startswith("Please"):
            return INVALID_RESPONSE
        return response

    @classmethod
    def clean_instruction(cls, instruction):
        instruction = instruction.strip()
        if not instruction:
            return INVALID_RESPONSE
        if instruction[0] == "[" and instruction[-1] == "]":
            instruction = instruction[1:-1]
        if instruction[-1] in [";", ",", ":"]:
            instruction = instruction[:-1]
        if instruction[-1] not in [".", "?", "!"]:
            instruction += "."
        return instruction

    @classmethod
    def post_process_generation(cls, output):
        try:
            output = cls.transform_to_valid_list(output)
            outputs = eval(output)
        except Exception as e:
            return {"instruction": INVALID_RESPONSE, "response": INVALID_RESPONSE}

        responses = []
        for o in outputs:
            try:
                if "Response:" in o:
                    response = re.split(r"Response\s*\d*:", o)[1].strip()
                else:
                    raise

                instruction = o.split("Response:")[0]
                instruction = re.split(r"Instruction\s*\d*:", instruction)[1].strip()

                instruction = instruction.replace("#", "")
                response = response.replace("#", "")

                instruction = cls.clean_instruction(instruction.strip())
                response = cls.clean_response(response.strip())

                if instruction is INVALID_RESPONSE or response is INVALID_RESPONSE:
                    raise ValueError("Invalid instruction or response.")

                data = {
                    "instruction": instruction,
                    "response": response,
                }
            except Exception as e:
                data = {"instruction": INVALID_RESPONSE, "response": INVALID_RESPONSE}
            responses.append(data)
        return responses

    @classmethod
    def apply(cls, context, output, domain, icl_examples=None):
        domain = domain.replace("_", " ")
        batch_size = "five"
        task_description = f"\nYour task is to come up with a set of {batch_size} instructions paired with ground truth response about the {domain} domain. These task instructions will be given to a GPT model and we will evaluate the GPT model for completing the instructions."
        task_description += f"\nTo ensure a diverse set of instructions, please adhere to the following requirements:\
        \n1. Try not to repeat the verb for each instruction to maximize diversity.\
        \n2. The language used for the instruction also should be diverse. For example, you should combine questions with imperative instructions.\
        \n3. The type of instructions should be diverse. Include diverse types of tasks like open-ended generation, multiple choice, classification, editing, etc.\
        \n4. A GPT language model should be able to complete the instruction. For example, do not ask the assistant to create any visual or audio output. For another example, do not ask the assistant to wake you up at 5pm or set a reminder because it cannot perform any action."
        task_description += f"\n5. The instructions should be 1 to 3 sentences long. Either an imperative sentence or a question is permitted.\
        \n6. Ensure diverse tasks are covered in the instructions and inputs, while focusing on the {domain} domain.\
        \n7. Provide a ground truth response for each of the 10 generated instruction.\
        \n8. Ensure that each instruction is complete, containing all the necessary context for successful execution."
        task_description += (
            "\n9. Your instruction must be grounded in the following context:"
        )
        task_description += f"\n\n{context}"

        if icl_examples is not None:
            task_description += f"\n10. Strive to match the style, tone, and length of these examples of good instructions:\n"
            for icl_example in icl_examples:
                task_description += f"\n### Instruction:\n{icl_example['instruction']}"
                task_description += f"\n{icl_example['options']}"
            task_description += "\n\n"

        task_description += f"Output format: please format your generated instructions and responses as a list, where each pair of instruction and response is enclosed in angle brackets < and >, and formated as: < ### Instruction: [generated instruction], ### Response: [response] >.\
            \nPlease follow these guidelines carefully when generating instructions and responses. Your role is vital in maintaining high standards of communication effectiveness."
        task_description += "\nYour output in the required format:"
        return task_description


class OAITemplate_Batched_MultiChoice:
    @classmethod
    def post_process_generation(cls, generated_output):
        # Regular expression pattern to split the string into problem, options, and response
        pattern = r"### Problem:\s*(.*?)\s*### Options:\s*(.*?)\s*### Response:\s*(.*?)\s*(?=##|$)"

        # Find all matches in the input string
        matches = re.findall(pattern, generated_output, re.DOTALL)

        data = []
        for match in matches:
            if not len(match) == 3:
                # skipping item
                continue

            instruction = match[0].strip()
            options = match[1].strip()
            response = match[2].strip()

            if not instruction or not options or not response:
                # skipping item
                continue

            options_split = re.split(r"\s*[A-D]\.\s*", options)
            if not len(options_split) == 5:
                # skipping item
                continue

            instruction = (
                "Question:\n{instruction}\nChoices:\n{options}\nAnswer:".format(
                    instruction=instruction, options=options
                )
            )
            data.append({"instruction": instruction, "response": response})
        return data

    @classmethod
    def apply(cls, context, output, domain, icl_examples=None):
        domain = domain.replace("_", " ")
        batch_size = "five"
        task_description = f"""Your task is to come up with a set of {batch_size} diverse multiple-choice problems, each with their answer options and ground-truth response about the following domain: {domain}.

Please stick to the following format for your output:

## Example [number of the example]
### Problem:
[your generated problem]

### Options:
A. [your first generated options]
B. [your second generated options]
C. [your third generated options]
D. [your fourth generated options]

### Response:
[the correct response, which should be A., B., C. or D.]

For example:

## Example 1
### Problem:
{icl_examples[0]['instruction']}

### Options:
{icl_examples[0]['options']}

### Response:
{icl_examples[0]['response']}

You can take inspiration from the following context to generate your questions:

{context}

Now generate {batch_size} diverse problems with their options and responses. Please follow these guidelines carefully when generating problems and responses:
"""
        return task_description


@dataclass
class QAModelSetting:
    inverse_model_path: str
    model_path: str
    instruction_template: str
    response_template: str

    @property
    def model_paths(self):
        return self.inverse_model_path, self.model_path


QA_MODEL_SETTINGS = {
    "platy": QAModelSetting(
        inverse_model_path="sordonia/llama2-13b-platypus-inverse",
        model_path="sordonia/llama2-13b-platypus",
        instruction_template=QAPlatyInstructionGenerationTemplate(),
        response_template=QAPlatyResponseGenerationTemplate(),
    ),
    "openai": QAModelSetting(
        inverse_model_path="gpt-35-turbo",
        model_path="gpt-35-turbo",
        instruction_template=OAITemplate(),
        response_template=OAITemplate(),
    ),
    "openai_batched": QAModelSetting(
        inverse_model_path="gpt-35-turbo",
        model_path="gpt-35-turbo",
        instruction_template=OAITemplate_Batched(),
        response_template=OAITemplate_Batched(),
    ),
    "openai_batched_multichoice": QAModelSetting(
        inverse_model_path="gpt-35-turbo",
        model_path="gpt-35-turbo",
        instruction_template=OAITemplate_Batched_MultiChoice(),
        response_template=OAITemplate_Batched_MultiChoice(),
    ),
}


@dataclass
class QATransformConfig(TransformConfig):
    model_setting: str
    max_context_length: int = 512
    max_tokens_instruction: int = 128
    max_tokens_response: int = 1024
    top_p: float = 0.9
    num_iterations: int = 1
    temperature: float = 0.7
    max_documents_per_subject: float = -1
    max_contexts_per_subject: float = -1
    icl_examples: int = 0
    icl_dataset: str = "lukaemon/mmlu"
    icl_split: str = "validation"
    icl_use_options: str = True


class MMLUICLSampler:
    def __init__(
        self,
        dataset="lukaemon/mmlu",
        split="validation",
        use_options=True,
    ):
        logger.info("Creating MMLU ICL Sampler...")

        self.dataset = dataset
        self.split = split
        self.use_options = use_options
        self._cache_dataset = defaultdict(lambda: None)

    def sample(self, num_examples, subject):
        examples = []
        if subject not in self._cache_dataset:
            dataset = load_dataset(self.dataset, subject, split=self.split)
            self._cache_dataset[subject] = dataset
        else:
            dataset = self._cache_dataset[subject]

        indices = np.random.choice(len(dataset), size=num_examples, replace=False)
        for idx in indices:
            idx = int(idx)
            example = dataset[idx]["input"]
            options = ""
            for ans_option in ["A", "B", "C", "D"]:
                options += (f"{ans_option}. " + dataset[idx][ans_option]) + "\n"
            options = options.strip()
            example = {
                "instruction": example,
                "options": options,
                "response": dataset[idx]["target"],
            }
            examples.append(example)
        return examples


class QATransformModel(TransformModel):
    """Transform a dataset of documents into a question answering dataset."""

    def __init__(
        self,
        config: QATransformConfig,
    ):
        self.config = config
        self.model_path = QA_MODEL_SETTINGS[self.config.model_setting].model_path
        self.inverse_model_path = QA_MODEL_SETTINGS[
            self.config.model_setting
        ].inverse_model_path
        self.instruction_template = QA_MODEL_SETTINGS[
            self.config.model_setting
        ].instruction_template
        self.response_template = QA_MODEL_SETTINGS[
            self.config.model_setting
        ].response_template
        self._llm = None

    def get_dataset_name(self, iter_signature=""):
        args = [
            f"qa-{self.config.model_setting}",
            f"icl{self.config.icl_examples}",
            f"clen{self.config.max_context_length}",
            f"maxD{self.config.max_documents_per_subject}",
            f"maxC{self.config.max_contexts_per_subject}",
            f"{iter_signature}.jsonl",
        ]
        return "_".join(args)

    def _load_llm(self, from_path, **options):
        del self._llm
        free_memory()
        self._llm = AutoEngine.from_path(from_path, **options)
        return self._llm

    def transform(
        self,
        dataset_name,
        filter_subjects,
        upload_to_hub=False,
        output_path="./generated.jsonl",
    ):
        output_path = os.environ.get("AMLT_OUTPUT_DIR", output_path)
        if upload_to_hub:
            assert (
                os.environ.get("HF_TOKEN") is not None
            ), "Please set HF_TOKEN env variable."

        # start dataset
        prev_dataset = self.get_seed_dataset_(dataset_name, filter_subjects)

        for i in range(self.config.num_iterations):
            inst_filename = os.path.join(
                output_path, self.get_dataset_name("inst_%d" % i)
            )
            answ_filename = os.path.join(output_path, self.get_dataset_name("%d" % i))

            self._load_llm(self.model_path)
            is_openai = type(self._llm) == OpenAI

            if not os.path.exists(inst_filename):
                instruction_dataset = self.generate_instructions_(
                    self._llm,
                    prev_dataset,
                    answ_filename if is_openai else inst_filename,
                )
            else:
                instruction_dataset = read_jsonl_dataset(inst_filename)

            # we generate only once w openai
            if is_openai:
                break

            self._load_llm(self.inverse_model_path)
            if not os.path.exists(answ_filename):
                answer_dataset = self.generate_answers_(
                    self._llm,
                    instruction_dataset,
                    answ_filename,
                )
            else:
                answer_dataset = read_jsonl_dataset(answ_filename)
            prev_dataset = answer_dataset

        if upload_to_hub:
            print("Uploading the final dataset to HuggingFace Hub...")
            upload_to_hf_(answ_filename, configuration=self.config)

    def generate_instructions_(
        self,
        llm: AutoEngine,
        dataset,
        dump_filename,
    ):
        """
        To generate instructions, we take num_contexts_per_document chunks of length max_context_length from each document,
        then sample 1 instruction from each chunk.

        All instructions are appended as jsonl into output_filename.
        """
        import copy

        def get_templated_context(entry):
            return self.instruction_template.apply(
                context=entry["context"],
                output=entry["response"] if "response" in entry else None,
                icl_examples=entry["icl_examples"],
                domain=entry["subject"],
            )

        templated_contexts = [get_templated_context(entry) for entry in dataset]

        print("Example generation requests...")
        for context in np.random.choice(templated_contexts, 5):
            print(context)
            print()

        result = iter(
            llm.generate(
                templated_contexts,
                top_p=self.config.top_p,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens_instruction,
                stream=True,
            )
        )

        new_dataset = []
        for entry, output_and_reason in zip(dataset, result):
            instruction, finish_reason = output_and_reason
            data = self.instruction_template.post_process_generation(instruction)
            if isinstance(data, list):
                for d in data:
                    if reject_output(d["instruction"], finish_reason):
                        continue

                    copied_entry = copy.deepcopy(entry)
                    copied_entry.update(
                        {
                            "author_instr": str(llm.model_name),
                        }
                    )
                    copied_entry.update(d)
                    new_dataset.append(copied_entry)
            else:
                if reject_output(data["instruction"], finish_reason):
                    continue

                copied_entry = copy.deepcopy(entry)
                copied_entry.update(
                    {
                        "author_instr": str(llm.model_name),
                    }
                )
                copied_entry.update(data)
                new_dataset.append(copied_entry)

            dump_jsonl_dataset(new_dataset, dump_filename)

        print("Created a new instruction dataset of size:", len(new_dataset))
        return new_dataset

    def generate_answers_(
        self,
        llm: AutoEngine,
        dataset,
        dump_filename,
    ):
        requests = []
        for instance in tqdm.tqdm(dataset):
            requests.append(
                self.response_template.apply(
                    instance["instruction"],
                    instance["context"],
                )
            )

        result = iter(
            llm.generate(
                requests,
                top_p=self.config.top_p,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens_response,
                stream=True,
            )
        )

        new_dataset = []
        for entry, output_and_reason in zip(dataset, result):
            response, finish_reason = output_and_reason

            import copy

            data = self.response_template.post_process_generation(response)
            if reject_output(data["response"], finish_reason):
                continue

            # lets also avoid outputs that contains repetitions of the same sentence more than once
            n_rep = count_repeating_sentences(response)
            if n_rep > 0:
                continue

            entry = copy.deepcopy(entry)
            entry.update(
                {
                    "response": data["response"],
                    "author_response": str(llm.model_name),
                }
            )
            new_dataset.append(entry)
            dump_jsonl_dataset(new_dataset, dump_filename)

        print("Created a new answer dataset of size:", len(new_dataset))
        return new_dataset

    def get_seed_dataset_(self, dataset_name, filter_subjects, **options):
        """
        Convert a seed dataset of retrieved content into a tuple of (context, subject, icl_examples).
        """
        dataset = load_dataset(dataset_name)["train"].to_pandas()
        converted_dataset = []

        if type(filter_subjects) == str:
            filter_subject = getattr(
                mmlu_subject_configs, filter_subjects, filter_subjects.split(",")
            )

        print("Filtering subjects:", filter_subject)

        pbar = tqdm.tqdm(filter_subject)

        for subject in pbar:
            subject_data = dataset[dataset["subject"] == subject]
            subject_data = subject_data.sort_values(by="dfq", ascending=False)

            subject_contexts = []
            num_contexts_per_doc = [0]
            for i in range(len(subject_data)):
                document = subject_data.iloc[i]
                text = document["text"]

                sentences = text.split(".")
                sentences = [
                    sentence.strip().replace("\n", " ").replace("  ", " ")
                    for sentence in sentences
                    if len(sentence.strip()) > 0
                ]

                # new document
                document_contexts = []
                for sentence in sentences:
                    sentence = sentence + "."
                    if not document_contexts:
                        document_contexts.append(sentence)
                    else:
                        if (
                            len(document_contexts[-1].split()) + len(sentence.split())
                            < self.config.max_context_length
                        ):
                            document_contexts[-1] += " " + sentence
                        else:
                            document_contexts.append(sentence)

                num_contexts_per_doc.append(len(document_contexts))
                subject_contexts.extend(
                    {
                        "text": context,
                        "docno": str(document["docno"]),
                    }
                    for context in document_contexts
                )

                if (
                    self.config.max_contexts_per_subject > 0
                    and len(subject_contexts) > self.config.max_contexts_per_subject
                ):
                    subject_contexts = subject_contexts[
                        : self.config.max_contexts_per_subject
                    ]
                    break

                if (
                    self.config.max_documents_per_subject > 0
                    and i > self.config.max_documents_per_subject
                ):
                    break

            pbar.set_description_str(
                f"Subject: {subject}, Contexts: {len(subject_contexts)}..."
            )

            for context in subject_contexts:
                converted_dataset.append(
                    {
                        "id": str(len(converted_dataset)),
                        "context": context["text"],
                        "docno": str(context["docno"]),
                        "subject": subject,
                        "icl_examples": self.icl_sampler.sample(
                            self.config.icl_examples, subject
                        )
                        if self.config.icl_examples > 0
                        else None,
                    }
                )
        return converted_dataset
