from projects.wiki_experts.src.data_transforms.engines import AutoEngine
from src.data_transforms.base import (
    DataTransformTemplate,
    TransformConfig,
    TransformModel,
)
from dataclasses import dataclass
from src.data_transforms.utils import (
    upload_to_hf_,
)
import numpy as np
from src import mmlu_subject_configs
from datasets import load_dataset
import tqdm
import os


class FactsTemplate(DataTransformTemplate):
    @classmethod
    def apply(cls, context):
        template = """Your task is to extract facts from the following paragraph. Facts should be conveyed with short, concise sentences, each ending with a period and separated by a newline.
For example:
- Fact 1.
- Fact 2.

Here is the paragraph:
{}

Now state facts about the paragraph:
""".format(context)
        return template


@dataclass
class FactsTransformConfig(TransformConfig):
    model_name: str = "gpt-3.5-turbo"
    max_contexts_per_subject: int = -1
    max_documents_per_subject: int = -1
    max_context_length: int = 128


class FactsTransformModel(TransformModel):
    """Transform a dataset of documents into a question answering dataset."""

    def __init__(
        self,
        config: FactsTransformConfig,
    ):
        self.config = config
        self.template = FactsTemplate
        self._llm = None

    def get_dataset_name(self):
        args = [
            f"facts-{self.config.model_setting}",
            f"clen{self.config.max_context_length}",
            f"maxD{self.config.max_documents_per_subject}",
            f"maxC{self.config.max_contexts_per_subject}.jsonl",
        ]
        return "_".join(args)

    def get_seed_dataset_(self, dataset_name, filter_subjects, **options):
        """
        Convert a seed dataset of retrieved content into a tuple of (context, subject, icl_examples).
        """
        dataset = load_dataset(dataset_name)["train"].to_pandas()
        converted_dataset = []

        if type(filter_subjects) == str:
            filter_subject = getattr(mmlu_subject_configs, filter_subjects)

        for subject in filter_subject:
            subject_data = dataset[dataset["subject"] == subject]
            subject_data.sort_values(by="dfq", ascending=False, inplace=True)

            subject_contexts = []
            num_contexts_per_doc = [0]

            for i in tqdm.tqdm(
                range(len(subject_data)), desc=f"Processing {subject}..."
            ):
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
                    print(
                        "Breaking early due to max_contexts_per_subject settings. ",
                        len(subject_contexts),
                    )
                    break

                if (
                    self.config.max_documents_per_subject > 0
                    and i > self.config.max_documents_per_subject
                ):
                    print(
                        "Breaking early due to max_documents_per_subject settings. ",
                        len(subject_contexts),
                    )
                    break

            print(
                "Contexts per document (Avg/Min/Max):",
                np.mean(num_contexts_per_doc),
                np.min(num_contexts_per_doc),
                np.max(num_contexts_per_doc),
            )

            for context in subject_contexts:
                converted_dataset.append(
                    {
                        "id": str(len(converted_dataset)),
                        "context": context["text"],
                        "docno": str(context["docno"]),
                        "subject": subject,
                    }
                )
        return converted_dataset

    def transform(
        self,
        dataset_name,
        filter_subjects,
        upload_to_hub=False,
        output_path="./generated.jsonl",
        **kwargs,
    ):
        output_path = os.environ.get("AMLT_OUTPUT_DIR", output_path)
        if upload_to_hub:
            assert (
                os.environ.get("HF_TOKEN") is not None
            ), "Please set HF_TOKEN env variable."

        # start dataset
        prev_dataset = self.get_seed_dataset_(dataset_name, filter_subjects)
        llm = AutoEngine.from_path(self.config.model_name, api_type="openai")

        templated_contexts = []
        for line in prev_dataset:
            templated_contexts.append(
                self.template.apply(line["context"])
            )
        outputs = llm.generate(templated_contexts, **kwargs)
        breakpoint()
