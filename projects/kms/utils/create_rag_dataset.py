import argparse
import os
from dataclasses import dataclass

import torch
import tqdm
from datasets import Dataset, load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from mttl.arguments import DataArgs
from mttl.models.library.dataset_library import DatasetLibrary
from mttl.utils import remote_login
from projects.kms.utils.dataset_augmenter import chunk_text


@dataclass
class RAGArguments(DataArgs):
    dataset: str = None
    source_dataset: str = None
    source_dataset_type: str = None
    source_document_field: str = "outputs"
    source_dataset_n_items: int = -1
    chunk_source: int = 1
    custom_split_file: str = None
    subsample_split: str = None
    model: str = "Salesforce/SFR-Embedding-2_R"
    final_dataset_id: str = None
    document_id_field: str = "document_id"
    document_field: str = "text"
    block_size: int = 64
    batch_size: int = 32
    top_k: int = 20
    num_workers: int = 1
    worker_id: int = 0
    token_overlap: float = 0.0


def main(args: RAGArguments):

    # create a logger whose prefix if : {worker_id}/{num_workers}
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.addHandler(logging.StreamHandler())
    logger.handlers[0].setFormatter(
        logging.Formatter(f"{args.worker_id}/{args.num_workers} - %(message)s")
    )

    # print all arguments
    for arg in vars(args):
        string = f"{arg}: {getattr(args, arg)}"
        logger.info(string)

    remote_login(os.environ["HF_TOKEN"])
    question_dataset = DatasetLibrary.pull_dataset_with_retry(args.dataset)["train"]

    if args.source_dataset is not None:
        source_dataset = DatasetLibrary.pull_dataset_with_retry(args.source_dataset)
        source_dataset = source_dataset["train"]
        if args.source_dataset_type is not None:
            source_dataset = source_dataset.filter(
                lambda x: x["type"] == args.source_dataset_type
            )
        if args.source_dataset_n_items > 0:

            def map_fn(x):
                x[args.source_document_field] = x[args.source_document_field][
                    : args.source_dataset_n_items
                ]
                return x

            source_dataset = source_dataset.map(map_fn)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = SentenceTransformer(args.model, device="cuda")
    final_dataset = []

    valid_files = None
    if args.custom_split_file is not None:
        import json

        with open(args.custom_split_file, "r") as f:
            valid_files = json.load(f)

        if args.subsample_split is not None:
            valid_files = valid_files[args.subsample_split]
        else:
            valid_files = (
                valid_files["train"] + valid_files["dev"] + valid_files["test"]
            )

    if valid_files is not None:
        question_dataset = question_dataset.filter(
            lambda x: x["document_id"] in valid_files
        )
        if args.source_dataset is not None:
            source_dataset = source_dataset.filter(
                lambda x: x["document_id"] in valid_files
            )

    for ex_id, ex_question in enumerate(tqdm.tqdm(question_dataset)):
        if ex_id % args.num_workers != args.worker_id:
            continue

        logger.info(f"document {ex_id}/{len(question_dataset)}")

        chunked_dataset = []
        doc_id = ex_question[args.document_id_field]

        # document used for the question
        if args.source_dataset is not None:
            documents = source_dataset.filter(lambda x: x["document_id"] == doc_id)
            # for summaries, there will be 1 document per chunk
            # for nqa dataset, a single entry. For quality (there are 2 entries)
            documents = documents[args.source_document_field]

            # For `ql-rag-summaries`, the dataset column `outputs` for each entry has a list of **dict**
            # Compared to `nqa-rag-summaries`, the dataset column `outputs` for each entry has a list of **str**
            if isinstance(documents[0], list) and isinstance(documents[0][0], dict):
                # documents is List[List[{"summary": str}]]
                documents = [
                    item[args.source_dataset_type]
                    for document in documents
                    for item in document
                ]
            # each entry in `document` is either a `str` (for standard datasets)
            # of list[str] if we have multiple summaries per chunk
            # in the latter case, let's flatten the array to have a single list of strings
            elif isinstance(documents[0], list):
                # concatenate `document` into a single array
                documents = [item for document in documents for item in document]

            # Finally, remove duplicates
            documents = list(set(documents))
        else:
            documents = ex_question[args.document_field]

        # Validate that each document in documents is a string
        assert all(isinstance(doc, str) for doc in documents)

        # first, we chunk the text into blocks
        if args.chunk_source:
            for chunk_idx, chunk in enumerate(
                chunk_text(
                    documents,
                    tokenizer,
                    args.block_size,
                    int(args.token_overlap * args.block_size),
                )
            ):
                text = chunk
                chunked_dataset += [text]
        else:
            chunked_dataset += documents

        # second, we preprocess the questions
        def get_detailed_query(q):
            return f"Instruct: Given a question about a work of literature, retrieve relevant passages that answer the question.\nQuery: {q}"

        questions = [get_detailed_query(q) for q in ex_question["questions"]]

        if ex_id == 0:
            logger.info(f"Retrieval Dataset [0]: {chunked_dataset[0]}")
            logger.info(f"Questions [0]: {questions[0]}")

        # finally, we compute the similarity scores
        # let's get the embeddings for the questions and the chunks
        embeds = []
        for seqs in [questions, chunked_dataset]:
            if isinstance(seqs[0], dict):
                # assert only one value is not None
                valid_keys = [k for k in seqs[0].keys() if seqs[0][k] is not None]
                assert len(valid_keys) == 1
                seqs = [seq[valid_keys[0]] for seq in seqs]

            embeds += [
                model.encode(
                    seqs,
                    batch_size=args.batch_size,
                    show_progress_bar=True,
                    convert_to_tensor=True,
                )
            ]

        q_embeds, chunk_embeds = embeds
        scores = model.similarity(q_embeds, chunk_embeds)
        k = min(args.top_k, chunk_embeds.size(0))
        topk_scores, topk_chunks = torch.topk(scores, k, dim=1)

        # finally, we store the most similar chunks in `document_field_name`, as a
        # list of excerpts for each question (so a list of lists)
        excerpts = []
        for q_id, (q, topk) in enumerate(zip(questions, topk_chunks)):
            excerpts += [[chunked_dataset[i] for i in topk]]

        ex_question[args.document_field] = excerpts
        final_dataset += [ex_question]

    final_dataset = Dataset.from_list(final_dataset)

    try:
        name = args.final_dataset_id.split("/")[-1]
        path = f"local:///mnt/output/{name}"
        if args.num_workers == 1:
            DatasetLibrary.push_dataset(final_dataset, path)
        else:
            DatasetLibrary.push_dataset(
                final_dataset, f"{path}-{args.worker_id}-{args.num_workers}"
            )
    except Exception as e:
        try:
            import pickle as pkl

            with open(f'{os.environ["AMLT_OUTPUT_DIR"]}/dataset.pkl', "wb") as f:
                pkl.dump(final_dataset, f)
        except Exception as n_e:
            logger.error(f"Nested Error: {n_e}")
        logger.error(f"Error while pushing the dataset: {e}")
        pass

    if args.num_workers == 1:
        DatasetLibrary.push_dataset(final_dataset, args.final_dataset_id)
    else:
        DatasetLibrary.push_dataset(
            final_dataset,
            f"{args.final_dataset_id}-{args.worker_id}-{args.num_workers}",
        )


if __name__ == "__main__":
    args = RAGArguments.parse()
    main(args)
