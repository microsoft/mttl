import argparse
import os

import torch
import tqdm
from datasets import Dataset, load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from mttl.models.library.dataset_library import DatasetLibrary
from mttl.utils import remote_login
from projects.kms.utils.dataset_augmenter import chunk_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="sordonia/narrativeqa_sanitized")
    parser.add_argument("--source_dataset", type=str, default=None)
    parser.add_argument("--source_dataset_type", type=str, default="summary")
    parser.add_argument("--source_document_field", type=str, default="outputs")
    parser.add_argument("--source_dataset_n_items", type=int, default=-1)
    parser.add_argument("--chunk_source", default=1, type=int)
    parser.add_argument(
        "--subsample_file", type=str, default=None
    )  # "nqa_splits/nqa_common_split.json")
    parser.add_argument("--subsample_split", type=str, default=None)
    parser.add_argument("--model", type=str, default="Salesforce/SFR-Embedding-2_R")
    parser.add_argument("--final_dataset_id", type=str, required=True)

    parser.add_argument("--document_id_field", type=str, default="document_id")
    parser.add_argument("--document_field", type=str, default="text")
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument(
        "--token_overlap",
        default=0.1,
        help="chunk (text segment between two `\n`) overlap between RAG passages",
    )

    args = parser.parse_args()

    remote_login(os.environ["HF_TOKEN"])
    question_dataset = DatasetLibrary.pull_dataset_with_retry(args.dataset)["train"]

    if args.source_dataset is not None:
        source_dataset = DatasetLibrary.pull_dataset_with_retry(args.source_dataset)
        source_dataset = source_dataset["train"]
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
    if args.subsample_file is not None:
        import json

        with open(args.subsample_file, "r") as f:
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
        source_dataset = source_dataset.filter(
            lambda x: x["document_id"] in valid_files
        )

    for ex_id, ex_question in enumerate(tqdm.tqdm(question_dataset)):
        if ex_id % args.num_workers != args.worker_id:
            continue

        print(f"document {ex_id}/{len(question_dataset)}")

        chunked_dataset = []
        doc_id = ex_question[args.document_id_field]

        # document used for the question
        if args.source_dataset is not None:
            document = source_dataset.filter(lambda x: x["document_id"] == doc_id)
            document = document[args.source_document_field]
            # concatenate `document` into a single array
            document = [item for sublist in document for item in sublist]
        else:
            document = ex_question[args.document_field]

        # first, we chunk the text into blocks
        if args.chunk_source:
            for chunk_idx, chunk in enumerate(
                chunk_text(
                    document,
                    tokenizer,
                    args.block_size,
                    int(args.token_overlap * args.block_size),
                )
            ):
                text = chunk
                chunked_dataset += [text]
        else:
            chunked_dataset += document

        # second, we preprocess the questions
        def get_detailed_query(q):
            return f"Instruct: Given a question about a work of literature, retrieve relevant passages that answer the question.\nQuery: {q}"

        questions = [get_detailed_query(q) for q in ex_question["questions"]]

        # finally, we compute the similarity scores
        # let's get the embeddings for the questions and the chunks
        embeds = []
        for seqs in [questions, chunked_dataset]:
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

    if args.num_workers == 1:
        DatasetLibrary.push_dataset(final_dataset, args.final_dataset_id)
    else:
        DatasetLibrary.push_dataset(
            final_dataset,
            f"{args.final_dataset_id}_{args.worker_id}_{args.num_workers}",
        )


if __name__ == "__main__":
    main()
