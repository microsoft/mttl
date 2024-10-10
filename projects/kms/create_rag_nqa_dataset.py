import argparse
import os

import torch
import tqdm
from dataset_augmenter import chunk_text
from datasets import Dataset, load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from mttl.utils import remote_login


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="sordonia/narrativeqa_sanitized")
    parser.add_argument("--model", type=str, default="Salesforce/SFR-Embedding-2_R")
    parser.add_argument("--hf_id", type=str, required=True)

    parser.add_argument("--document_id_field", type=str, default="document_id")
    parser.add_argument("--document_field", type=str, default="text")
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--worker_id", type=int, default=0)

    args = parser.parse_args()

    remote_login(os.environ["HF_TOKEN"])
    dataset = load_dataset(args.dataset)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = SentenceTransformer(args.model, device="cuda")
    final_dataset = []

    for split in ["train", "validation", "test"]:

        if len(dataset) == 3:
            dataset_split = dataset[split]
        else:
            dataset_split = dataset["train"].filter(lambda x: x["split"] == split)

        for ex_id, example in enumerate(tqdm.tqdm(dataset_split)):

            if ex_id % args.num_workers != args.worker_id:
                continue

            chunked_dataset = []
            document = example[args.document_field]
            doc_id = example[args.document_id_field]

            # first, we chunk the text into blocks
            for chunk in chunk_text(document, tokenizer, args.block_size):
                chunk_id, text = chunk
                chunked_dataset += [text]

            # second, we preprocess the questions
            def get_detailed_query(q):
                return f"Instruct: Given a question about a work of literature, retrieve relevant passages that answer the question.\nQuery: {q}"

            questions = [get_detailed_query(q) for q in example["questions"]]

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

            example[args.document_field] = excerpts
            final_dataset += [example]

    final_dataset = Dataset.from_list(final_dataset)
    final_dataset.push_to_hub(f"{args.hf_id}_{args.worker_id}_{args.num_workers}")


if __name__ == "__main__":
    main()
