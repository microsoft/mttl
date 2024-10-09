import argparse
import os
from collections import defaultdict

import torch 
import tqdm
from datasets import Dataset, load_dataset

from dataset_augmenter import chunk_text
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="sordonia/narrativeqa_sanitized")
    parser.add_argument("--model", type=str, default="Salesforce/SFR-Embedding-2_R")
    parser.add_argument("--hf_id", type=str, required=True)

    parser.add_argument("--document_id_field", type=str, default="document_id")
    parser.add_argument("--document_field", type=str, default="text")
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=20)

    args = parser.parse_args()

    dataset = load_dataset(args.dataset)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = SentenceTransformer(args.model, device="cuda")

    for split in ["train", "validation", "test"]:
    
        if len(dataset) == 3:
            dataset_split = dataset[split]
        else:
            dataset_split = dataset['train'].filter(lambda x : x['split'] == split)

        i = 0
        for example in tqdm.tqdm(dataset_split):

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
                    model.encode(seqs, batch_size=args.batch_size, show_progress_bar=True, convert_to_tensor=True)
                ]
            
            q_embeds, chunk_embeds = embeds
            scores = model.similarity(q_embeds, chunk_embeds)
            k = min(args.top_k, chunk_embeds.size(0))
            topk_scores, topk_chunks = torch.topk(scores, k, dim=1)

            breakpoint()
            xx = 1

            



if __name__ == "__main__":
    main()
