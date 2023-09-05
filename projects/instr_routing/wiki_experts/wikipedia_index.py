import datasets
import numpy as np
import os

import tqdm
import pyterrier as pt

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
pt.init()


mmlu = datasets.load_dataset("cais/mmlu", "all")

def make_index(wikipedia):
    os.makedirs("/datadrive2/wikipedia-index", exist_ok=True)

    iter_indexer = pt.IterDictIndexer(
        "/datadrive2/wikipedia-index",
        meta={'docno': 20, 'title': 100, 'text': 4096},
        meta_reverse=['docno'],
        threads=24
    )

    def aug_with_id():
        for i, document in tqdm.tqdm(enumerate(wikipedia['train']), total=len(wikipedia['train'])):
            yield {'docno': str(i), 'title': document['title'], 'text': document['text']}

    iter_indexer.index(aug_with_id())


wikipedia = datasets.load_dataset("wikipedia", "20220301.en")
make_index(wikipedia)