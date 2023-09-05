import datasets
import os
from collections import defaultdict
import pyterrier as pt
import json


os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
pt.init()


def norm_query(query):
    query = "".join([x if x.isalnum() else " " for x in query])
    return query


def search(query):
    index = pt.IndexFactory.of("/datadrive2/wikipedia-index/data.properties")
    bm25 = pt.BatchRetrieve(
        index,
        verbose=True,
        num_results=100,
        wmodel="BM25",
        threads=1,
        metadata=["docno", "title"],
    )
    query = [norm_query(q) for q in query]
    print("Querying with:", query[0])
    result = bm25.transform(query)
    return result


def retrieve(split):
    # load dataset
    mmlu = datasets.load_dataset("cais/mmlu", "all")

    # group by subjects
    group_by_subject = defaultdict(list)
    for ex in mmlu[split]:
        group_by_subject[ex["subject"]].append(ex["question"])

    # issue a query per subject
    documents_by_subject = {}

    for subject, questions in group_by_subject.items():
        results = search(questions)
        docnos = list(results["docno"])
        scores = list(results["score"])
        titles = list(results["title"])

        # agg by score
        docscore = {}
        for docno, score, title in zip(docnos, scores, titles):
            if docno not in docscore:
                docscore[docno] = {"score": score, "title": title, "dfq": 1}
            else:
                docscore[docno]["score"] += score
                docscore[docno]["dfq"] += 1

        print(f"Subject: {subject}")
        print(f"Number of questions: {len(questions)}")
        print(f"Number of unique documents retrieved: {len(docscore)}")

        sorted_docscore = sorted(docscore.items(), key=lambda x: x[1]["score"], reverse=True)
        print(
            f"Top 10 documents: {sorted_docscore[:10]}"
        )
        documents_by_subject[subject] = sorted_docscore

        with open(f"./documents_by_subject_split{split}.json", "w") as f:
            f.write(json.dumps(documents_by_subject, indent=2))


retrieve("validation")