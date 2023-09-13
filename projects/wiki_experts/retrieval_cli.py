from collections import defaultdict
import json
import click
import os
import tqdm
import pyterrier as pt
import tqdm
import datasets
from datasets import Dataset


os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
pt.init()


def norm_query(query):
    query = "".join([x if x.isalnum() else " " for x in query])
    return query


def search(index, query):
    index = pt.IndexFactory.of(index)
    bm25 = pt.BatchRetrieve(
        index,
        verbose=True,
        num_results=100,
        wmodel="BM25",
        threads=1,
        metadata=["docno"],
    )
    query = [norm_query(q) for q in query]
    print("Querying with:", query[0])
    result = bm25.transform(query)
    return result


def get_dataset(dataset_name):
    if dataset_name == "wikipedia":
        dataset = datasets.load_dataset("wikipedia", "20220301.en")
    elif dataset_name == 'redpajama-sample':
        dataset = datasets.load_dataset("togethercomputer/RedPajama-Data-1T-Sample")
    elif dataset_name == 'redpajama':
        dataset = datasets.load_dataset("togethercomputer/RedPajama-Data-1T", 'default')
    return dataset


def read_infos(index):
    with open(index + '/infos.json', 'r') as f:
        return json.load(f)


def make_index(dataset_name, path):
    os.makedirs(path, exist_ok=True)

    dataset = get_dataset(dataset_name)
    iter_indexer = pt.IterDictIndexer(
        path,
        fields=['text'],
        meta={'docno': 20},
        meta_reverse=['docno'],
        threads=24
    )

    def aug_with_id():
        for i, document in tqdm.tqdm(enumerate(dataset['train']), total=len(dataset['train'])):
            data = {'docno': str(i), 'text': document['text']}
            yield data

    document_iter = aug_with_id()
    iter_indexer.index(document_iter)

    with open(path + "/infos.json", "w") as f:
        json.dump({'dataset_name': dataset_name}, f)


@click.group()
def cli():
    pass


@cli.command('index')
@click.option("--dataset")
@click.option("--path")
def index(dataset, path):
    make_index(dataset, path=path)


@cli.command('retrieve')
@click.option("--index")
@click.option("--split", help="MMLU split")
def retrieve(index, split):
    # load dataset
    mmlu = datasets.load_dataset("cais/mmlu", "all")

    # group by subjects
    group_by_subject = defaultdict(list)
    for ex in mmlu[split]:
        group_by_subject[ex["subject"]].append(ex["question"])

    # issue a query per subject
    documents_by_subject = {'_index_infos': read_infos(index)}
    dataset_name = documents_by_subject['_index_infos']['dataset_name']

    for subject, questions in tqdm.tqmd(group_by_subject.items()):
        results = search(index, questions)
        docnos = list(results["docno"])
        scores = list(results["score"])

        # agg by score
        docscore = {}
        for docno, score in zip(docnos, scores):
            if docno not in docscore:
                docscore[docno] = {"score": score, "dfq": 1}
            else:
                docscore[docno]["score"] += score
                docscore[docno]["dfq"] += 1

        print(f"Subject: {subject}")
        print(f"Number of questions: {len(questions)}")
        print(f"Number of unique documents retrieved: {len(docscore)}")

        # dfq filter
        docscore = {k: v for k, v in docscore.items() if v["dfq"] >= 2}
        print(f"Number of documents after filtering: {len(docscore)}")

        sorted_docscore = sorted(docscore.items(), key=lambda x: x[1]["score"], reverse=True)
        print(
            f"Top 10 documents: {sorted_docscore[:10]}"
        )
        documents_by_subject[subject] = sorted_docscore

        with open(f"./docs_from_{dataset_name}_for_mmlu_split{split}.json", "w") as f:
            f.write(json.dumps(documents_by_subject, indent=2))


@cli.command('create_dataset')
@click.option("--docs_json", type=str)
@click.option("--hub_name", type=str)
@click.option("--max_tokens", type=int, default=-1)
def create_dataset(docs_json, max_tokens, hub_name):
    with open(docs_json, "rt") as f:
        documents_by_subject = json.load(f)

    infos = documents_by_subject.pop('_index_infos')
    dataset_ = get_dataset(infos['dataset_name'])['train']

    dataset = {
        "subject": [],
        "docno": [],
        "score": [],
        "dfq": [],
    }
    for key in dataset_.features:
        dataset[key] = []

    for subject, documents in documents_by_subject.items():
        num_tokens = 0

        for j, document in enumerate(documents):
            docno = document[0]
            text = dataset_[int(docno)]["text"]
            dataset["subject"].append(subject)
            dataset["docno"].append(int(docno))
            dataset["dfq"].append(document[1]["dfq"])
            dataset["score"].append(document[1]["score"])

            for key in dataset_.features:
                dataset[key].append(dataset_[int(docno)][key])

            num_tokens += len(text.split())
            if num_tokens > max_tokens and max_tokens != -1:
                break

        print(f"=====================")
        print(f"Subject: {subject}")
        print(f"Number of documents: {len(documents)}")
        print(f"Number of tokens: {num_tokens}")
        print(f"Number of added documents: {j + 1}")

    dataset = Dataset.from_dict(dataset)
    dataset.push_to_hub(hub_name, token=os.environ.get("HF_TOKEN"))


if __name__ == '__main__':
    cli()
