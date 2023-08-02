import argparse
import gzip 
import itertools
import json
import numpy as np
import pandas as pd
import pickle
import torch
import tqdm      
from itertools import chain
from pathlib import Path      
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm
from typing import Dict
from kmeans_pytorch import KMeans as BalancedKMeans
from datasets import load_dataset
import sys
sys.path.append("/home/v-oostapenko/dev/mttl")  
from mttl.dataloader.human_dataset_readers import encode_with_messages_format

class ShardedIterator(IterableDataset):
    """A sharded wrapper around an iterable, padded to length.

    Args:
        iterable (iterable): iterable to wrap
        num_shards (int): number of shards to split the iterable into
        shard_id (int): which shard to iterator over
        fill_value (Any, optional): padding value when the iterable doesn't
            evenly divide *num_shards* (default: None).
    """

    def __init__(self, iterable, num_shards, shard_id, fill_value=None):
        if shard_id < 0 or shard_id >= num_shards:
            raise ValueError('shard_id must be between 0 and num_shards')

        self._sharded_len = len(iterable) // num_shards
        if len(iterable) % num_shards > 0:
            self._sharded_len += 1

        self.itr = itertools.zip_longest(
            range(self._sharded_len),
            itertools.islice(iterable, shard_id, len(iterable), num_shards),
            fillvalue=fill_value,
        )

    def __len__(self):
        return self._sharded_len

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.itr)[1]

        
class IterableDomain(IterableDataset):
    def __init__(self,
                domain_directory: Path=None,
                file: Path=None,
                add_bos_token: bool = False,
                bos_token: str = "<|endoftext|>",
                sample_files: int = None,
                track_token_count: bool = False,
                silent: bool=False,
                ignore_ids: Dict[str, int]={},
                text_field: str = "text",
                json: bool = True,
                path_to_clusters: Path=None
                ):
        if file is not None:
            self.files = file
            self.num_files = 1
        else:
            self.files = list(chain(domain_directory.glob(r"*.json.gz"),
                                    domain_directory.glob(r"*.jsonl.gz"),
                                    domain_directory.glob(r"*.txt.gz")))
            if sample_files:
                self.files = np.random.choice(self.files, sample_files)
            self.num_files = len(self.files)
        self.add_bos_token = add_bos_token
        self.track_token_count = track_token_count
        self.bos_token = bos_token
        self.domain_directory = domain_directory
        self.text_field = text_field
        self.ignore_ids = ignore_ids
        self.json = json
        self.path_to_clusters = path_to_clusters
        self.num_docs = 0
        self.count_lines()

    def count_lines(self):
        if torch.utils.data.get_worker_info() is not None:
            worker_total_num = torch.utils.data.get_worker_info().num_workers
            worker_id = torch.utils.data.get_worker_info().id
        else:
            worker_total_num = 1
            worker_id = 0
        for json_file in tqdm(itertools.islice(self.files,
                                               worker_id,
                                               None,
                                               worker_total_num),
                              total=self.num_files):
            if str(json_file).endswith(".gz"):
                file_itr = gzip.open(json_file, 'rb')
            else:
                file_itr = open(json_file, 'r')
            for ix in tqdm(file_itr):
                self.num_docs += 1

    def line_mapper(self, line):
        if self.json:
            sample = json.loads(line)
            try:
                sample.get(self.text_field)
            except:
                sample = {self.text_field : line}
        else:
            sample = {self.text_field: line}
        if not sample.get(self.text_field):
            return None, None, None
        token_count = len(sample[self.text_field].split())
        text = sample.pop(self.text_field)
        if self.add_bos_token:
            text = self.bos_token + " " + text
        return text, token_count, sample
    
    def __len__(self):
        return self.num_docs

    def __iter__(self):
        if torch.utils.data.get_worker_info() is not None:
            worker_total_num = torch.utils.data.get_worker_info().num_workers
            worker_id = torch.utils.data.get_worker_info().id
        else:
            worker_total_num = 1
            worker_id = 0
        if self.path_to_clusters:
            cluster_itr = np.load(self.path_to_clusters)
        for json_file in tqdm(itertools.islice(self.files,
                                                worker_id,
                                                None,
                                                worker_total_num),
                                disable=True,
                                total=self.num_files):
            if str(json_file).endswith(".gz"):
                file_itr = gzip.open(json_file, 'rb')
            else:
                file_itr = open(json_file, 'r')
            mapped_itr = map(self.line_mapper, file_itr)
            
            for ix, item in enumerate(mapped_itr):
                if item[0] is not None:
                    res = {'id': ix, 'file': str(json_file), 'text': item[0], 'token_count': item[1]}
                    if self.path_to_clusters:
                        res['cluster'] = cluster_itr[ix]
                    yield res


def load_model(path_to_model: Path):
    with open(path_to_model, 'rb') as f:
        out = pickle.load(f)
    return out


def transformer(vectorizer, dataset, num_parallel_shards, i):
    vecs = []
    iterator_ = ShardedIterator(dataset, num_parallel_shards, shard_id=i, fill_value = {})
    vecs = vectorizer.transform([x['text'] for x in iterator_ if x])
    return vecs


def vectorize_sklearn(model, file, batch_size=512, parallel=False, num_parallel_shards=10):
    dataset = IterableDomain(file=file)
    if parallel:
        import joblib
        from joblib import delayed
        vecs = joblib.Parallel(n_jobs=num_parallel_shards)(delayed(transformer)(model, 
                                                                                dataset,
                                                                                num_parallel_shards,
                                                                                i) 
                                                            for i  in tqdm(range(num_parallel_shards)))
        vecs = np.concatenate(vecs, 0)
    else:
        vecs = vectorizer.transform(tqdm([x['text'] for x in dataset]))
    return vecs


def get_files(data_dir):
    directory = Path(data_dir)
    files = list(directory.glob(r"*.jsonl"))
    print(directory)
    print(files)
    return files


def get_top_terms(vectorizer, kmeans):
    # this will only work if you use TFIDF vectorizer (which maintains vocab)
    original_space_centroids = vectorizer['svd'].inverse_transform(kmeans.cluster_centers.cpu())
    order_centroids = original_space_centroids.argsort()[:, ::-1]
    vocab = vectorizer['tfidf'].get_feature_names_out()
    top_terms = []
    for i in range(kmeans.n_clusters):
        terms = {}
        for j in range(10):
            terms[f'term_{j}'] = vocab[order_centroids[i, j]]
        top_terms.append(terms)
    return pd.DataFrame(top_terms)


def number_normalizer(tokens):
    """Map all numeric tokens to a placeholder.

    For many applications, tokens that begin with a number are not directly
    useful, but the fact that such a token exists can be relevant.  By applying
    this form of dimensionality reduction, some methods may perform better.
    """
    return ("#NUMBER" if token[0].isdigit() else token for token in tokens)


class NumberNormalizingVectorizer(TfidfVectorizer):
    # this vectorizer replaces numbers with #NUMBER token
    def build_tokenizer(self):
        tokenize = super().build_tokenizer()
        return lambda doc: list(number_normalizer(tokenize(doc)))



def train_kmeans(vecs, n_clusters, path_to_kmeans, balanced=False):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    kmeans = BalancedKMeans(n_clusters=n_clusters, device=device, balanced=balanced)

    batches = np.array_split(vecs, vecs.shape[0] // 50000, axis=0)

    for i, batch in tqdm(enumerate(batches)):
        kmeans.fit(torch.from_numpy(batch), iter_limit=20, online=True, iter_k=i)
    with open(path_to_kmeans,  'wb+') as f:
        _ = pickle.dump(kmeans, f)
    return kmeans




def eval(vectorizer, kmeans, data_dir):
    top_terms = get_top_terms(vectorizer, kmeans)
    files = get_files(data_dir)
    metadata = []
    for file in tqdm(files):
        metadata.append(pd.read_json(file, lines=True))
    metadata = pd.concat(metadata, axis=0)
    metadata = metadata.sample(min(10000, metadata.shape[0]))
    metadata['cluster'] = kmeans.predict(torch.from_numpy(vectorizer.transform(metadata.text)))
    return metadata, top_terms

def get_texts(dataset):
    # dataset = IterableDomain(file=dataset)
    # dataloader = DataLoader(dataset, num_workers=0, batch_size=512)
    domains = []
    for i,batch in tqdm(enumerate(dataset)):
        # for item in zip(batch['id'], batch['text']):
            # domains.append({"id": item[0].item(), "text": item[1]})
        domains.append({"id": i, "text": batch['instruction']+batch['input']})
    return pd.DataFrame(domains)

def get_texts_flan(dataset):
    # dataset = IterableDomain(file=dataset)
    # dataloader = DataLoader(dataset, num_workers=0, batch_size=512)
    domains = []
    for i,batch in tqdm(enumerate(dataset)):
        # for item in zip(batch['id'], batch['text']):
            # domains.append({"id": item[0].item(), "text": item[1]})
        sample = ""
        for c in batch['messages']:
            if c['role']=='user':   
                sample += c['content'] + " "
        domains.append({"id": i, "text": sample.strip()})
    return pd.DataFrame(domains)

# def train_vectorizer(file, path_to_vectorizer):
# english stopwords plus the #NUMBER dummy token        
stop_words = list(text.ENGLISH_STOP_WORDS.union(["#NUMBER"]))

model = Pipeline([('tfidf', NumberNormalizingVectorizer(stop_words=stop_words)),
                    ('svd', TruncatedSVD(n_components=100)),
                    ('normalizer', Normalizer(copy=False))])
  
# dataset = load_dataset("yahma/alpaca-cleaned")["train"]
# texts = get_texts(dataset)

   
datasets={
    "flan_v1": "/home/v-oostapenko/dev/open-instruct/data/processed/flan_v2/flan_v2_data.jsonl",
    # "CoT": "/home/v-oostapenko/dev/open-instruct/data/processed/cot/cot_data.json",  
    # "Dolly": "/home/v-oostapenko/dev/open-instruct/data/processed/dolly/dolly_data.json",
    # "oasst1": "/home/v-oostapenko/dev/open-instruct/data/processed/oasst1/oasst1_data.json",
}
# import json     
# from datasets import Dataset
# import pandas as pd  
dataset = load_dataset("json", data_files=datasets.values())["train"]
texts = get_texts_flan(dataset)

vecs = model.fit_transform(tqdm(texts.text))  
path_to_vectorizer = "/home/v-oostapenko/dev/mttl/inst_follow/cluster_infos/cbtm/tfidf_flv2.pkl"
with open(path_to_vectorizer,  'wb+') as f:
    _ = pickle.dump(model, f)

for n_clusters in [4,8,16,32]:
    # n_clusters=8
    balanced = False       
    path_to_kmeans = f"/home/v-oostapenko/dev/mttl/inst_follow/cluster_infos/kmeans_flan_v2{n_clusters}.pkl"
    kmeans = train_kmeans(vecs, n_clusters, path_to_kmeans, balanced=balanced)
    print("done")

# return vectorizer, kmeans

# return model, vecs


# def main(data_dir,
#          model='tfidf',
#          n_clusters=16,
#          balanced=False,
#          output_dir=Path("cluster_output/"),
#          kmeans_only=False):
# output_dir = "."
# # files = get_files(data_dir)
# dataset = load_dataset("yahma/alpaca-cleaned")["train"]
# if not output_dir.is_dir():
#     output_dir.mkdir(parents=True, exist_ok=True)
# path_to_vectorizer = output_dir / "tfidf.pkl"
# if not kmeans_only:
#     vectorizer, vecs = train_vectorizer(files, path_to_vectorizer)
# else:
#     vectorizer = load_model(path_to_vectorizer)
#     vecs = vectorize_sklearn(vectorizer, files, batch_size=10000, parallel=True)
# path_to_kmeans = output_dir / "kmeans.pkl"
# kmeans = train_kmeans(vecs, n_clusters, path_to_kmeans, balanced=balanced)
# return vectorizer, kmeans

# if __name__ == '__main__':

# parser = argparse.ArgumentParser() 
# parser.add_argument('--data-dir', required=True, type=str)
# parser.add_argument('--num-clusters', required=True, type=int)
# parser.add_argument('--balanced', action='store_true')
# parser.add_argument('--output-dir', required=True, type=Path)
# parser.add_argument('--eval-only', action='store_true')

# args = parser.parse_args()

# if not args.eval_only:
#     vectorizer, kmeans = main(data_dir=args.data_dir,
#                             n_clusters=args.num_clusters,
#                             balanced=args.balanced,
#                             output_dir=args.output_dir,
#                             kmeans_only=False)

# path_to_vectorizer = args.output_dir / "tfidf.pkl"
# path_to_kmeans = args.output_dir / "kmeans.pkl"
# vectorizer= load_model(path_to_vectorizer)
# kmeans = load_model(path_to_kmeans)
# metadata, top_terms = eval(vectorizer, kmeans, args.data_dir)
# print(top_terms)