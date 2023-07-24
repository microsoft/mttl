import argparse
import os
import numpy as np
import pandas as pd
import pickle
# import submitit
import torch
import uuid
import sys              
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm.auto import tqdm
from kmeans_pytorch import KMeans
from datasets import load_dataset
sys.path.append("/home/v-oostapenko/dev/mttl")    
from mttl.utils import hash_example
from mttl.dataloader.alpaca_dataset_readers import AlpacaTemplateForHash
from sklearn.feature_extraction.text import TfidfVectorizer
# from metaseq.data import JsonlDataset
# from metaseq.cbtm_constants import DEFAULT_SLURM_ACCOUNT, DEFAULT_SLURM_CONSTRAINT, DEFAULT_SLURM_PARTITION

def get_shard_str(epoch, data_dir, split):
    shards = {}
    for shard_id in os.listdir(os.path.join(data_dir, split)):
        assert (
            int(shard_id) not in shards
        ), f"shard id: {shard_id} not in shards: {shards}"
        shards[int(shard_id)] = shard_id
    assert min(shards.keys()) == 0
    assert max(shards.keys()) == len(shards) - 1
    cur_shard_str = shards[(epoch - 1) % len(shards)]
    return cur_shard_str

       
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

def example_in_cluster(text, vectorizer, kmeans, random_clusters=False, distances = False):
    if distances:
        from kmeans_pytorch import pairwise_distance
        centers = kmeans.cluster_centers
        distances_to_centers = pairwise_distance(torch.from_numpy(vectorizer.transform(text)).cuda(), centers)
        return list(distances_to_centers)
    if random_clusters:       
        clusters = np.random.choice(range(kmeans.n_clusters), len(text))
    else:
        clusters = kmeans.predict(torch.from_numpy(vectorizer.transform(text)))
    return list(clusters)
    

def cluster_file(file, data_dir, split, cur_shard_str, tfidf, kmeans, num_clusters, output_prefix):
    path = os.path.join(data_dir, split, cur_shard_str, file.split('/')[-1])
    output_dir = os.path.join(output_prefix, split, cur_shard_str)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output = output_dir + f"/{file.split('/')[-1]}"
    dataset = JsonlDataset(
                path=path,
                tokenizer=None,
                include_path_infos_in_jsonl_dataset=True
            )
    dataloader = DataLoader(dataset, batch_size=10000, num_workers=0, collate_fn=_collate_fn)
    zs = []
    counter = 0
    for batch in tqdm(dataloader):
        text = [x['item']['text'] for x in batch]
        ids = [x['sp_id'] for x in batch]
        cluster = example_in_cluster(text,  tfidf, kmeans, random_clusters=False)
        zs.extend([{"sp_id": x, "cluster": y.item()} for x,y in zip(ids, cluster)])
        counter += 1
    df = pd.DataFrame(zs)
    df.to_json(output, lines=True, orient='records')

                   
def load_model(path_to_model):
    with open(path_to_model, 'rb') as f:
        out = pickle.load(f)
    return out


def get_shared_folder() -> Path:
    user = os.getenv("USER")
    if Path("/tmp/").is_dir():
        p = Path(f"/tmp/")
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")


def get_init_file():
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


# if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--job-dir")
    # parser.add_argument("--data-dir")
    # parser.add_argument("--path-to-clusterer")
    # parser.add_argument("--num-clusters")
    # parser.add_argument("--output-prefix")
    # parser.add_argument("--split")
    # parser.add_argument('--slurm-partition', type=str, default=DEFAULT_SLURM_PARTITION)
    # parser.add_argument('--slurm-account', type=str, default=DEFAULT_SLURM_ACCOUNT)
    # parser.add_argument('--slurm-constraint', type=str, default=DEFAULT_SLURM_CONSTRAINT)

    # parser.add_argument('--run', type=str, default='slurm', choices=['slurm', 'local'])

    # cmd_args = parser.parse_args()
# executor = submitit.AutoExecutor(folder=cmd_args.job_dir, slurm_max_num_timeout=30)

num_gpus_per_node = 1
nodes = 1
timeout_min = 60
distance=True
kwargs = {}    
path_to_clusterer = "/home/v-oostapenko/dev/mttl/inst_follow/cluster_infos/cbtm/"
# path_to_clusterer = Path(cmd_args.path_to_clusterer)
# num_clusters=8 
for num_clusters in [4,8, 16, 32]:        
    clusterer=f"kmeans_flan_v2{num_clusters}"        
    kmeans = load_model(path_to_clusterer+f"/{clusterer}.pkl")
    tfidf = load_model(path_to_clusterer+"/tfidf_flv2.pkl")

    def flatten(ls):
        return [y for x in ls for y in x]

    def _collate_fn(items):
        return items

    # dataset = load_dataset("yahma/alpaca-cleaned")["train"]      
    
    sys.path.append("/home/v-oostapenko/dev/mttl")  
    from mttl.dataloader.human_dataset_readers import encode_with_messages_format
    from datasets import load_dataset
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
    dataloader = DataLoader(dataset, batch_size=10000, num_workers=0, collate_fn=_collate_fn)
    zs = []  
    counter = 0         
    from transformers import LlamaForCausalLM, LlamaTokenizer       
    tok_model = "yahma/llama-7b-hf"       
    tokenizer = LlamaTokenizer.from_pretrained(tok_model, add_eos_token=True)
    max_input_length = 512 # has no impact here
    for i,batch in tqdm(enumerate(dataloader)):
        # text = [x['instruction']+x['input'] for x in batch]   
        text = []
        for x in batch:
            text_c=""
            for c in x['messages']:
                if c['role']=='user':         
                    text_c += c['content'] + " " 
            text.append(text_c)     
                
        # ids = [hash_example(AlpacaTemplateForHash().apply(example)) for example in batch]          
        ids = [hash_example(str(example['messages'])) for example in batch]                 
        cluster = example_in_cluster(text,  tfidf, kmeans, random_clusters=False, distances=distance)
        if distance:
            zs.extend([{"sp_id": x, "cluster": y.tolist()} for x,y in zip(ids, cluster)])
        else:
            zs.extend([{"sp_id": x, "cluster": y.item()} for x,y in zip(ids, cluster)])
        counter += 1
    df = pd.DataFrame(zs)
    output=path_to_clusterer+f"/clustered_{num_clusters}_flnv2.json"
    if distance: 
        output=path_to_clusterer+f"/clustered_{num_clusters}_distances_flnv2.json"
    df.to_json(output, lines=True, orient='records')