



from typing import List
import pickle
import tqdm
import os
import time
import torch       
from mttl.config import parse_config 
from nomic import atlas, AtlasProject
import numpy as np 
import argparse
import openai
from transformers import AutoTokenizer, AutoModel
from mttl.datamodule.alpaca_data_module import AlpacaDataModule, AlpacaDataset
from mttl.cluster_tuning.encodings import ClusterInfos
from finetune_llama import parse_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def store_embeddings(embeddings, args):
    global embeddings_path
    with open(embeddings_path, 'wb') as f:
            pickle.dump({"embeddings":embeddings, "meta_data":{"embedding_model":args.embedding_model,"cluster_with":args.cluster_with}}, f)

def get_embeddings(documents):   
    if args.embedding_model=='open_ai':
        batch_size= 128
        embeddings = []   
        for i in tqdm.tqdm(range(0, len(documents), batch_size), desc='Computing embeddings'):
            batch_instructions = documents[i:i+batch_size]
            response = openai.Embedding.create(       
                            input=batch_instructions,    
                            engine="text-embedding-ada-002")
            embeddings.extend([np.array(d['embedding']) for d in response.data])
        embeddings = np.stack(embeddings)
        return embeddings        
    else:  
        model = AutoModel.from_pretrained(args.embedding_model)#.to(device)
        tokenizer = AutoTokenizer.from_pretrained(args.embedding_model)#.to(device)
        embeddings = [] 
        with torch.no_grad():
            batch_size = 128 # lower this if needed
            for i in tqdm.tqdm(range(0, len(documents), batch_size)):    
                batch = [document for document in documents[i:i+batch_size]]   
                encoded_input = tokenizer(batch, return_tensors='pt', padding=True)
                cls_embeddings = model(**encoded_input)['last_hidden_state'][:, 0]
                embeddings.append(cls_embeddings)

        embeddings = torch.cat(embeddings).numpy()
        return embeddings

def get_atlas_map(dm:AlpacaDataModule, args):
    def create_new_proj():    
        print("Creating new project and map")
        dataset:AlpacaDataset = dm.get_dataset()
        print(" Getting embeddings...")
        embeddings = get_embeddings([d[args.cluster_with] for d in dataset.dataset])
        # save embeddings to a file
        store_embeddings(embeddings, args)
        instructions_dicts = [{"full":f"Instruction: {d['instruction']} \n Input: {d['input']} \n Output: {d['output']}", 
                               "instruction":d["instruction"], 
                               "input": d['input'], 
                               "output": d["output"]} for d in dataset.dataset]
        project = atlas.map_embeddings(data=instructions_dicts, #[d[args.cluster_with] for d in instructions_dicts], #performs bag of words based topic modeling
                            # indexed_field=args.cluster_with,
                            embeddings=embeddings,
                            name=f'Alpaca_{args.cluster_with}',
                            reset_project_if_exists=True,     
                            topic_label_field=args.cluster_with,
                            build_topic_model=True
                          )
        print(" waiting for project to be ready...")
        #wait until project is ready
        while not project.is_accepting_data:
            time.sleep(5)
        
        return project           
    if not args.rebuild_embeddings:
        try:  
            project = AtlasProject(name=f'Alpaca_{args.cluster_with}')
            map = project.get_map(name=f'Alpaca_{args.cluster_with}')    
        except Exception as e:
            print(e) 
            project = create_new_proj()
            map = project.get_map(name=f'Alpaca_{args.cluster_with}')
    else: 
        project = create_new_proj()
        map = project.get_map(name=f'Alpaca_{args.cluster_with}')
    return map

def main(args, config):      
    dm = AlpacaDataModule(config)
    dm.setup()
    map = None
    # # load embeddings from GPT
    global embeddings_path   
    embeddings_path = args.embeddings_path+f"/embeddings_of_{args.cluster_with}_2.pkl"        
    if args.use_atlas:   
        map = get_atlas_map(dm, args) if map is None else map
        topics_all = map.get_topic_data() 
        # try loading embeddings file
        if os.path.exists(embeddings_path):
            with open(embeddings_path, 'rb') as f:
                embeddings_file = pickle.load(f)
        else:            
            raise Exception("Embeddings file not found, try running the script with --rebuild_embeddings flag set to True")
        # remove key from embeddings_file
        depth = 2
        emb_column_name = f"atlas_topics_by_{args.cluster_with}_l{depth}"
        if not emb_column_name in embeddings_file:
            embeddings_file[emb_column_name]=[]
            batch_size = 28 # lower this if needed   
            for r in tqdm.tqdm(range(0, len(embeddings_file["embeddings"]), batch_size)):
                batch = embeddings_file["embeddings"][r:r+batch_size]
                # q = np.expand_dims(r, axis=0)
                topics = map.vector_search_topics(queries=batch, depth=depth)['topics']     
                embeddings_file[emb_column_name].extend(topics)
            #save responses as pkl 
            with open(embeddings_path, 'wb') as f:
                pickle.dump(embeddings_file, f)
        
        cluster_infos = ClusterInfos() 
        indices_train = dm.train_dataset.indices  
        print("Getting cluster infos for train set")
        for i,example in tqdm.tqdm(enumerate(dm.train_dataset)):   
            topics = embeddings_file[emb_column_name][indices_train[i]]
            hash = example.hash
            probs = np.zeros((len(topics_all)))
            for i, k in enumerate(topics):
                probs[int(k)-1]=topics[k] 
            assert sum(probs)==1
            main_t = np.argmax(probs)
            cluster_infos.is_test.extend([0])   
            cluster_infos.task_names.extend([topics_all[int(main_t)]["topic_short_description"]])
            cluster_infos.cluster_ids.extend([int(main_t)])    
            cluster_infos.hashes.extend([hash])
            cluster_infos.cluster_dists.extend([probs.tolist()])
        
        print("Getting cluster infos for dev set")
        indices_dev = dm.dev_dataset.indices
        for i,example in tqdm.tqdm(enumerate(dm.dev_dataset)): 
            topics = embeddings_file[emb_column_name][indices_dev[i]]
            hash = example.hash
            probs = np.zeros((len(topics_all)))
            for i, k in enumerate(topics):
                probs[int(k)-1]=topics[k] 
            assert sum(probs)==1
            main_t = np.argmax(probs)
            cluster_infos.is_test.extend([1]) 
            cluster_infos.task_names.extend([topics_all[int(main_t)]["topic_short_description"]])
            cluster_infos.cluster_ids.extend([int(main_t)])    
            cluster_infos.hashes.extend([hash])
            cluster_infos.cluster_dists.extend([probs.tolist()])
            
        cluster_infos.save(args.example_to_ids_path)
    else:        
        raise NotImplementedError
        # TODO: double check this
        import faiss
        if not args.rebuild_embeddings:
            if os.path.exists(embeddings_path):
                with open(embeddings_path, 'rb') as f:
                    embeddings_file = pickle.load(f)
            else:            
                raise Exception("Embeddings file not found, try running the script with --rebuild_embeddings flag set to True")
        else:            
            dataset:AlpacaDataset = dm.get_dataset()
            embeddings = get_embeddings([d[args.cluster_with] for d in dataset.dataset])
            # save embeddings to a file                
            store_embeddings(embeddings, args)
        
        n_clusters = args.n_clusters 
        embedings = np.stack(embeddings_path["embedings"])
        # for k in tqdm.tqdm(k_range):
        kmeans = faiss.Kmeans(
            embedings.shape[-1],
            k=n_clusters, 
            niter=10,
            verbose=True,
            gpu=True,
            nredo=5,            
            max_points_per_centroid=10_000_000,
        )  
        kmeans.train(embedings)

        cluster_infos = ClusterInfos() 
        D, I = kmeans.index.search(embedings, n_clusters)   
        cluster_infos.centroids=kmeans.centroids
        # Inference:
        # kmeans.train(logits, init_centroids=centroids) # this ensures that kmeans.index is created
        # assert np.sum(kmeans.centroids - centroids) == 0, "centroids are not the same" # sanity check
        # cluster_distances, cluster_indices = kmeans.assign(new_data)
        distances = np.zeros((D.shape[0], D.shape[1]))
        for i in range(D.shape[0]):
            for j in range(D.shape[1]):
                distances[i, I[i, j]] = D[i, j]             
        indices_train = dm.train_dataset.indices  
        for i,example in tqdm.tqdm(enumerate(dm.train_dataset)):
            hash = example.hash
            idx = indices_train[i]
            cluster_infos.hashes.append(hash)
            cluster_infos.cluster_ids.append(I[idx, 0])
            cluster_infos.is_test.append(0)
            cluster_infos.cluster_dists.append(distances[idx].tolist())
            # assert chunk_data.input_type == cluster_infos.input_type
        
        indices_dev = dm.dev_dataset.indices
        for i,example in tqdm.tqdm(enumerate(dm.dev_dataset)):
            hash = example.hash
            idx = indices_dev[i]    
            cluster_infos.hashes.append(hash)
            cluster_infos.cluster_ids.append(I[idx, 0])
            cluster_infos.is_test.append(1)
            cluster_infos.cluster_dists.append(distances[idx].tolist())

        assert len(cluster_infos.hashes) == len(cluster_infos.cluster_ids)
        cluster_sizes = np.bincount(cluster_infos.cluster_ids)

        print("Sorted cluster sizes:", sorted(cluster_sizes))
        print(
            "Bigger to smaller ratio:",
            np.max(cluster_sizes) / (np.min(cluster_sizes) + 0.1),
        )
        
        cluster_infos.save(config.example_to_ids_path)

if __name__ == "__main__":
    # add params with default values
    config = parse_config()      
    parser = argparse.ArgumentParser()         
    parser.add_argument("--cluster_with", type=str, default="instruction", choices=["full", "instruction", "output"])
    parser.add_argument("--n_clusters", type=int, default=20) 
    parser.add_argument("--use_atlas", type=bool, default=True)     
    parser.add_argument("--use_topic_modeling", type=bool, default=False)  # if True uses topic modeling for lcustering, if False uses OpenAI embeddings
    parser.add_argument("--embeddings_path", type=str, default="/home/v-oostapenko/dev/mttl/inst_follow/data/self_instruct_GPT3_embeddings")
    parser.add_argument("-c", "--config", type=str, default="config/mttl/mttl.yaml")
    parser.add_argument("--example_to_ids_path", type=str, default="/home/v-oostapenko/dev/mttl/inst_follow/data/cluster_infos/atlas_by_instr_text-embedding-ada-002_ldalayer2.pkl")
    parser.add_argument("--rebuild_embeddings",  type=bool, default=False) 
    parser.add_argument("--embedding_model",  type=str, default='open_ai') 
    args = parser.parse_args()
    main(args, config)