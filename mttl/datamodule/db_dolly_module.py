import torch      
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from mttl.datamodule.ni_data_module import CollateWrapperFn, CollateWrapperFnCLM
from mttl.dataloader.alpaca_dataset_readers import AlpacaDataset
from mttl.dataloader.db_dolly_ds_reader import DB_DollyDataset
from transformers import LlamaTokenizer
from mttl.cluster_tuning.cluster_reader import ClusterResult

class DatBricksDollyModule(LightningDataModule):
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.config.train_batch_size,
            shuffle=True,
            num_workers=16,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=CollateWrapperFnCLM(self.pad_token_id),
        )

    def val_dataloader(self):
        return DataLoader(
            self.dev_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=False,
            num_workers=16,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=CollateWrapperFnCLM(self.pad_token_id),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.config.train_batch_size,
            shuffle=False,
            num_workers=16,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=CollateWrapperFnCLM(self.pad_token_id),
        )

    @property    
    def all_instructions(self):
        return self.dataset.read_all_instructions()

    def __init__(self, config, cluster_result: ClusterResult = None):
        super().__init__()

        self.config = config
        self.cluster_result = cluster_result
        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     "/home/v-oostapenko/llms/", #config.model, 
        #     model_max_length=config.max_input_length
        # )     
             
        tok_model = config.model if config.model is not None else "yahma/llama-7b-hf"
        self.tokenizer = LlamaTokenizer.from_pretrained(tok_model, add_eos_token=True) # tloen does not add eos token
        # self.tokenizer.pad_token_id = 
        self.tokenizer.pad_token_id = 0 
        
        if self.config.padding_side == "left":
            self.tokenizer.padding_side = "left"  # Allow batched inference, used by tloen also in training
        self.pad_token_id = self.tokenizer.pad_token_id

        self.task2id = {'alpaca_full':0}

    def get_dataset(self, idxs=None, loss_for_keywords=True):
        return DB_DollyDataset(
            
            self.tokenizer,           
            self.config.max_input_length, 
            self.config.max_output_length, 
            self.config.train_dir, 
            self.config.train_on_inputs,  
            self.config.dst_dir, idxs, self.cluster_result, self.config.predict_cluster, loss_for_keywords=loss_for_keywords,
        )
    
    def setup(self, stage=None): 
        idxs_cluster=[]                  
        if self.config.train_only_cluster is not None:
            assert self.cluster_result is not None
            idxs_cluster = np.where(np.array(self.cluster_result._instance.infos.cluster_dists)[:,self.config.train_only_cluster]==1)[0]
            # idxs_cluster = idxs_cluster.tolist()
        dataset = self.get_dataset()  # max_length 512 covers 97% of the dataset
        if not self.config.use_test_set: # default alpaca setting
            # always use the same split for the dataset
            rng = torch.Generator().manual_seed(1234)        
            if len(idxs_cluster)>0:
                dataset.dataset = dataset.dataset.select(idxs_cluster)
            n_tr_samples = int(len(dataset) * (1-self.config.validation_portion))   #len(dataset) - 
            self.train_dataset, self.dev_dataset = torch.utils.data.random_split(
                dataset, [n_tr_samples, len(dataset) - n_tr_samples, ], generator=rng
            )
            self.test_set = self.dev_dataset    
                
            print("Training steps:", len(self.train_dataloader()))
            print("Validation steps:", len(self.val_dataloader()))
        else:
            assert self.cluster_result is not None
            # use the is_test property to split the dataset 
            training_idxs = np.where(-1*(np.array(self.cluster_result.infos.is_test)-1))[0] 
            #find overlap with idxs_cluster
            if len(idxs_cluster)>0:     
                training_idxs = np.intersect1d(training_idxs, idxs_cluster)            
            test_idxs  = np.where(np.array(self.cluster_result.infos.is_test))[0] # test on all clusters togather
            
            self.test_set = self.get_dataset(test_idxs, loss_for_keywords=False)
            train_valid_set = self.get_dataset(training_idxs)
            rng = torch.Generator().manual_seed(1234)
            n_tr_samples = int(len(train_valid_set) * (1-self.config.validation_portion))                    
            self.train_dataset, self.dev_dataset = torch.utils.data.random_split(train_valid_set, [n_tr_samples, len(train_valid_set) - n_tr_samples, ], generator=rng)
            
    
    def get_per_cluster_test_loaders(self):
        pc_test_loaders=[]  
        test_idxs  = np.where(np.array(self.cluster_result.infos.is_test))[0] # test on all clusters togather
        for cluster_id in range(self.cluster_result.n_clusters()):
                cluster_idxs = np.where(np.array(self.cluster_result._instance.infos.cluster_dists)[:,cluster_id]==1)[0]
                cluster_idxs = np.intersect1d(cluster_idxs, test_idxs)
                pc_test_loaders.append(DataLoader(self.get_dataset(cluster_idxs),
                                                batch_size=self.config.train_batch_size,
                                                shuffle=False,
                                                num_workers=16,
                                                pin_memory=True,
                                                persistent_workers=True,
                                                collate_fn=CollateWrapperFnCLM(self.pad_token_id)))
        return pc_test_loaders


