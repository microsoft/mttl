import os 
import sys
import ast
import argparse
import json
import numpy as np
import torch
import wandb        
from string import Template
import pytorch_lightning as pl 
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))


from mttl.models.poly import get_selector       
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from projects.instr_routing.models.clm import CLM 
from mttl.callbacks import ProgressCallback  
from mttl.datamodule.alpaca_data_module import AlpacaDataModule
from mttl.datamodule.platypus_module import PlatypusModule
from mttl.datamodule.flan_module import FlanModule
from mttl.config import Config as MTTLConfig
from mttl.models.monitors import get_monitors 
from mttl.utils import get_mlf_logger         
from projects.instr_routing.models.modify_model import modify_transformer
from transformers import AutoModelForCausalLM, LlamaForCausalLM

from peft import prepare_model_for_int8_training

##################################################  
def number_normalizer(tokens):
    """Map all numeric tokens to a placeholder.

    For many applications, tokens that begin with a number are not directly
    useful, but the fact that such a token exists can be relevant.  By applying
    this form of dimensionality reduction, some methods may perform better.
    """
    return ("#NUMBER" if token[0].isdigit() else token for token in tokens)
from sklearn.feature_extraction.text import TfidfVectorizer        
class NumberNormalizingVectorizer(TfidfVectorizer):
    # this vectorizer replaces numbers with #NUMBER token
    def build_tokenizer(self):
        tokenize = super().build_tokenizer()   
        return lambda doc: list(number_normalizer(tokenize(doc)))       

def remove_non_serializable(d):
    """
    Recursively remove non-JSON serializable values from a dictionary.
    """
    for k, v in d.items():
        if isinstance(v, (list, dict)):
            remove_non_serializable(v)
        elif not json.dumps(v, default=lambda x: None):
            del d[k]


class Config(MTTLConfig):
    def __init__(self, **kwargs):
        self.rank = 1                
        self.load_dtype = "float32" 
        self.prune_unused_loras = True
        self.init_b_random = False
        self.lora_dropout = 0
        self.lora_alpha = 16 
        self.same_lora_init = 0
        self.load_in_8bit = False
        self.micro_batch_size = 4  
        self.share_lora_at_attn = 0
        self.share_lora_a  = False   
        self.merge_A_B_seperately = True
        self.train_on_inputs = False
        self.padding_side = "right"
        self.adapter_modules = None  
        self.poly_selector_use_distances = False
        self.adapter_layers = 0  # llama adapter
        self.adapter_len = 0  # llama adapter
        self.use_4_bit_backbone = False
        self.wandb_project = None
        self.switch_to_average = 0
        # self.balanced = 0
                 
        self.router_weight_decay = None
        self.param_names_added_to_sd = "" # define additional params that will be added to state dict additionally to the trainable ones.
        # self.validate_before_start = False 
        self.predict_cluster = None # topic or skill
        self.dst_dir = None # dir of jsonl dataset
        
        self.fast_dev_run = False
        self.fast_debug_run = False
        
        self.eval_ds_limit = 1
        self.train_only_cluster = None
        self.validation_portion = 0.03  
        self.per_cluster_test = False
        self.use_test_set = False # wether to use examples marked as is_test = 1 in ClusterInfo as test set
        
        self.aux_mi_loss_factor = 1
        
        # XRouter                 
        self.xrouter_load_balancing = False
        self.xrouter_x_cond = True 
        self.xrouting_option = 0 # only applies to x_router routing, depreciated 
        self.xrouter_normalize_weights = False
        self.xrouter_normalize_input = False
        self.xrouter_reverse_kl = False
        self.xrouter_normal_innit = True
        self.xrouter_use_attn = False
        self.xrouter_sim_metric = "kl"       
        self.xrouting_sep_teacher_student = False
        self.xrouter_init_scale = 0.02  
        self.xrouter_x4target_detach = True
        self.xr4_option = None #"switch" #, "default", "teacher_output"
        self.xrouter_x4_target = "posterior" #, "prior" -- wich router logits to use for x4 at trainign time.
        
        self.superni_eval_batchsize = 2
        self.router_learning_rate = None        
        self.eval_hellaswag = True
        self.eval_arc = True
        self.eval_truthfulqa = True
        self.eval_superni = True    
        self.eval_mmlu = True    
        self.eval_superni_use_outputs = False
        self.gen_alpaca_eval = False
        
        super().__init__(**kwargs)
        self._updated_kwargs = set()
        # to reproduce setup in https://github.com/daanelson/alpaca-lora
        self.gradient_accumulation_steps = (
            self.train_batch_size // self.micro_batch_size
        )
        self.train_batch_size = self.micro_batch_size
    
    def update_kwargs(self, kwargs, eval=True, raise_error=True):
        for (k, v) in kwargs.items():
            if eval:
                try:
                    v = ast.literal_eval(v)
                except (ValueError, SyntaxError):
                    v = v
            else:
                v = v
            if not hasattr(self, k) and raise_error:
                raise ValueError(f"{k} is not in the config")

            if eval:
                print("Overwriting {} to {}".format(k, v))

            if k == 'finegrained':
                k = 'poly_granularity'
                v = 'finegrained' if v else 'coarsegrained'
            elif k in ['train_dir', 'output_dir']:
                # this raises an error if the env. var does not exist
                v = Template(v).substitute(os.environ)

            setattr(self, k, v)   
            if isinstance(self._updated_kwargs, set):
                self._updated_kwargs.add(k)
            else:
                self._updated_kwargs[k]=v


def parse_config(extra_kwargs=None, raise_error=True, parent=None, return_parser=False, c=None):
    import itertools

    # dont do it if called from jupyter notebook
    if c is None:
        parser = (
            argparse.ArgumentParser(parents=[parent])
            if parent
            else argparse.ArgumentParser()
        )
        parser.add_argument("-c", "--config_files", required=False)
        parser.add_argument("-k", "--kwargs", nargs="*", action="append")
        args = parser.parse_args()
    else:
        args = argparse.Namespace()
        args.kwargs = None
        args.config_files = c
    kwargs = {}
    if args.kwargs:
        kwargs_opts = list(itertools.chain(*args.kwargs))
        for value in kwargs_opts:
            key, _, value = value.partition("=")
            kwargs[key] = value

    args.kwargs = kwargs
    if extra_kwargs:  
        args.kwargs.update(extra_kwargs)

    config = Config(
        filenames=args.config_files, kwargs=args.kwargs, raise_error=raise_error
    )

    print(config.to_json())
    if return_parser:
        return config, args
    return config




def run_multitask(args):             
    seed_everything(args.seed, workers=True)
    # get directory of the current file
    print(os.path.dirname(os.path.realpath(__file__)))
    cluster_result = None
    if args.example_to_ids_path:
        raise NotImplementedError()
        from mttl.cluster_tuning.cluster_reader import ClusterResult

        cluster_result = ClusterResult(args.example_to_ids_path)
        args.n_tasks = cluster_result.n_clusters()

        if args.poly_selector in ["cluster_soft", "cluster_hard"]:
            args.n_skills = cluster_result.n_clusters()
        else:
            # raise NotImplementedError()
            args.n_skills = 1
    
    
    if args.train_only_cluster is not None:
            # we only want to train on a single cluster, hence model qith a single skill
            args.n_skills = 1          
            skill_ids_to_keep = np.where((np.array(cluster_result._instance.infos.cluster_dists).sum(0))> 0)[0]     
            cluster_result.remove_skills(skill_ids_to_keep)
            
            
    # select dataloader
    model_class = CLM      
    if args.dataset == "alpaca":                
        dm = AlpacaDataModule(args, cluster_result=cluster_result)
    elif args.dataset == "flan_v2":
        dm = FlanModule(args, cluster_result=cluster_result)
    elif args.dataset == "human":
        dm = FlanModule(args, cluster_result=cluster_result)
    elif args.dataset == "platypus":
        dm = PlatypusModule(args, cluster_result=cluster_result)
    else:
        raise NotImplementedError()
      
    args.n_tasks = len(dm.task2id)  
    if args.load_dtype == "float32":
        load_dtype = torch.float32
    elif args.load_dtype == "float16":
        load_dtype = torch.float16
    else:
        raise NotImplementedError()
    if "llama" in args.model:    
            args.model_object =LlamaForCausalLM.from_pretrained(
                args.model,    
                load_in_8bit=args.load_in_8bit, # this doesnt work right now with current implementatio of lora
                torch_dtype=load_dtype,
                device_map="auto",
            )
    else:     
        args.model_object = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto") 
         
    if args.model_object.config.vocab_size != len(dm.tokenizer): #if adding [EOI] in LongForm dataset
        args.model_object.resize_token_embeddings(len(dm.tokenizer))
    if args.load_in_8bit:
        args.model_object = prepare_model_for_int8_training(args.model_object)
      
    args.model_object = modify_transformer(args.model_object, args)

    if args.checkpoint is not None:         
        import copy       
        from mttl.utils import get_checkpoint_path
        from projects.instr_routing.utils import load_model        
        checkpoint_path = get_checkpoint_path(args.checkpoint)
        kwargs = copy.deepcopy(args)
        # kwargs.pop("checkpoint")  
        #remove attribute checkpoint from kwargs
        # if hasattr(kwargs, "checkpoint"):
        #     delattr(kwargs, "checkpoint")
        # module = model_class.load_from_checkpoint(
        #     checkpoint_path, **kwargs, tokenizer=dm.tokenizer
        # )           
        module, _, _ = load_model(kwargs, model_path=checkpoint_path)
        module.args.output_dir = args.output_dir
        # add XRouter if needed       
        if args.poly_selector in ["x_router", "x_router_hard"]:        
            args.n_skills = module.args.n_skills    
            module.model.set_selector(args, selector_to_replace=get_selector(module.args).__class__, new_selector = get_selector(args).__class__)
            # freeze the experts    
            # args.trainable_param_names=".*selector.*"  
            module.args = args        
            
        del args.model_object
        
    else:
        module = model_class(**vars(args), tokenizer=dm.tokenizer)
        del args.model_object

    if args.poly_selector in ["cluster_soft", "cluster_hard"]:
        if args.n_skills > 1 and args.prune_unused_loras:
            # prune unused loras
            counts = m = np.bincount(cluster_result._instance.infos.cluster_ids)
            # skill_ids_to_keep = np.where(
            #     np.bincount(cluster_result._instance.infos.cluster_ids) > 0
            # )[0]                                 
            skill_ids_to_keep = np.where((np.array(cluster_result._instance.infos.cluster_dists).sum(0))> 0)[0]        
            module.model.remove_skills(skill_ids_to_keep)
            cluster_result.remove_skills(skill_ids_to_keep)
                
    if args.switch_to_average > 0:  
        module.model.switch_selector_to_average(selector_to_replace=get_selector(args).__class__)


    # legit logging
    loggers = []    
    if os.environ.get("WANDB_API_KEY") or args.wandb_project:
        # args_=args.__dict__.copy()
        # remove_non_serializable(args_)   
        project = "alpaca_tuning_ncb" if args.wandb_project is None else args.wandb_project
        project = os.environ.get("WANDB_PROJECT", project)
        project+=f"_{args.dataset}"
        wandb_logger = pl.loggers.WandbLogger(
            project=project,
            name=os.environ.get("AMLT_JOB_NAME", args.exp_name),  # , config=args_
        )
        wandb_logger.experiment.save("*.py")
        loggers.append(wandb_logger)
    else:
        wandb_logger = None

    mlf_logger = get_mlf_logger()
    if mlf_logger:
        loggers.append(mlf_logger)

    loggers.append(pl.loggers.CSVLogger(save_dir=args.output_dir, name="csv_metrics"))

    kwargs = {"val_check_interval": args.eval_every} if args.eval_every else {}

    # get metric monitors for models
    callbacks = get_monitors(args)
    callbacks.append(ProgressCallback())

    monitor = "val/loss"
    mode = "min"
     
    model_name = args.model.replace("/", "_")
    # check if wandb run exists
    if wandb_logger:
        # get run id
        run_id = wandb_logger.experiment.id
        model_name += run_id
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        monitor=monitor,
        filename=f"{model_name}" + f"_{args.exp_name}" + "-{" + monitor + ":.004f}",
        save_top_k=1,
        save_last=True,
        save_weights_only=True,  # make checkpoints smaller
        mode=mode,
    )
    callbacks.append(checkpoint_callback)

    trainer = Trainer(
        gpus=1,
        accelerator="gpu",
        logger=loggers,      
        num_sanity_val_steps=5,
        amp_backend="native",    
        default_root_dir=args.output_dir,
        max_epochs=args.num_train_epochs,   
        max_steps=args.total_steps + 1 if args.total_steps != -1 else -1,
        gradient_clip_val=args.max_grad_norm,
        log_every_n_steps=20,    
        strategy=args.compute_strategy if args.compute_strategy else None,
        callbacks=callbacks,    
        accumulate_grad_batches=args.gradient_accumulation_steps,
        precision=int(args.precision)
        if args.precision in ["16", "32"]
        else args.precision,
        fast_dev_run = args.fast_dev_run,
        **kwargs,
    )
    
    trainer.fit(module, dm)

    # try:    
    ckpt_path="best" if not args.fast_dev_run else None   
    trainer.validate(dataloaders=dm, ckpt_path=ckpt_path)
    if args.use_test_set:  
       module.model.checkpoint_tested="best"  
       trainer.test(dataloaders=dm, ckpt_path=ckpt_path)       
       module.model.checkpoint_tested="last"
       trainer.test(dataloaders=dm, ckpt_path="last")   
    # if args.dataset in ["ni", "xfit", "alpaca", "longform", "wizzerd", "db_dolly"]:
    path_best_model = trainer.checkpoint_callback.best_model_path
    path_last_model = trainer.checkpoint_callback.last_model_path
    
    print(f"Best model path: {path_best_model}")
    print(f"Last model path: {path_last_model}")
    # empty memory
    del module
    del dm
    del trainer
    # empty cache  

          
    ds_limit = args.eval_ds_limit if not args.fast_debug_run else 0.05
    
    torch.cuda.empty_cache()

    if args.eval_superni:         
        print("#"*50)           
        print("Evaluating on super NI")              
        from projects.instr_routing.eval.ni.gen_ni_predictions import eval_superni
        rouge_L_super_ni = eval_superni(model_name="",    
                     batch_size=args.superni_eval_batchsize,
                     out_prefix=f"{args.exp_name}",  
                     model_path=path_best_model,          
                     nshot=0, use_outputs=args.eval_superni_use_outputs, ds_limit=ds_limit)
        if wandb.run is not None:           
            wandb.log({"rouge_L_super_ni": rouge_L_super_ni})
    
    if args.eval_mmlu:       
        # from projects.instr_routing.eval.lm_eval_harness.run_eval import eval_lm   
        from projects.instr_routing.eval.mmlu.run_mmlu_eval import eval_mlu
        print("#"*50)
        print("Evaluating on MMLU")     
        acc=eval_mlu(ntrain=5, model_name="",model_path=path_best_model, eval_batch_size=3)
        # results_dict=eval_lm(model_path=path_best_model, model_name="", task="mmlu", batch_size=5, nshot=0, ds_limit=ds_limit)
        if wandb.run is not None:           
            # wandb.log(results_dict)
            wandb.log({"mmlu_acc": acc})
         
    if args.eval_arc:       
        from projects.instr_routing.eval.lm_eval_harness.run_eval import eval_lm
        print("#"*50)   
        print("Evaluating on ARC")       
        results_dict=eval_lm(model_path=path_best_model, model_name="", task="arc_challenge", batch_size=5, nshot=25, ds_limit=ds_limit)
        if wandb.run is not None:           
            wandb.log(results_dict)

    if args.eval_truthfulqa: 
        from projects.instr_routing.eval.lm_eval_harness.run_eval import eval_lm
        print("#"*50)         
        print("Evaluating on TruthfulQA")
        results_dict=eval_lm(model_path=path_best_model, model_name="", task="truthfulqa_mc", batch_size=5, nshot=0, ds_limit=ds_limit)
        if wandb.run is not None: 
            wandb.log(results_dict)         
    
    
    if args.eval_hellaswag:
        from projects.instr_routing.eval.lm_eval_harness.run_eval import eval_lm
        print("#"*50)
        print("Evaluating on HellaSwag")           
        results_dict=eval_lm(model_path=path_best_model, 
                             model_name="", task="hellaswag", 
                             batch_size=5, nshot=10, ds_limit=ds_limit)
        if wandb.run is not None:           
            wandb.log(results_dict)
    
    # if args.gen_alpaca_eval:
    #     print("Generting alpaca_eval")     
    #     from projects.instr_routing.eval.alpaca_eval.gen_alpaca_eval_predictions import gen_alpaca_evl
    #     try:       
    #         gen_alpaca_evl(   
    #             llama_model=args.model,
    #             batch_size=2,           
    #             model_name=args.exp_name, 
    #             model_path=path_best_model, 
    #             run_all_clusters=False)
            
    #     except Exception as e:
    #         # print e
    #         print("Failed to generate alpaca eval")
    #         print(e)

        
if __name__ == "__main__":
    args = parse_config()
    run_multitask(args)
