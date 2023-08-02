import os
import argparse
import json
import numpy as np
import torch
#import partial
from functools import partial
import pytorch_lightning as pl 
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from inst_follow.models.clm import CLM
from mttl.callbacks import ProgressCallback
from mttl.datamodule.alpaca_data_module import AlpacaDataModule
from mttl.datamodule.alpaca_self_improvement import AlpacaSelfImprovement
from mttl.datamodule.longform_data_module import LongFormDataModule
from mttl.datamodule.wizzard_data_module import WizzardDataModule
from mttl.models.encoder_decoder import EncoderDecoder
from mttl.models.t0_encoder_decoder import T0EncoderDecoder
from finetune_llama import Config as MTTLConfig
from finetune_llama import parse_config
from mttl.models.monitors import get_monitors
from mttl.utils import get_mlf_logger, from_pretrained
from mttl.models.modify_model import modify_transformer
from transformers import LlamaForCausalLM, LlamaTokenizer
from mttl.dataloader.data_utils import ExampleInfo
from mttl.datamodule.ni_data_module import CollateWrapperFn, CollateWrapperFnCLM
from mttl.utils import get_ni_tasks_from_file, trim_batch, hash_example
from typing import List

# from peft import prepare_model_for_int8_training

def format(examples):
    out=""
    for example in examples:
        out+=f"\n###Input: {example['input']}\nOutput: {example['output']}\n"
    out+="\n###Input:"
    return out

def general_context_understanding():
    in_context_samples = [
            {'input': "Given this context:Generate mathematical experssions like 2+2=4.", 'output': "2+2=4"},
            {'input': "Calculate the result of the mathematical expression: 2+10.", 'output': "12"},
            # {'input': "Which mathematical operations do you know?", 'output': "I know how to add, subtract, multiply and divide."},
        ]
    return format(in_context_samples)
class Trainer(Trainer): 
    def generate_new_samples(self,module:CLM, dm:AlpacaSelfImprovement): 
        if not hasattr(dm, "train_dataset"):
            dm.setup()
        in_context_samples = dm.sample(2)
        # in_context_samples = [
        #     {'input': "Generate mathematical experssions like 2+2=4.", 'output': "2+2=4"},
        #     {'input': "Calculate the result of the mathematical expression: 2+10.", 'output': "12"},
        #     # {'input': "Which mathematical operations do you know?", 'output': "I know how to add, subtract, multiply and divide."},
        # ]
        # construct a batch, 
        prompt = f"Generate 5 examples of instruction - response pairs in math domain. Here are some examples:\n"
        for i, sample in enumerate(in_context_samples):
            # prompt += "\n ###Start of a sample: \n"
            prompt += f"\nInstruction: {sample['instruction']}\n"
            if len(sample['input'])>0:
                prompt += f"\nInput: {sample['input']}\n"
            prompt += f" \nResponse: {sample['output']}\n"
            # prompt += "\n ###End of a sample.\n"
        # prompt += "\n ###Start of a sample: \n"
        # prompt = format(in_context_samples)
        
        import copy        
        tokenizer =  LlamaTokenizer.from_pretrained("yahma/llama-7b-hf", padding_side='left')   
        tokenizer.pad_token_id = 0 #tokenizer.eos_token_id

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(module.model.device)
        labels = input_ids.clone()
        task_ids = -1     
        hashes = hash_example(prompt)
        instruction_hashes = hashes
        
        collated_example = CollateWrapperFnCLM(tokenizer.pad_token_id)([ExampleInfo(input_ids, labels, task_ids, hashes, instruction_hashes)])
        
        response = module.generate(collated_example,
                                    do_sample=True,
                                    temperature=0.7, 
                                    max_new_tokens=2000,     
                                    return_dict_in_generate=True)
        response_str = tokenizer.batch_decode(response.sequences, skip_special_tokens=True)
     
# Phase 1: generate general problems for model to undertand the context.


# os.environ["AP_DATA_DIR"] = "/home/v-oostapenko/data" # temp for gcr
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
        super().__init__(**kwargs)

def run_multitask(args):             
    seed_everything(args.seed, workers=True)
    # get directory of the current file
    print(os.path.dirname(os.path.realpath(__file__)))
    cluster_result = None
    if args.example_to_ids_path:
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
            
            
    # select dataloader
    model_class = CLM      
    dm = AlpacaSelfImprovement(args, cluster_result=cluster_result)

    args.n_tasks = len(dm.task2id)  
    if args.use_4_bit_backbone:
        args.model_object = from_pretrained(
            args.model)   
            # max_memory_MB=80000,  
            # add_lora_f = partial(modify_transformer, config=args))
    else:
        args.model_object =LlamaForCausalLM.from_pretrained(
            args.model,
            # load_in_8bit=args.load_in_8bit, # this doesnt work right now with current implementatio of lora
            # torch_dtype=torch.float16,
            device_map="auto",
        )  # , load_in_8bit=True, torch_dtype=torch.float16)  
    
    
    if args.model_object.config.vocab_size != len(dm.tokenizer): #if adding [EOI] in LongForm dataset
        args.model_object.resize_token_embeddings(len(dm.tokenizer))
    args.model_object = modify_transformer(args.model_object, args)
    # if args.load_in_8bit:
    #     args.model_object = prepare_model_for_int8_training(args.model_object)

    if args.checkpoint is not None:         
        from mttl.utils import get_checkpoint_path
        from inst_follow.utils import load_model        
        checkpoint_path = get_checkpoint_path(args.checkpoint)
        kwargs = vars(args)
        kwargs.pop("checkpoint")    
        # module = model_class.load_from_checkpoint(
        #     checkpoint_path, **kwargs, tokenizer=dm.tokenizer
        # )           
        module, _, _ = load_model(kwargs, model_path=checkpoint_path)
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



    # legit logging
    loggers = []
    if os.environ.get("WANDB_API_KEY"):
        # args_=args.__dict__.copy()
        # remove_non_serializable(args_)
        wandb_logger = pl.loggers.WandbLogger(
            project="alpaca_tuning" if args.wandb_project is None else args.wandb_project,
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
    
            
    new_examples = trainer.generate_new_samples(module, dm)

    trainer.fit(module, dm)

    # try:    
    ckpt_path="best" if not args.fast_dev_run else None
    if args.use_test_set:  
       module.model.checkpoint_tested="best"  
       trainer.test(dataloaders=dm, ckpt_path=ckpt_path)       
       module.model.checkpoint_tested="last"
       trainer.test(dataloaders=dm, ckpt_path="last")   
    
       
       
       
       
       
       
       
       
       
       
    if args.dataset in ["ni", "xfit", "alpaca", "longform", "wizzerd"]:
        best_model_path = trainer.checkpoint_callback.best_model_path
        last_model_path = trainer.checkpoint_callback.last_model_path
        # last_model_path = trainer.checkpoint_callback.bes
        print(f"Best model path: {best_model_path}")
        # Rename the file at best_model_path to 'best_model'
        path_best_model = best_model_path.split("loss=")[0] + f"best_model_{args.exp_name}_loss=" + best_model_path.split("loss=")[1]
        path_last_model = last_model_path.split("last")[0] + f"last_model_{args.exp_name}_v" + last_model_path.split("last")[1]
        #args.output_dir + f"/best_model_{args.exp_name}"
        # create dir if not exists
        if not os.path.exists(path_best_model):
            os.makedirs(path_best_model)
        # copy the best model to the best_model dir
        os.system("mv " + best_model_path + " " + path_best_model + "/")
        os.system("mv " + last_model_path + " " + path_last_model + "/")
        print(f"Best model path: {path_best_model}")
        print(f"Last model path: {path_last_model}")


if __name__ == "__main__":
    args = parse_config()
    run_multitask(args)
