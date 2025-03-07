import os
import sys
import torch
import copy
import wandb
import numpy as np
from copy import deepcopy
import torch.nn.functional as F
from pytorch_lightning import seed_everything
import json

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from mttl.models.library.expert_library import ExpertLibrary
from mttl.models.modifiers.expert_containers.selectors import (
    PerTokenSelector,
    Selector,
    SelectorConfig,
)
from mttl.models.modifiers.lora import LoRAConfig

from mttl.logging import setup_logging, logger
from mttl.utils import remote_login  #logger, setup_logging
from mttl.models.expert_model import MultiExpertModel, ExpertModel
from mttl.arguments import ExpertConfig

from mttl.evaluators.base import EvaluatorRunner, setup_evaluators
from mttl.models.modifiers.expert_containers.library_transforms import (
    ABWeightedLinearMerge,
    ABWeightedLinearMergeConfig,
    WeightedLinearMerge,
    WeightedLinearMergeConfig,
    HiddenStateComputer,
    HiddenStateComputerConfig,
    TiesMerge,
    TiesMergeConfig,
    ArrowTransform,
    ArrowConfig,
    PhatgooseTransform,
    PhatgooseConfig,
    SparseWeightLinearMerge,
    SparseWeightLinearMergeConfig, 
)

from mttl.models.lightning.callbacks import LossCallback
from mttl.datamodule.base import get_datamodule
from mttl.evaluators.rouge_evaluator import RougeEvaluator
from projects.wiki_experts.src.utils.utils import TableLogger


def get_hidden_states(library, args):
    cfg = HiddenStateComputerConfig(
        use_base_model_only=args.use_base_model_only,
        max_samples_per_task=args.max_samples_per_task,
        name=args.expert_embeds_save_name,
        track=args.track,
        pool=args.pool,
    )
    output = HiddenStateComputer(cfg).transform(
        library, recompute=args.recompute_prototypes, default_args=args
    )

    return output


def get_arrow_embeddings(library, args):
    cfg = ArrowConfig(
        name=args.expert_embeds_save_name,
        ab_only=args.ab_only,
        tie_params=args.tie_params,
        tie_op=args.tie_op,
    )
    return ArrowTransform(cfg).transform(
        library,
        recompute=args.recompute_prototypes,
        add_base_proto=args.base_model_proto,
    )


def get_phatgoose_embeddings(library, args):
    phatgoose_transform = PhatgooseTransform(
        PhatgooseConfig(
            n_steps=args.n_steps_pg,
            learning_rate=args.learning_rate_pg,
            name=args.expert_embeds_save_name,
        )
    )
    return phatgoose_transform.transform(
        library, default_args=args, recompute=args.recompute_prototypes
    )


def patch_prototypes(module, library, args, proto_inits=None):
    if not proto_inits and args.router_selector == "arrow_router":
        proto_inits = get_arrow_embeddings(library, args)
    elif not proto_inits and args.router_selector == "avg_act_router":
        proto_inits = get_hidden_states(library, args)
    elif not proto_inits and args.router_selector == "phatgoose_router":
        proto_inits = get_phatgoose_embeddings(library, args)

    for mod in module.modules():
        if isinstance(mod, PerTokenSelector):
            prototypes = []
            patched_layer_name = mod.layer_name.replace(".selector", "")
            for expert_name in mod.expert_names:
                layer_names = proto_inits[expert_name].keys()
                valid_layer_names = [
                    k
                    for k in layer_names
                    if patched_layer_name in k  # k.startswith(patched_layer_name)
                ]
                assert len(valid_layer_names) <= 2, breakpoint()
                key = sorted(valid_layer_names)[0]
                proto = proto_inits[expert_name][key]
                if isinstance(proto, np.ndarray):
                    proto = torch.from_numpy(proto)

                prototypes += [proto.squeeze()]

            logger.info(
                f"setting prototypes for selector at {mod.layer_name} with hidden states from {key}"
            )
            prototypes = torch.stack(prototypes)
            mod.overwrite_prototypes(prototypes)


def eval_in_distribution(module, args: ExpertConfig, tasks: list):
    args.include_task_source = "*"
    transfer_table = TableLogger()
    print(f'eval metric: {args.eval_metric}')

    for i, task in enumerate(tasks):
        args.finetune_task_name = task
        args.predict_batch_size = 16
        if args.eval_metric in ["val_loss", "loss"]:
            dm = get_datamodule(args)
            evaluator = LossCallback(
                dm.val_dataloader(), output_dir=args.output_dir, name=task + "_val"
            )
            metric = evaluator.test(pl_module=module).item()

        elif args.eval_metric == "test_loss":
            dm = get_datamodule(args)
            evaluator = LossCallback(
                dm.test_dataloader(), output_dir=args.output_dir, name=task + "_test"
            )
            metric = evaluator.test(pl_module=module).item()
        elif args.eval_metric == "val_rougeL":
            dm = get_datamodule(args, for_generation=True)
            evaluator = RougeEvaluator(
                datamodule=dm,
            )
            metric = evaluator.evaluate(
                module,
                split="val",
                verbose=False,
            )
        elif args.eval_metric == "rougeL":
            dm = get_datamodule(args, for_generation=True)
            evaluator = RougeEvaluator(
                datamodule=dm,
            )
            metric = evaluator.evaluate(
                module,
                split="test",
                verbose=False,
            )
        else:
            raise ValueError(f"Unknown eval metric {args.eval_metric}")
        if wandb.run is not None:
            wandb.log({f"test/{args.eval_metric}_{task}": metric})
        transfer_table.log({"task": task, args.eval_metric: metric})

    if wandb.run is not None:
        wandb.log(
            {f"mean_{args.eval_metric}": transfer_table.df[args.eval_metric].mean()}
        )

    transfer_table.log(
        {
            "task": "mean",
            args.eval_metric: transfer_table.df[args.eval_metric].mean(),
        }
    )
    transfer_table.log_final_table()

def eval_in_distribution_sparse_model(module, library, expert, args: ExpertConfig, tasks: list):
    args.include_task_source = "*"
    transfer_table = TableLogger()

    for i, task in enumerate(tasks):
        # update the mask correspond to the task
        expert.update_module_mask(module, library[task])

        args.finetune_task_name = task
        args.predict_batch_size = 16
        if args.eval_metric in ["val_loss", "loss"]:
            dm = get_datamodule(args)
            evaluator = LossCallback(
                dm.val_dataloader(), output_dir=args.output_dir, name=task + "_val"
            )
            metric = evaluator.test(pl_module=module).item()

        elif args.eval_metric == "test_loss":
            dm = get_datamodule(args)
            evaluator = LossCallback(
                dm.test_dataloader(), output_dir=args.output_dir, name=task + "_test"
            )
            metric = evaluator.test(pl_module=module).item()
        elif args.eval_metric == "val_rougeL":
            dm = get_datamodule(args, for_generation=True)
            evaluator = RougeEvaluator(
                datamodule=dm,
            )
            metric = evaluator.evaluate(
                module,
                split="val",
                verbose=False,
            )
        elif args.eval_metric == "rougeL":
            dm = get_datamodule(args, for_generation=True)
            evaluator = RougeEvaluator(
                datamodule=dm,
            )
            metric = evaluator.evaluate(
                module,
                split="test",
                verbose=False,
            )
        else:
            raise ValueError(f"Unknown eval metric {args.eval_metric}")
        if wandb.run is not None:
            wandb.log({f"test/{args.eval_metric}_{task}": metric})
        transfer_table.log({"task": task, args.eval_metric: metric})

    if wandb.run is not None:
        wandb.log(
            {f"mean_{args.eval_metric}": transfer_table.df[args.eval_metric].mean()}
        )

    transfer_table.log(
        {
            "task": "mean",
            args.eval_metric: transfer_table.df[args.eval_metric].mean(),
        }
    )
    transfer_table.log_final_table()
def run_eval(args: ExpertConfig):
    seed_everything(args.seed, workers=True)

    # get directory of the current file
    setup_logging(args.output_dir)

    logger.info("Args: {}".format(args.to_json()))

    remote_login(args.remote_token)


    # args.library_id = 'local://sample_lib/phi_3/lora/'
    # args.library_id = 'local://sample_lib/phi_3/sparse/'
    # args.K = None # default is None of 256
    # args.merge_or_route = "uniform_sparse_weight"   # for sparse weight merge
    # args.pipeline_eval_tasks = "in_distribution"
    
    
    #args.K = None # default is None of 256

    #args.library_id = 'local://library/phi_3_finetune_0winit_sparse_kr_0.002/'
    #args.merge_or_route = "uniform_sparse_weight"   # for sparse weight merge

    #args.library_id = 'local://library/phi_3_finetune_lora/'
    #args.merge_or_route = "uniform"   # for sparse weight merge

    # args.library_id = 'local://library/phi_3_finetune_0winit_noisy_sparse_kr_0.002/'
    # args.library_id = 'local://library/phi_3_finetune_0.002_nr_0.02/'
    # args.merge_or_route = "uniform_sparse_weight"   # for sparse weight merge
    
 
    # args.pipeline_eval_tasks = "in_distribution"

    
    selection = None
    # -------------------- TODO remove this section ---------------------
    #selection = ['duorc_ParaphraseRC_extract_answer']
    # selection = ['duorc_ParaphraseRC_extract_answer',
    #               'wiki_qa_Topic_Prediction_Question_and_Answer_Pair',
    #               'duorc_ParaphraseRC_build_story_around_qa',
    #               'cos_e_v1_11_description_question_option_id','adversarial_qa_droberta_answer_the_following_q']
    # selection = ['wiqa_what_is_the_final_step_of_the_following_process', 'sciq_Multiple_Choice', 'adversarial_qa_droberta_answer_the_following_q', 'duorc_SelfRC_question_answering', 'cos_e_v1_11_description_question_option_id', 'wiki_qa_Is_This_True_', 'quail_description_context_question_text', 'wiki_hop_original_explain_relation', 'duorc_ParaphraseRC_build_story_around_qa', 'yelp_polarity_reviews_0_2_0', 'squad_v1_1_3_0_0', 'web_questions_potential_correct_answer', 'quoref_Found_Context_Online', 'quoref_Given_Context_Answer_Question', 'web_questions_get_the_answer', 'cot_sensemaking', 'wiki_qa_Topic_Prediction_Question_and_Answer_Pair', 'dbpedia_14_given_list_what_category_does_the_paragraph_belong_to', 'duorc_ParaphraseRC_extract_answer', 'super_glue_rte_1_0_2']
    # --------------------
    if selection is None:
        exclude_phi_tasks = [
            "hellaswag_1_1_0",
            "ai2_arc_ARC_Challenge_1_0_0",
            "ai2_arc_ARC_Easy_1_0_0",
            "piqa_1_0_0",
            "winogrande_1_1_0",
            "bool_q_1_0_0",
            "openbookqa_0_1_0",
        ]
    else: exclude_phi_tasks = None
    # exclude_phi_tasks=['duorc_ParaphraseRC_build_story_around_qa',
    #                     'duorc_SelfRC_question_answering',
    #                     'cot_sensemaking',
    #                     'adversarial_qa_droberta_answer_the_following_q',
    #                     'yelp_polarity_reviews_0_2_0',
    #                     'duorc_ParaphraseRC_extract_answer']

    print(args.library_id)
    library = ExpertLibrary.get_expert_library(
        repo_id=args.library_id,
        token=args.remote_token,
        exclude_selection=exclude_phi_tasks,
        destination_id=args.destination_library_id,
        selection=selection,
        K=args.K
    )
    an_expert = library[next(iter(library.keys()))]
    train_cfg = deepcopy(an_expert.training_config)
    train_cfg.device_map = "cpu"
    # For starts, always overwrite the following arguments
    for arg_name in [
        "output_dir",
        "eval_metric",
        "remove_phi_eval_tasks",
        "include_task_source",
    ]:
        value = getattr(args, arg_name, None)
        setattr(train_cfg, arg_name, value)

    """ Parameter Merging Approaches """
    if args.merge_or_route in ["uniform", "ties"]:
        if args.merge_or_route == "uniform":
            module = WeightedLinearMerge(WeightedLinearMergeConfig()).transform_module(library).to("cuda")
            # expert = WeightedLinearMerge(WeightedLinearMergeConfig()).transform(library)
            # module = MultiExpertModel(**vars(expert.training_config)).to("cuda")
            # module.add_expert_instance(expert, is_default=True)
            #module.load_from_module_dict(expert._expert_weights)
            
        elif args.merge_or_route == "ties":
            cfg = TiesMergeConfig(top_k=args.transform_sparsity)

            # default: causes error at `module.add_expert_instance`
            # expert = TiesMerge(cfg).transform(library)
            # module = MultiExpertModel(**vars(expert.training_config)).to("cuda")
            # module.add_expert_instance(expert, is_default=True)

            module = TiesMerge(cfg).transform_module(library).to("cuda")


    elif args.merge_or_route == "SLERP":
        from mttl.models.modifiers.expert_containers.library_transforms import SLERPMerge, SLERPMergeConfig
        module = SLERPMerge(SLERPMergeConfig()).transform(library).to("cuda")

    elif args.merge_or_route == "uniform_lora_before_op":
        module = ABWeightedLinearMerge(ABWeightedLinearMergeConfig()).transform(library).to("cuda")

    elif args.merge_or_route == "uniform_sparse_weight":
        """uncomment to do a uniform merge of all weights"""
        #module = SparseWeightLinearMerge(SparseWeightLinearMergeConfig()).transform(library).to("cuda")
        expert = SparseWeightLinearMerge(SparseWeightLinearMergeConfig())
        module = expert.transform(library).to("cuda")

        """masked weight for single task"""
        # TODO: remove provide weights of only one task
        #expert = SparseWeightLinearMerge(SparseWeightLinearMergeConfig())
        #expert_names = list(library.keys())   #TODO remove
        #module = expert.transform_dummy(library, expert_names[0]).to("cuda") #TODO remove
        

    elif args.merge_or_route == "uniform_lora_after_op":
        # Here we merge the LoRA experts after the outer product we cannot really do it
        # with the lib transform, cause this would require storing large matrices in memory
        # Instead we do it with a uniform selector
        assert type(an_expert.expert_info.expert_config) == LoRAConfig
        train_cfg.router_selector = "uniform"
        train_cfg.lora_merge_after = True
        module = MultiExpertModel(**vars(train_cfg)).to("cuda")
        module.load_from_module_dict(library)

    elif args.merge_or_route == "base":
        module = ExpertModel(**vars(train_cfg)).to("cuda")

    elif args.merge_or_route in ["phatgoose", "arrow", "avg_act"]:
        """Routing Approaches"""
        args.router_selector = f"{args.merge_or_route}_router"

        selector_config = SelectorConfig.from_training_config(args)

        module = MultiExpertModel(
            **vars(train_cfg), selector_config=selector_config
        ).to("cuda")
        module.load_from_module_dict(library)
        patch_prototypes(module, library, args)

    elif args.merge_or_route == "oracle":
        """TaskNameSelector"""
        args.router_selector = "task_selector"
        selector_config = SelectorConfig.from_training_config(args)

        module = MultiExpertModel(
            **vars(train_cfg), selector_config=selector_config
        ).to("cuda")
        module.load_from_module_dict(library)
    else:
        raise ValueError(f"Unknown merge_or_route {args.merge_or_route}")

    metric_logger = Selector.metric_logger

    if wandb.run is None and os.environ.get("WANDB_API_KEY"):
        # wandb.init(
        #     project=os.environ.get("WANDB_PROJECT", "0shot_routing"),
        #     config=dict(module.hparams),
        #     name=os.environ.get("AMLT_JOB_NAME", None),
        # )
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "0shot_routing"),
            #project=os.environ.get("WANDB_PROJECT", "0shot_routing_lora_merge_check"),
            # project=os.environ.get("WANDB_PROJECT", "0shot_routing_vary_K_random_rerun"),
            config=dict(module.hparams),
            name=os.environ.get("AMLT_JOB_NAME", None),
        )

        # update config
        wandb.config.update({f"cmd_args_{k}": v for k, v in vars(args).items()})


    #expert.get_experts(library)

    if args.pipeline_eval_tasks in [
        "in_distribution",
    ]:
        tasks = [expert.expert_task_name for expert in library.data.values()]

        # TODO: remove the following 2 lines after in-distrbution experiment
        #import random
        #tasks = random.sample(tasks, 20)
        #tasks = random.sample(tasks, 2)
        
        if tasks[0] is None:
            # for some older version of lib (in case of joint experts) no expert_task_name was set
            tasks = json.load(open(args.flan_tasks_path))["flan256"]
        # make sure we evaluate each task seperately (so the mean is over tasks at the end)
        tasks = ",".join(tasks).split(",")
        train_cfg.eval_metric = args.eval_metric
        train_cfg.subsample_dev = args.subsample_dev


        #tasks = [expert_names[0]]    #TODO: remove
        #scores = eval_in_distribution_sparse_model(module, library, expert, train_cfg, tasks) #TODO: adds sparse mask
        scores = eval_in_distribution(module, train_cfg, tasks)
    
    elif args.pipeline_eval_tasks in [
        "out_distribution",
    ]:
        # give eval tasks in `finetune_task_name` argument
        if isinstance(args.finetune_task_name,tuple):
            tasks = list(args.finetune_task_name)
        elif isinstance(args.finetune_task_name,str):
            tasks = args.finetune_task_name.split(",")

        train_cfg.eval_metric = args.eval_metric
        train_cfg.subsample_dev = args.subsample_dev
        #tasks = [expert_names[0]]    #TODO: remove
        #scores = eval_in_distribution_sparse_model(module, library, expert, train_cfg, tasks) #TODO: adds sparse mask
        scores = eval_in_distribution(module, train_cfg, tasks)
    
    
    else:
        if args.pipeline_eval_tasks == "all":
            args.pipeline_eval_tasks = "arc-challenge,arc-easy,boolq,hellaswag,humaneval,mbpp,openbookqa,piqa,bbh-fast,winogrande"

        with torch.no_grad():
            runner: EvaluatorRunner = setup_evaluators(
                model_type=module.hparams.model,
                model_family=module.hparams.model_family,
                max_input_length=module.hparams.max_input_length,
                max_output_length=module.hparams.max_output_length,
                predict_batch_size=args.predict_batch_size,
                truncation_side=module.hparams.truncation_side,
                tasks=args.pipeline_eval_tasks,
                output_path=os.path.join(args.output_dir, "DOWNSTREAM"),
                add_eos_to_targets=args.add_eos_to_downstream_targets,
            )
            scores = runner.run(module)

    if len(metric_logger) > 0:
        task_table = metric_logger.pretty_table(match_on="task|.*uniform.*")
        layer_table = metric_logger.pretty_table(match_on="layer|.*uniform.*")
        expert_p = metric_logger.pretty_table(match_on=".*expert_p|.*uniform.*")
        angle = metric_logger.pretty_table(match_on=".*angle.*")
        print(task_table)
        print(layer_table)
        print(expert_p)
        print(angle)

    if wandb.run is not None:
        if scores is not None:
            wandb.log({f"downstream/{k}": v for k, v in scores.items()})
        if len(metric_logger) > 0:
            wandb.log({k: v.avg for k, v in metric_logger.meters.items()})

        wandb.finish()





def compare_svd(args: ExpertConfig):
    seed_everything(args.seed, workers=True)
    # get directory of the current file
    setup_logging(args.output_dir)
    logger.info("Args: {}".format(args.to_json()))
    remote_login(args.remote_token)

    library = ExpertLibrary.get_expert_library(
    repo_id=args.library_id,
    token=args.remote_token,
    exclude_selection=None,
    destination_id=args.destination_library_id,
    selection=None,
    K=args.K)


    expert_names = list(library.keys())
    experts = [library[name] for name in expert_names]



    train_cfg = deepcopy(experts[0].training_config)
    train_cfg.device_map = "cpu"

    from mttl.models.expert_model import ExpertModel
    train_cfg.model_modifier = None
    base_model = ExpertModel(**vars(train_cfg))


    weight_names = [n for n in base_model.model.state_dict().keys() if ( 'qkv_proj.weight' in n)]

    from sklearn.metrics.pairwise import cosine_similarity
    for _, expert in zip(expert_names[1:], experts[1:]):
        for w_name,b_w in base_model.model.state_dict().items():

            if w_name in weight_names[-10:]:

                l = '.'.join(w_name.split('.')[:-1])
                lora_ab = torch.matmul(expert.expert_weights[f'{l}.lora_a'],expert.expert_weights[f'{l}.lora_b']).T
                e_w = b_w + lora_ab

                u_b,s_b,_ = np.linalg.svd(b_w.to('cpu').to(torch.float32))
                u_e,s_e,_ = np.linalg.svd(e_w.to('cpu').to(torch.float32))
                score_self = cosine_similarity(u_b,u_b)
                score_compare = cosine_similarity(u_b,u_e)

                for threshold in [1e-10, 1e-5, 1e-3, 1e-2, 1e-1]:
                    # Effective rank: Count singular values greater than threshold * largest_singular_value
                    largest_singular_value = s_b[0]
                    effective_rank_base = np.sum(s_b > threshold * largest_singular_value)

                    #largest_singular_value = s_e[0]
                    effective_rank_expert = np.sum(s_e > threshold * largest_singular_value)
                    print(f'{w_name} {threshold} base | expert :', effective_rank_base, effective_rank_expert)

    print('done')

if __name__ == "__main__":
    args = ExpertConfig.parse()
    run_eval(args)
    #compare_svd(args)
