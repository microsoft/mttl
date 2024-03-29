import json
import os
import torch

from mttl.config import Config
from mttl.utils import logger


class ExpertConfig(Config):
    def _set_defaults(self):
        super()._set_defaults()

        self.load_in_8bit = False
        self.wandb_project = None
        self.tensorboard = False

        self.remote_token = None
        self.library_id = None
        self.destination_library_id = None

        self.do_train = True

        # just a lame flag to 0 out all adapter weights
        self.baseline = False
        # sparsify adapter weights to this sparsity level
        self.sparsity = 0.0
        # only use a very small portion of the available experts
        self.subsample_library_experts = 0
        # rank / retrieve top k experts
        self.ranker_top_k = 1
        self.ranker_path = None
        self.ranker_model = None

        self.expert_name = None
        self.routing = "subject"
        self.mmlu_test_split = "test"
        self.load_module = None
        self.micro_batch_size = None
        self.validation_portion = 0.03

        self.use_instruct_template = False
        self.source_template = None
        self.augment_few_shot = 0

        self.subsample_train = None
        self.subsample_dev = None

        self.moe_num_experts = 8
        self.moe_emb_dim = 128
        self.moe_rkhs_dim = 512
        self.moe_ent_reg = 0.0
        self.moe_ent_free_bits = 0.0
        self.moe_top_k = None

        self.data_dir = os.getenv("AMLT_DATA_DIR", "~/data/")
        self.output_dir = os.getenv("AMLT_OUTPUT_DIR", "tmp/instruction_learning/")

        self.mmlu_use_hard_prompt = None  # use a hard prompt for mmlu

        self.eval_mmlu_few_shot = True  # use few-shot for mmlu, default
        self.eval_mmlu_flag = False  # eval mmlu performance during training
        self.eval_rouge_flag = False  # eval rouge during training
        self.pipeline_eval_tasks = "all"

        self.eval_metric = "loss"
        self.use_vllm = False

        # for finetuning a library
        self.hf_repo_query = (
            None  # for retrieval, we take query expert from this library
        )
        self.sk = 5  # number of experts to retrieve from a library
        self.finetune_regime = None  # polylib_full, lib_mu, polylib_selector
        self.eval_before_training = True

        # hidden state computation transform
        self.use_base_model_only = False
        self.max_samples_per_task = 100
        self.track = "each_layer"
        self.pool = "last"

        # Ties Merging
        self.transform_sparsity = 1.0

        # Per Token Router [Arrow, Phatgoose, hidden]
        self.router_temp = None
        self.notes = None
        self.proto_init = None  # also "arrow"
        self.input_norm_fn = None
        self.proto_norm_fn = None
        self.ab_only = True
        self.base_model_proto = False  # compute Arrow embeddings for the backbone
        self.tie_op = "concat"  # or "sum"

        # Eval Library
        self.merge_or_route = None  # "uniform", "ties", "clown"
        self.tasksets_path = None
        self.remove_experts = None
        self.create_transfer_matrix = False
        self.es_metric = "loss"
        self.n_ng_iterations = 30  # number of iterations for LoraHub
        self.recompute_prototypes = False
        self.expert_embeds_save_name = None

        self.phi_2_align_heads = False
        self.lora_merge_after = False  # if True, tried to merge after the outer product, currently only applicable to LoRA

        # phatgoose gate learning
        self.n_steps_pg = 2000
        self.learning_rate_pg = 0.01

        self.save_each_epoch = False
        self.add_eos_to_downstream_targets = True

    def post_init(self, silent=False):
        if self.micro_batch_size is None:
            self.micro_batch_size = self.train_batch_size

        self.gradient_accumulation_steps = (
            self.train_batch_size // self.micro_batch_size
        )
        self.train_batch_size = self.micro_batch_size

        n_devices = torch.cuda.device_count()
        if n_devices > 1:
            logger.warn(
                "You have multiple GPUs, but your device count is not being taken "
                + "into account when computing `gradient_accumulation_steps`."
            )

        if self.finetune_task_name is not None and isinstance(
            self.finetune_task_name, str
        ):
            # resolve task keys
            task_names = []
            tasks = self.finetune_task_name.split(",")
            if self.tasksets_path is not None and os.path.exists(self.tasksets_path):
                # load task names from json file
                task_sets = json.load(open(self.tasksets_path))
                for task_name in tasks:
                    # try to fetch task_names from the file
                    if task_name in task_sets:
                        task_names.extend(task_sets[task_name])
                    else:
                        task_names.append(task_name)
            else:
                task_names = tasks

            self.finetune_task_name = ",".join(task_names)
