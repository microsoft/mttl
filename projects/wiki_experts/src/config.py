import os
import torch

from mttl.config import Config
import mttl.datamodule.task_sequences
import mttl.datamodule.task_cluster_flan
from mttl.utils import logger


class ExpertConfig(Config):
    def _set_defaults(self):
        super()._set_defaults()

        self.load_in_8bit = False
        self.wandb_project = None
        self.tensorboard = False
        self.hf_token_hub = None
        self.hf_lib_id = None
        self.hf_repo_id = None
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
        self.module_graph = None
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
        self.moe_top_k = -1

        self.expand_val_set_w_downstream = False

        self.eval_mmlu_callbacks_every = 0
        self.eval_test_set_callback_every = 0
        self.eval_rougeL_callback_every = 0
        self.test_sets_callbacks = []

        self.use_custom_valid_callback = False  # if True use custom callback to early stop on eval loss instead of lightning callback

        self.data_dir = os.getenv("AMLT_DATA_DIR", "~/data/")
        self.output_dir = os.getenv("AMLT_OUTPUT_DIR", "tmp/instruction_learning/")

        self.mmlu_use_hard_prompt = None  # use a hard prompt for mmlu

        self.eval_mmlu_few_shot = True  # use few-shot for mmlu, default
        self.eval_mmlu_flag = False  # eval mmlu performance during training
        self.eval_rouge_flag = False  # eval rouge during training
        self.pipeline_eval_tasks = None

        self.eval_metric = "loss"
        self.use_vllm = False
        self.reset_lr = False
        self.reset_optim = False

        # for finetuning a library
        self.hf_repo_query = (
            None  # for retrieval, we take query expert from this library
        )
        self.sk = 5  # number of experts to retrieve from a library
        self.finetune_regime = None  # polylib_full, lib_mu, polylib_selector
        self.library_to_expert_transform = None
        self.eval_before_training = True

    def post_init(self):
        if self.micro_batch_size is None:
            self.micro_batch_size = self.train_batch_size

        # to reproduce setup in https://github.com/daanelson/alpaca-lora
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
            tasks = self.finetune_task_name.split(
                "+"
            )  # use "+" for assign multiple task set vars to be found in task_sequences
            for task_name in tasks:
                if task_name in mttl.datamodule.task_sequences.__dict__:
                    task_names.extend(
                        getattr(mttl.datamodule.task_sequences.__dict__, task_name)
                    )
                elif task_name in mttl.datamodule.task_cluster_flan.__dict__:
                    task_names.extend(
                        getattr(mttl.datamodule.task_cluster_flan, task_name)
                    )
                else:
                    task_names.extend([task_name])
            self.finetune_task_name = ",".join(task_names)
