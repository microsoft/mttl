import copy
import gc
import math
import os
import shutil
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import vllm
from lightning_fabric import seed_everything
from transformers import AutoModelForCausalLM
from transformers.trainer import (
    TRAINER_STATE_NAME,
    TRAINING_ARGS_NAME,
    ExportableState,
    OptimizerNames,
    ParallelMode,
    TrainerState,
    TrainingArguments,
    TrainOutput,
    get_model_param_count,
    is_accelerate_available,
    is_apex_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_xla_available,
    speed_metrics,
)

from mttl.datamodule.utils import maybe_filter_hf_dataset_by_task
from mttl.models.get_scheduler import get_scheduler

if is_accelerate_available():
    from accelerate import Accelerator
    from accelerate import __version__ as accelerate_version
    from accelerate import skip_first_batches
    from accelerate.utils import (
        DistributedDataParallelKwargs,
        DistributedType,
        GradientAccumulationPlugin,
        load_fsdp_model,
        load_fsdp_optimizer,
        save_fsdp_model,
        save_fsdp_optimizer,
    )


if is_apex_available():
    from apex import amp


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    from torch_xla import __version__ as XLA_VERSION

from datasets import DatasetDict
from km_datamodule import KMDatasetConfig, KMDatasetModule
from torch import distributed as dist
from torch import functional as F
from torch import nn
from torch.utils.data import DataLoader
from train_km import DCDTrainer

from mttl.arguments import ExpertConfig
from mttl.datamodule.base import DataModule, DatasetConfig, get_datamodule
from mttl.logging import logger, setup_logging
from mttl.models.expert_model import ExpertModel, ExpertModelConfig
from mttl.models.get_optimizer import get_optimizer_and_scheduler
from mttl.utils import remote_login


@dataclass
class TextDatasetConfig(DatasetConfig):
    text_column: str = "text"


@DataModule.register("text_dataset", config_cls=TextDatasetConfig)
class TextDatamodule(DataModule):
    """Just a datamodule that loads a dataset with a 'text' column."""

    def collate_fn(self, examples):
        return None

    def setup_dataset(self):
        from mttl.models.library.dataset_library import DatasetLibrary

        dataset = DatasetLibrary.pull_dataset(self.config.dataset)

        if self.config.text_column not in dataset["train"].column_names:
            raise ValueError(f"The dataset must contain a `{self.config.text}` column")

        (
            self._task_names,
            self._task_to_id,
            self.train_dataset,
            _,
            _,
        ) = maybe_filter_hf_dataset_by_task(
            dataset,
            self.config.task_name_field,
            self.config.finetune_task_name,
        )

        self.dev_dataset = None
        self.test_dataset = None


class IterativeDCDTrainer(DCDTrainer):
    """Iterative DCD trainer that generates new datasets and distills the model in each epoch."""

    def __init__(self, model, args, **kwargs):
        super().__init__(model, args, **kwargs)

        self.temp_directory = f"/tmp/{np.random.randint(0, 1e6)}"
        self.current_epoch_datamodule = None
        self.reference_model = model

    def _inner_training_loop(
        self,
        batch_size=None,
        args=None,
        resume_from_checkpoint=None,
        trial=None,
        ignore_keys_for_eval=None,
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size

        logger.debug(
            f"Currently training with a batch size of: {self._train_batch_size}"
        )
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        total_train_batch_size = (
            self._train_batch_size * args.gradient_accumulation_steps * args.world_size
        )

        len_dataloader = None
        num_train_tokens = None

        len_dataloader = len(train_dataloader)
        num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        num_examples = self.num_examples(train_dataloader)

        if args.max_steps > 0:
            max_steps = args.max_steps
            num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                args.max_steps % num_update_steps_per_epoch > 0
            )
            # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
            # the best we can do.
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = (
                    self.num_tokens(train_dataloader, args.max_steps)
                    * args.gradient_accumulation_steps
                )
        else:
            max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
            num_train_epochs = math.ceil(args.num_train_epochs)
            num_train_samples = (
                self.num_examples(train_dataloader) * args.num_train_epochs
            )
            if args.include_tokens_per_second:
                num_train_tokens = (
                    self.num_tokens(train_dataloader) * args.num_train_epochs
                )

        delay_optimizer_creation = False

        self.mttl_args.total_steps = max_steps
        if self.mttl_args.warmup_steps == -1 or self.mttl_args.warmup_proportion > 0.0:
            logger.warning(
                "Warmup proportion is set to {}, has priority over warmup_steps".format(
                    self.mttl_args.warmup_proportion
                )
            )

            self.mttl_args.warmup_steps = int(
                self.mttl_args.warmup_proportion * max_steps
            )

        self.lr_scheduler = get_scheduler(self.optimizer, self.mttl_args)

        if self.is_deepspeed_enabled:
            raise ValueError("DeepSpeed is not supported in iterative training.")

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState(
            stateful_callbacks=[
                cb
                for cb in self.callback_handler.callbacks + [self.control]
                if isinstance(cb, ExportableState)
            ]
        )
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs
            )

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                self._fsdp_qlora_plugin_updates()
                self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(
                        self.model, self.optimizer
                    )
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )
        elif self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            # In this case we are in DDP + LOMO, which should be supported
            self.optimizer = self.accelerator.prepare(self.optimizer)

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(
            f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}"
        )
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(
                f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}"
            )
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(
            f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}"
        )

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
            )
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            self._load_callback_state()
            epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (
                    num_update_steps_per_epoch
                )
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info(
                "  Continuing training from checkpoint, will skip to saved global_step"
            )
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(
                f"  Continuing training from global step {self.state.global_step}"
            )
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader

        self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        grad_norm: Optional[float] = None
        self.control = self.callback_handler.on_train_begin(
            args, self.state, self.control
        )

        if args.eval_on_start:
            self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            # every epoch, we create a new datamodule for km training!
            if epoch % self.mttl_args.generate_every_n_epoch == 0:
                epoch_iterator = self.get_train_dataloader()

            if hasattr(epoch_iterator, "set_epoch"):
                epoch_iterator.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(
                args, self.state, self.control
            )

            if (
                epoch == epochs_trained
                and resume_from_checkpoint is not None
                and steps_trained_in_current_epoch == 0
            ):
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(
                    epoch_iterator, steps_trained_in_current_epoch
                )
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1

                if self.args.include_num_input_tokens_seen:
                    main_input_name = getattr(
                        self.model, "main_input_name", "input_ids"
                    )
                    if main_input_name not in inputs:
                        logger.warning(
                            "Tried to track the number of tokens seen, however the current model is "
                            "not configured properly to know what item is the input. To fix this, add "
                            "a `main_input_name` attribute to the model class you are using."
                        )
                    else:
                        self.state.num_input_tokens_seen += (
                            torch.sum(
                                self.accelerator.gather(
                                    torch.tensor(
                                        inputs[main_input_name].numel(),
                                        device=self.args.device,
                                        dtype=torch.int64,
                                    )
                                )
                            )
                            .cpu()
                            .item()
                        )
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(
                        args, self.state, self.control
                    )

                with self.accelerator.accumulate(model):
                    tr_loss_step = self.training_step(model, inputs)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_xla_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (
                        1 + self.state.global_step - self._globalstep_last_logged
                    )
                else:
                    if tr_loss.device != tr_loss_step.device:
                        raise ValueError(
                            f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                        )
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                is_last_step_and_steps_less_than_grad_acc = (
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                )

                if (
                    total_batched_samples % args.gradient_accumulation_steps == 0
                    or
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    is_last_step_and_steps_less_than_grad_acc
                ):
                    # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                    # in accelerate. So, explicitly enable sync gradients to True in that case.
                    if is_last_step_and_steps_less_than_grad_acc:
                        self.accelerator.gradient_state._set_sync_gradients(True)

                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        # deepspeed does its own clipping

                        if is_sagemaker_mp_enabled() and args.fp16:
                            _grad_norm = self.optimizer.clip_master_grads(
                                args.max_grad_norm
                            )
                        elif self.use_apex:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            _grad_norm = nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer),
                                args.max_grad_norm,
                            )
                        else:
                            _grad_norm = self.accelerator.clip_grad_norm_(
                                model.parameters(),
                                args.max_grad_norm,
                            )

                        if (
                            is_accelerate_available()
                            and self.accelerator.distributed_type
                            == DistributedType.DEEPSPEED
                        ):
                            grad_norm = model.get_global_grad_norm()
                            # In some cases the grad norm may not return a float
                            if hasattr(grad_norm, "item"):
                                grad_norm = grad_norm.item()
                        else:
                            grad_norm = _grad_norm

                    self.optimizer.step()

                    self.control = self.callback_handler.on_optimizer_step(
                        args, self.state, self.control
                    )

                    optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                    if optimizer_was_run:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(
                            self.lr_scheduler,
                            torch.optim.lr_scheduler.ReduceLROnPlateau,
                        ):
                            self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = (
                        epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    )
                    self.control = self.callback_handler.on_step_end(
                        args, self.state, self.control
                    )

                    self._maybe_log_save_evaluate(
                        tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval
                    )
                else:
                    self.control = self.callback_handler.on_substep_end(
                        args, self.state, self.control
                    )

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    # PyTorch/XLA relies on the data loader to insert the mark_step for
                    # each step. Since we are breaking the loop early, we need to manually
                    # insert the mark_step here.
                    if is_torch_xla_available():
                        xm.mark_step()
                    break
            if step < 0:
                logger.warning(
                    "There seems not to be a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(
                args, self.state, self.control
            )
            self._maybe_log_save_evaluate(
                tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval
            )

            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info(
            "\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n"
        )
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_xla_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                raise ValueError(
                    "load_best_model_at_end is not supported with SageMaker Model Parallelism"
                )

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(
            self.state.global_step, 0.001
        )  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(
            use_mtime=False, output_dir=run_dir
        )

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if (
            self.args.should_save
            and self.state.best_model_checkpoint is not None
            and self.args.save_total_limit == 1
        ):
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(
                        f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit"
                    )
                    shutil.rmtree(checkpoint, ignore_errors=True)

        self.control = self.callback_handler.on_train_end(
            args, self.state, self.control
        )

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def get_eval_dataloader(self, eval_dataset=None):
        if self.current_epoch_datamodule is not None:
            logger.info("Using current epoch datamodule for evaluation.")
            return self.current_epoch_datamodule.val_dataloader()
        else:
            raise ValueError("No current epoch datamodule found.")

    def create_datamodule_for_generated_data(self, dataset_path):
        args_copy = copy.deepcopy(self.mttl_args)
        args_copy.dataset = "local://" + dataset_path
        args_copy.dataset_type = "dcd_km"
        return get_datamodule(args_copy)

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        current_state_epoch = math.ceil(self.state.epoch or 0)
        dataset_path = self.mttl_args.output_dir + f"/gen__epoch_{current_state_epoch}"

        if not os.path.exists(dataset_path):
            train_dataset = self.generate_epoch_data(self.train_dataset)
            train_dataset.save_to_disk(dataset_path)

        datamodule = self.create_datamodule_for_generated_data(dataset_path)
        self.current_epoch_datamodule = datamodule
        return self.accelerator.prepare(datamodule.train_dataloader())

    def generate_epoch_data(self, train_dataset):
        # Clean up temporary directory
        import shutil

        from dataset_augmenter import DatasetAugmenter

        from mttl.models.utils import model_loader_helper

        shutil.rmtree(self.temp_directory, ignore_errors=True)
        os.makedirs(self.temp_directory, exist_ok=True)

        device = self.model.device
        if self.reference_model:
            self.reference_model.to("cpu")

        # we need to save the model in the temp directory, this moves it to CPU so that VLLM
        # doesn't complain, one alternative is to use 2 gpus, one for generation, one for training
        self.model.merge_and_save_base_model(self.temp_directory, device="cpu")

        # Generate a set of prompts
        augmenter = DatasetAugmenter(
            self.temp_directory,
            block_size=2048,
            max_continuation_length=768,
            num_generations=16,
            generation_top_p=0.95,
        )
        augmenter.add_task("summary")
        augmented_dataset = augmenter.augment(dataset=train_dataset)

        # Force garbage collection
        del augmenter
        gc.collect()
        torch.cuda.empty_cache()
        self.model.to(device)

        # reloading the reference model, this is with the previous adapter merged!
        # self.reference_model = model_loader_helper(
        #    self.temp_directory,
        #    device_map=device,
        #    bf16=self.mttl_args.precision == "bf16",
        #    fp16=self.mttl_args.precision == "fp16",
        #    attn_implementation=self.mttl_args.attn_implementation,
        # )
        return DatasetDict({"train": augmented_dataset})


@dataclass
class IterKMArguments(ExpertConfig):
    # set the following if you want to enable the NQA callback during training
    dataset_type: str = "text_dataset"
    dataset: str = "sordonia/narrativeqa"
    nqa_dataset: str = "sordonia/narrativeqa"
    generate_every_n_epoch: int = 5
    accumulate_datasets: bool = False


def train_km(training_args):
    seed_everything(training_args.seed, workers=True)

    # get directory of the current file
    setup_logging(training_args.output_dir)
    logger.info("Args: %s", training_args.to_json())

    remote_login(training_args.remote_token)

    model_config = ExpertModelConfig(
        base_model=args.model,
        task_name=args.finetune_task_name,
        expert_name=args.expert_name or args.finetune_task_name,
        modifier_config=args.modifier_config,
    )

    model = ExpertModel(
        model_config,
        load_in_4bit=training_args.load_in_4bit,
        load_in_8bit=training_args.load_in_8bit,
        device_map=training_args.device_map,
        attn_implementation=training_args.attn_implementation,
    )

    callbacks = []
    if training_args.nqa_dataset is not None:
        # load the NQA callback to monitor zero-shot performance
        from nqa_callback import NQAZeroShotCallback

        data_args = copy.deepcopy(training_args)
        data_args.dataset = training_args.nqa_dataset
        callback = NQAZeroShotCallback(model, data_args)
        callbacks.append(callback)

    trainer = IterativeDCDTrainer(
        model=model,
        args=training_args,
        callbacks=callbacks,
    )

    trainer.train()

    # Get the best checkpoint
    best_model_path = trainer.state.best_model_checkpoint
    if best_model_path:
        logger.info("Best model checkpoint: %s", best_model_path)


if __name__ == "__main__":
    args = IterKMArguments.parse()

    train_km(args)
