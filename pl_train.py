import os
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from mttl.config import parse_config
from mttl.callbacks import ProgressCallback
from mttl.datamodule.ni_data_module import NIPretrainDataModule
from mttl.datamodule.xfit_data_module import XFitPretrainDataModule
from mttl.datamodule.t0_data_module import T0PretrainDataModule
from mttl.models.encoder_decoder import EncoderDecoder
from mttl.models.t0_encoder_decoder import T0EncoderDecoder
from mttl.models.monitors import get_monitors
from mttl.utils import get_mlf_logger
import torch
from sklearn.decomposition import PCA
import numpy as np
import torch.nn as nn
import math


def tensor_product_construct(
    args, in_features, out_features, weight_leafs, tensor_rank, embedding_dim, flag="up"
):
    w = weight_leafs
    order = args.order
    rank = args.lora_rank
    embedding_dim_leaf_a = math.ceil((in_features) ** (1 / order))
    embedding_dim_leaf_b = math.ceil((out_features) ** (1 / order))
    layerone_normalization_a = nn.LayerNorm(
        normalized_shape=[rank, embedding_dim_leaf_a**2]
    )
    # self.layertwo_normalization_a = nn.LayerNorm(
    #     normalized_shape=[self.rank, self.embedding_dim_leaf_a**2])
    layerone_normalization_b = nn.LayerNorm(
        normalized_shape=[rank, embedding_dim_leaf_b**2]
    )
    w01 = w[0, :, :, :, None] * w[1, :, :, None, :]
    # print(w[:,:,:,:].size())
    w01 = w01.view(tensor_rank, rank, -1)
    if flag == "up":
        w01 = layerone_normalization_a(w01)
    elif flag == "down":
        w01 = layerone_normalization_b(w01)
    # print(w01.size())
    return w01[:, :, :embedding_dim]


def run_multitask(args):
    seed_everything(args.seed, workers=True)

    # select dataloader
    if args.dataset == "xfit":
        model_class = EncoderDecoder
        dm = XFitPretrainDataModule(args)
    elif args.dataset == "ni":
        model_class = EncoderDecoder
        dm = NIPretrainDataModule(args)
    elif args.dataset == "t0":
        model_class = T0EncoderDecoder
        dm = T0PretrainDataModule(args)
    else:
        raise NotImplementedError()

    args.n_tasks = len(dm.task2id)

    if args.example_to_ids_path:
        from models.cluster_reader import ClusterResult

        cluster_result = ClusterResult(args.example_to_ids_path)
        args.n_tasks = cluster_result.n_clusters()

        if args.poly_selector in ["cluster_soft", "cluster_hard"]:
            args.n_skills = cluster_result.n_clusters()
        else:
            raise NotImplementedError()

    save_name = args.model_modifier
    if args.checkpoint is not None:
        from mttl.utils import get_checkpoint_path

        checkpoint_path = get_checkpoint_path(args.checkpoint)

        kwargs = vars(args)
        kwargs.pop("checkpoint")
        module = model_class.load_from_checkpoint(
            checkpoint_path, **kwargs, tokenizer=dm.tokenizer
        )
    else:
        module = model_class(**vars(args), tokenizer=dm.tokenizer)

    # compute the similarity across different tasks
    # get the selector weights
    routing = (
        module.model.decoder.block[0].layer[0].SelfAttention.k.selector.module_logits
    )

    # get the similarity score
    def get_adapter_similarity():
        adapter_embedding = []
        out_features, in_features = (
            module.model.decoder.block[0].layer[0].SelfAttention.k.weight.shape
        )
        for task_name in dm.task2id:
            routing_distribution = routing[dm.task2id[task_name]]

            if args.model_modifier == "tensorpoly_lora":

                adapater_weights_lora_a = (
                    module.model.decoder.block[0]
                    .layer[0]
                    .SelfAttention.k.weight_leafs_a
                )
                adapater_weights_lora_b = (
                    module.model.decoder.block[0]
                    .layer[0]
                    .SelfAttention.k.weight_leafs_b
                )
                lora_a = tensor_product_construct(
                    args,
                    in_features,
                    out_features,
                    weight_leafs=adapater_weights_lora_a,
                    tensor_rank=args.n_skills,
                    embedding_dim=in_features,
                    flag="up",
                )

                lora_b = tensor_product_construct(
                    args,
                    in_features,
                    out_features,
                    weight_leafs=adapater_weights_lora_b,
                    tensor_rank=args.n_skills,
                    embedding_dim=out_features,
                    flag="down",
                )
                lora_a = lora_a.transpose(2, 1).unsqueeze(0)
                lora_b = lora_b.unsqueeze(0)

                # A is    n_splits, n_skills, D // n_splits, rank
                # we want bs,       n_splits, D // n_splits, rank
                adapter_weights = torch.einsum(
                    "abir, abro -> abio",
                    lora_a,
                    lora_b,
                )
                result = torch.einsum(
                    "bi,bijk->bijk",
                    routing_distribution.reshape(1, -1),
                    adapter_weights,
                )

                adapter = result.sum(dim=1).squeeze(0).reshape(1, -1)
            elif args.model_modifier == "tensororderpoly_lora":
                mixing_weights = routing_distribution.reshape(
                    args.n_splits, args.n_skills
                )

                adapater_weights_lora_a = (
                    module.model.decoder.block[0]
                    .layer[0]
                    .SelfAttention.k.weight_leafs_a
                )
                adapater_weights_lora_b = (
                    module.model.decoder.block[0]
                    .layer[0]
                    .SelfAttention.k.weight_leafs_b
                )
                A = torch.einsum(
                    "os,osrl->orl", (mixing_weights, adapater_weights_lora_a)
                )
                # unsqueeze the rank dimension
                A = A.unsqueeze(1)
                B = torch.einsum(
                    "os,osrl->orl", (mixing_weights, adapater_weights_lora_b)
                )
                B = B.unsqueeze(1)

                lora_a = tensor_product_construct(
                    args,
                    in_features,
                    out_features,
                    weight_leafs=A,
                    tensor_rank=1,
                    embedding_dim=in_features,
                    flag="up",
                )
                lora_b = tensor_product_construct(
                    args,
                    in_features,
                    out_features,
                    weight_leafs=B,
                    tensor_rank=1,
                    embedding_dim=out_features,
                    flag="down",
                )

                lora_a = lora_a.transpose(2, 1).unsqueeze(0)
                lora_b = lora_b.unsqueeze(0)

                # A is    n_splits, n_skills, D // n_splits, rank
                # we want bs,       n_splits, D // n_splits, rank
                adapter_weights = torch.einsum(
                    "abir, abro -> abio",
                    lora_a,
                    lora_b,
                )
                result = torch.einsum(
                    "bi,bijk->bijk",
                    routing_distribution.reshape(1, -1),
                    adapter_weights,
                )

                adapter = result.sum(dim=1).squeeze(0).reshape(1, -1)
            elif args.model_modifier == "poly_lora":
                # get the lora weights
                adapater_weights_lora_a = (
                    module.model.decoder.block[0].layer[0].SelfAttention.k.lora_a
                )
                adapater_weights_lora_b = (
                    module.model.decoder.block[0].layer[0].SelfAttention.k.lora_b
                )

                # compute the final adapter according to the routing
                adapter_weights = torch.einsum(
                    "abir, abro -> abio",
                    adapater_weights_lora_a,
                    adapater_weights_lora_b,
                )

                result = torch.einsum(
                    "bi,bijk->bijk",
                    routing_distribution.reshape(1, -1),
                    adapter_weights,
                )

                adapter = result.sum(dim=1).squeeze(0).reshape(1, -1)
            adapter_embedding.append(adapter)

        # compute the similarity across different tasks
        adapter_embedding = torch.cat(adapter_embedding, dim=0)
        # reduce the dimensionality
        pca = PCA(n_components=100)
        adapter_embedding = pca.fit_transform(adapter_embedding.cpu().detach().numpy())

        # create the adapter dict
        adapter_dict = {}
        for i, task_name in enumerate(dm.task2id):
            adapter_dict[task_name] = adapter_embedding[i]

        # save the adapter embedding to npy
        np.save(f"adapter_embedding_{save_name}.npy", adapter_dict)
        breakpoint()

    # adapter_embedding = torch.tensor(adapter_embedding).to(module.device)
    # adapter_embedding = adapter_embedding / (
    #     adapter_embedding.norm(dim=-1, keepdim=True, p=2) + 1e-6
    # )

    # # compute the consine similarity
    # similarity = torch.einsum("ij, kj -> ik", adapter_embedding, adapter_embedding)
    # # save the similarity matrix to npy
    # np.save("similarity.npy", similarity.cpu().detach().numpy())
    def get_adapter_grad_similarity():
        dm.setup()
        adapter_embedding_grad = {}
        count = 0
        for batch in dm.val_dataloader():
            task_id = batch["task_ids"].numpy()[0]
            loss, _ = module.validation_step(batch, 0)
            loss.backward()

            routing_distribution = routing[task_id]
            # get the lora weights
            adapater_weights_grad_lora_a = (
                module.model.decoder.block[0].layer[0].SelfAttention.k.lora_a
            ).grad
            adapater_weights_grad_lora_b = (
                module.model.decoder.block[0].layer[0].SelfAttention.k.lora_b
            ).grad

            represent_a = torch.einsum(
                "bi,bijk->bijk",
                routing_distribution.reshape(1, -1),
                adapater_weights_grad_lora_a,
            )

            represent_b = torch.einsum(
                "bi,bijk->bijk",
                routing_distribution.reshape(1, -1),
                adapater_weights_grad_lora_b,
            )

            adapter_a = represent_a.sum(dim=1).squeeze(0).reshape(1, -1)
            adapter_b = represent_b.sum(dim=1).squeeze(0).reshape(1, -1)

            # concat
            adapter = torch.cat((adapter_a, adapter_b), dim=1)
            adapter_embedding_grad[task_id] = adapter
            count += 1
            # if count > 2:
            #     break
            print(loss)

        np.save(f"adapter_embedding_grad_{save_name}.npy", adapter_embedding_grad)

    if args.similarity_analysis == "weight":
        get_adapter_similarity()
    elif args.similarity_analysis == "grad":
        get_adapter_grad_similarity()
    # adapter_embedding_grad = torch.cat(adapter_embedding_grad, dim=0)
    # # reduce the dimensionality
    # pca = PCA(n_components=20)
    # adapter_embedding_grad = pca.fit_transform(
    #     adapter_embedding_grad.cpu().detach().numpy()
    # )
    # adapter_embedding_grad = torch.tensor(adapter_embedding_grad).to(module.device)
    # # convert it to unit vector

    # adapter_embedding_grad = adapter_embedding_grad / (
    #     adapter_embedding_grad.norm(dim=-1, keepdim=True, p=2)
    # )

    # # compute the consine similarity
    # similarity_grad = torch.einsum(
    #     "ij, kj -> ik", adapter_embedding_grad, adapter_embedding_grad
    # )
    # # save the similarity matrix to npy
    # np.save("similarity_grad.npy", similarity_grad.cpu().detach().numpy())
    # breakpoint()
    # legit logging
    loggers = []
    if os.environ.get("WANDB_API_KEY"):
        wandb_logger = pl.loggers.WandbLogger(
            project="polytropon-ni",
            name=os.environ.get("AMLT_JOB_NAME", args.exp_name),
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

    if args.dataset in ["ni", "xfit"]:
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.output_dir,
            monitor=monitor,
            filename=f"{args.model}" + "-{" + monitor + ":.004f}",
            save_top_k=1,
            save_last=True,
            save_weights_only=True,  # make checkpoints smaller
            mode=mode,
        )
        callbacks.append(checkpoint_callback)
    else:
        # no need for checkpointing in t0 as we checkpoint manually in the module
        kwargs["enable_checkpointing"] = False

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
        log_every_n_steps=50,
        strategy=args.compute_strategy if args.compute_strategy != "null" else None,
        callbacks=callbacks,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        precision=(
            int(args.precision) if args.precision in ["16", "32"] else args.precision
        ),
        **kwargs,
    )

    # trainer.fit(module, dm)

    try:
        trainer.validate(module, dm)
    except:
        pass


if __name__ == "__main__":
    args = parse_config()
    run_multitask(args)
