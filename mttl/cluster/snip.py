import torch
import torch.nn as nn
import torch.nn.functional as F
import types
import copy
import re
import json
from tqdm import tqdm
from typing import Type
from dataclasses import dataclass
import torch
from pytorch_lightning import Trainer, seed_everything

from mttl.callbacks import (
    DownstreamEvalCallback,
    LiveCheckpointCallback,
    NanoMMLUCallback,
    RougeCallback,
)
from mttl.config import Args, ExpertConfig
from mttl.datamodule.base import get_datamodule
from mttl.logging import get_pl_loggers, logger, setup_logging
from mttl.models.expert_model import ExpertModel, MoEModel
from mttl.models.library.expert import Expert, load_expert
from mttl.models.library.expert_library import ExpertLibrary, LocalExpertLibrary
from mttl.models.monitors import get_monitors
from mttl.utils import generate_random_string, rank_zero_only_and_wait, remote_login
from mttl.models.utils import transfer_batch_to_device
from mttl.cluster.trak_projectors import BasicProjector, CudaProjector, ProjectionType


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def snip_forward_conv2d(self, x):
    return F.conv2d(
        x,
        self.weight * self.weight_mask,
        self.bias,
        self.stride,
        self.padding,
        self.dilation,
        self.groups,
    )


def snip_forward_conv1d(self, x):
    return F.conv1d(x, self.weight * self.weight_mask, self.bias)


def snip_forward_linear(self, x):
    return F.linear(x, self.weight * self.weight_mask, self.bias)


forward_mapping_dict = {
    "Linear": snip_forward_linear,
    "Conv2d": snip_forward_conv2d,
    "Conv1d": snip_forward_conv1d,
}


def get_trak_projector(device: torch.device):
    """
    OO: this is taken from https://github.com/princeton-nlp/LESS
    OO: CudaProjector only works on V100 for me, not on A100
    Get trak projectors (see https://github.com/MadryLab/trak for details)
    """
    try:
        num_sms = torch.cuda.get_device_properties(device.index).multi_processor_count
        import fast_jl

        # test run to catch at init time if projection goes through
        fast_jl.project_rademacher_8(
            torch.zeros(8, 1_000, device=device), 512, 0, num_sms
        )
        projector = CudaProjector
        print("Using CudaProjector")
    except:
        projector = BasicProjector
        print("Using BasicProjector")
    return projector


@dataclass
class SNIPConfig(ExpertConfig):
    compression_factor: float = (
        0.05  # if float < 1, keep that fraction of parameters, otherwise keep that many parameters
    )
    layers: str = (
        ".*fc1|.*fc2"  # regex pattern to match the layers w.r.t. masks of which will calculate the grads
    )
    out_file: str = "snip_importance.jsonl"
    rand_proj_dim: int = (
        8192  # dimension of the random projection, used if use_rand_proj is True
    )
    use_rand_proj: bool = (
        True  # if 'True' use random projection like in SNIP, otherwise will take top-k params
    )


class SNIP:
    """
    Based on https://github.com/shivamsaboo17/PySNIP
    Uses gradient w.r.t mask to calculate the parameter importance.
    """

    def __init__(self, config: SNIPConfig, model: ExpertModel, criterion, dataloader):
        self.config: SNIPConfig = config
        self.model: ExpertModel = model.to(device)
        self.criterion = criterion.to(device)
        self.dataloader = dataloader
        self.layers = config.layers

        self.out_file = config.out_file
        if config.finetune_task_name is not None:
            self.out_file = (
                f"{config.out_file.split('.')[0]}_{config.finetune_task_name}.jsonl"
            )

        self.variance_scaling_init()
        self.update_forward_pass()
        dtype = next(self.model.parameters()).dtype

        projector_cls = get_trak_projector(device)
        dim_grads = self.number_of_considered_params()

        self.projector = nn.Identity()
        if dim_grads > self.config.rand_proj_dim and self.config.use_rand_proj:
            self.projector = projector_cls(
                grad_dim=dim_grads,
                proj_dim=self.config.rand_proj_dim,
                seed=0,
                proj_type=ProjectionType.rademacher,
                device=device,
                dtype=dtype,
                block_size=32,
                max_batch_size=32,
            )
            
    def _get_top_indices(self, grad):
        # if compression_factor is integer, keep that many parameters
        # otherwise keep (1 - compression_factor) fraction of parameters
        keep_params = (
            int((1 - self.config.compression_factor) * len(grad))
            if not self.config.compression_factor.is_integer()
            else int(self.config.compression_factor)
        )
        values, idxs = torch.topk(grad, keep_params, sorted=False)
        return idxs

    def _get_grad_list(self) -> list:
        """
        Returns a list of gradients for the masked parameters in the model
        """
        grads_list = []
        offset = 0
        for m_name, m in self.model.named_modules():
            if re.fullmatch(self.layers, m_name) and isinstance(m, nn.Linear):
                if hasattr(m, "weight_mask"):
                    g = torch.flatten(torch.abs(m.weight_mask.grad)).to(torch.float16)
                    g /= g.sum()
                    if not self.config.use_rand_proj:
                        # use top-k indices
                        g = self._get_top_indices(g) + offset
                        offset += m.weight_mask.numel()
                        grads_list += g.tolist()
                    else:
                        grads_list.append(g)

        if self.config.use_rand_proj:
            # use random projection
            # we project all concatenated gradients at once
            t = torch.cat(grads_list).unsqueeze(0)
            grads_list = self.projector.project(t, model_id=0)[0].tolist()
        return grads_list

    def param_importance(self):
        """
        For each datapoint in the dataset, compute the importance of each parameter
        if hard is True, returns one-hot importance, with top compression_factor% of parameters set to 1,
        otherwise returns the importance of each parameter as a float
        """
        fields = [
            "task_name",
            "task_ids",
            "source",
            "target",
            "task_source",
            "template_type",
            "template_idx",
            "split",
        ]
        with open(self.out_file, "a") as file:
            for i, batch in tqdm(
                enumerate(self.dataloader), total=len(self.dataloader)
            ):
                batch = transfer_batch_to_device(batch, device)
                loss = self.model.get_loss_for_all(batch, i)
                for ex_i, ex in enumerate(loss):
                    self.model.zero_grad()
                    ex.backward()
                    grads_list = self._get_grad_list()
                    # append line to file
                    json_line = {
                        f: (
                            batch[f][ex_i].cpu().tolist()
                            if isinstance(batch[f][ex_i], torch.Tensor)
                            else batch[f][ex_i]
                        )
                        for f in fields
                    }
                    json_line["snip_grads"] = grads_list
                    file.write(json.dumps(json_line) + "\n")
                    file.flush()
        print("\n ##### Done #####")

    def number_of_considered_params(self):
        """
        Returns number of masked parameters in the model. This is the dimention that will go into the random projector.
        """
        n = 0
        for m_name, m in self.model.named_modules():
            if re.fullmatch(self.layers, m_name) and isinstance(m, nn.Linear):
                if hasattr(m, "weight_mask"):
                    n += m.weight_mask.numel()
        return int(n)

    def variance_scaling_init(self):
        for m_name, module in self.model.named_modules():
            if re.fullmatch(self.layers, m_name) and isinstance(module, nn.Linear):
                assert isinstance(module, nn.Linear), "Only Linear layers are supported"
                module.weight_mask = nn.Parameter(
                    torch.ones_like(module.weight).to(device)
                )
                module.weight.requires_grad = False

    def update_forward_pass(self):
        for m_name, module in self.model.named_modules():
            if re.fullmatch(self.layers, m_name) and isinstance(module, nn.Linear):
                assert isinstance(module, nn.Linear), "Only Linear layers are supported"
                module.forward = types.MethodType(snip_forward_linear, module)


def main(args: Args, model_class: Type[ExpertModel]):
    seed_everything(args.seed, workers=True)
    # get directory of the current file
    setup_logging(args.output_dir)

    logger.info("Args: {}".format(args.to_json()))
    dm = get_datamodule(args)
    module = model_class(**args.asdict())
    snip = SNIP(args, module, nn.CrossEntropyLoss(), dm.train_dataloader())
    snip.param_importance()


if __name__ == "__main__":
    main(SNIPConfig.parse(), ExpertModel)
