from dataclasses import dataclass
from typing import List, Union

import torch


@dataclass
class SelectorOutput:
    ALL_EXPERTS = "all"

    def __post_init__(self):
        if hasattr(self, "weights") and self.weights.ndim != len(self.dim_names):
            raise ValueError(
                "Weights should have the same number of dimensions as dim_names for this SelectorOutput."
            )

    @property
    def dim_names(self):
        raise NotImplementedError("dim_names not implemented for this selector output.")


@dataclass
class BatchExpertsSelectorOutput(SelectorOutput):
    """A selector output that contains a list of experts without weights.

    experts: names of the selected experts for each element in the batch
    """

    experts: List[str]


@dataclass
class ExpertsAndWeightsSelectorOutput(SelectorOutput):
    """A selector output that contains a list of experts and weights shared across the batch.

    experts: names of the selected experts
    weights: their weights
    """

    experts: List[str]
    weights: torch.Tensor

    @property
    def dim_names(self):
        return ["experts"]


@dataclass
class BatchExpertsAndWeightsSelectorOutput(SelectorOutput):
    """A selector output that contains a list of experts and weights for each example.

    experts: either names or indices of the selected experts
    weights: their weights
    """

    experts: Union[List[List[str]], torch.Tensor]
    weights: torch.Tensor

    @property
    def dim_names(self):
        return ["batch", "experts"]


@dataclass
class ExpertsSplitsAndWeightsSelectorOutput(ExpertsAndWeightsSelectorOutput):
    """A selector output that contains a list of experts and weights for each split (MHR) and expert shared across the batch.

    experts: names of the selected experts
    weights: their weights
    """

    @property
    def dim_names(self):
        return ["splits", "experts"]


@dataclass
class BatchExpertsSplitsAndWeightsSelectorOutput(BatchExpertsAndWeightsSelectorOutput):
    """A selector output that contains a list of experts and weights for each split (MHR) and expert shared across the batch.

    experts: names of the selected experts
    weights: their weights
    """

    @property
    def dim_names(self):
        return ["batch", "splits", "experts"]


@dataclass
class BatchSequenceExpertsAndWeightsSelectorOutput(SelectorOutput):
    """A selector output that contains a list of experts and weights for each example and token.

    experts: indices of the selected experts
    weights: their weights
    """

    experts: torch.Tensor
    weights: torch.Tensor

    @property
    def dim_names(self):
        return ["batch", "sequence", "experts"]


@dataclass
class MultiheadBatchSequenceExpertsAndWeightsSelectorOutput(SelectorOutput):
    """A selector output that contains a list of experts and weights for each example and token.

    experts: indices of the selected experts
    weights: their weights
    """

    experts: torch.Tensor
    weights: torch.Tensor

    @property
    def dim_names(self):
        return ["batch", "sequence", "head", "experts"]


@dataclass
class BatchSequenceExpertsSplitsAndWeightsSelectorOutput(
    BatchSequenceExpertsAndWeightsSelectorOutput
):
    """A selector output that contains a list of experts and weights for each example, token and split (MHR)."""

    @property
    def dim_names(self):
        return ["batch", "sequence", "splits", "experts"]
