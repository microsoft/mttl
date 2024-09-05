from dataclasses import dataclass
from typing import Dict, List, Union

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


@dataclass
class SelectorOutputsContainer:
    """A container for multiple SelectorOutputs."""

    selector_outputs: List[SelectorOutput]
    selector_indices: Dict[int, torch.Tensor] = None

    def __post_init__(self):
        # make sure that all selector outputs are of the same type
        if len(set(type(so) for so in self.selector_outputs)) != 1:
            raise ValueError("All selector outputs should be of the same type.")

    @property
    def dim_index(self):
        raise NotImplementedError(
            "dim_index needs to be specified in order to know which dimension to split across SelectorOutputs"
        )


class SequenceSelectorOutputsContainer(SelectorOutputsContainer):

    def __post_init__(self):
        super().__post_init__()
        if any(
            so.dim_names[self.dim_index] == "sequence" for so in self.selector_outputs
        ):
            raise ValueError(
                "All selector outputs should not have 'sequence', as we are splitting across this dimension"
            )

        # make sure that indices don't overlap across different selector outputs
        all_indices = torch.cat(self.selector_indices)
        assert all_indices.unique().size(0) == all_indices.size(
            0
        ), "Indices should not overlap across different selector outputs"

    @property
    def dim_index(self):
        return 1
