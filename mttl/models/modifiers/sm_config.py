from dataclasses import dataclass

from mttl.models.modifiers.sparse_utils.sparse_linear import SparseLinearConfig


@dataclass
class SparseMaskConfig(SparseLinearConfig):
    steps_in_mask_selection: int = (
        1  # fo how many batches stay in mask update regime where sparse weights are fixed but masks are updated
    )
    mask_reselection_interval: int = (
        100  # every how many steps to switch to mask update regime
    )
    n_max_mask_reselection: int = (
        -1
    )  # how many mask updates to do. If > 0, the mask updater will be removed after this many updates
    mask_updater: str = None  # "snip"
    skip_zeros_mask_update: bool = (
        False  # DEPRECATED if True, until the first mask update operate in full FT regime. DEPRECATED
    )
