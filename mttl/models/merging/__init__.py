"""
Standalone merging and transformation functions for Expert objects.

This module provides standalone routines that take a list of Expert objects
as input and return merged or transformed experts. These functions are decoupled
from library transforms and can be used independently.
"""

from .wudi import wudi_merge, wudi_merge_after
from .weighted_linear import weighted_linear_merge
from .ties import ties_merge
from .arrow import arrow_transform
from .phatgoose import (
    extract_phatgoose_prototypes,
    validate_phatgoose_training,
    initialize_phatgoose_gates,
)

__all__ = [
    "wudi_merge",
    "wudi_merge_after",
    "weighted_linear_merge",
    "ties_merge",
    "arrow_transform",
    "extract_phatgoose_prototypes",
    "validate_phatgoose_training",
    "initialize_phatgoose_gates",
]
