"""
Utility package for reusable helpers across the repository.

Exposes common utilities such as positional embedding interpolation for
text encoders used in SigLIP/SigLIP2 experiments.
"""

from .positional_interpolation import (
    extend_text_positional_embedding,
    apply_text_position_interpolation,
)

__all__ = [
    "extend_text_positional_embedding",
    "apply_text_position_interpolation",
]

