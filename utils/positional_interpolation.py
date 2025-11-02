"""positional_interpolation
Reusable helpers to extend text positional embeddings via interpolation.

This is designed for models like SigLIP/SigLIP2 where we want to increase
the maximum number of text positions (e.g., from 64 to 128) without
retraining the entire embedding from scratch.

Example usage:

    from utils.positional_interpolation import (
        extend_text_positional_embedding,
        apply_text_position_interpolation,
    )

    # Directly resize an Embedding layer
    new_emb = extend_text_positional_embedding(old_emb, new_max_positions=128)

    # Or, apply in-place to a HF model by attribute path
    apply_text_position_interpolation(
        model,
        attr_path="text_model.embeddings.position_embeddings",
        new_max_positions=128,
    )

Notes
-----
- Uses 1D linear interpolation over the position dimension.
- By default, preserves the first and last position vectors exactly and
  interpolates the interior positions. Set `preserve_ends=False` to interpolate
  the entire range instead.
- Updates common `config` fields (e.g., `max_position_embeddings`) if present.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


def _interpolate_positions(
    weight: torch.Tensor,
    new_len: int,
    mode: str = "linear",
    preserve_ends: bool = True,
) -> torch.Tensor:
    """Interpolate a 2D positional weight matrix to a new length.

    Parameters
    ----------
    weight: torch.Tensor
        Shape ``[old_len, dim]`` positional embedding weights.
    new_len: int
        Desired number of positions after interpolation.
    mode: str, default "linear"
        Interpolation mode for ``torch.nn.functional.interpolate``.
    preserve_ends: bool, default True
        If True and both old/new lengths >= 2, preserve the first and last
        vectors exactly and interpolate only the interior positions.

    Returns
    -------
    torch.Tensor
        New weight tensor with shape ``[new_len, dim]``.
    """

    old_len, dim = weight.shape

    if new_len == old_len:
        return weight.clone()

    if old_len < 2 or new_len < 2 or not preserve_ends:
        # Interpolate across the whole range
        w = weight.T.unsqueeze(0)  # [1, dim, old_len]
        w_new = F.interpolate(w, size=new_len, mode=mode, align_corners=True)
        return w_new.squeeze(0).T  # [new_len, dim]

    # Preserve first and last entries; interpolate the interior
    first = weight[:1]
    last = weight[-1:]
    interior = weight[1:-1]  # [old_len-2, dim]

    if interior.numel() == 0:
        # Degenerate case: old_len == 2 -> just duplicate ends appropriately
        w = weight.T.unsqueeze(0)
        w_new = F.interpolate(w, size=new_len, mode=mode, align_corners=True)
        return w_new.squeeze(0).T

    target_interior = max(new_len - 2, 0)
    if target_interior == 0:
        # All positions collapse to ends; distribute as evenly as possible
        # Here, fallback to whole-range interpolation
        w = weight.T.unsqueeze(0)
        w_new = F.interpolate(w, size=new_len, mode=mode, align_corners=True)
        return w_new.squeeze(0).T

    w = interior.T.unsqueeze(0)  # [1, dim, old_len-2]
    w_new = F.interpolate(w, size=target_interior, mode=mode, align_corners=True)
    interior_new = w_new.squeeze(0).T  # [target_interior, dim]
    return torch.cat([first, interior_new, last], dim=0)


def extend_text_positional_embedding(
    embedding: nn.Embedding,
    new_max_positions: int,
    *,
    mode: str = "linear",
    preserve_ends: bool = True,
) -> nn.Embedding:
    """Create a resized positional ``nn.Embedding`` via 1D interpolation.

    This keeps the embedding on the same device/dtype and copies over the
    interpolated weights. Grad settings are preserved.

    Parameters
    ----------
    embedding: nn.Embedding
        The original positional embedding layer.
    new_max_positions: int
        Desired maximum number of positions (e.g., 128 instead of 64).
    mode: str, default "linear"
        Interpolation mode for resizing.
    preserve_ends: bool, default True
        Preserve the first/last vectors exactly; interpolate interior only.

    Returns
    -------
    nn.Embedding
        A new ``nn.Embedding`` with shape ``[new_max_positions, hidden_dim]``.
    """

    old_num, dim = embedding.num_embeddings, embedding.embedding_dim
    if new_max_positions <= 0:
        raise ValueError("new_max_positions must be positive")

    device = embedding.weight.device
    dtype = embedding.weight.dtype
    requires_grad = embedding.weight.requires_grad

    with torch.no_grad():
        new_weight = _interpolate_positions(
            embedding.weight.detach().to(device=device, dtype=dtype),
            new_len=new_max_positions,
            mode=mode,
            preserve_ends=preserve_ends,
        )

    new_emb = nn.Embedding(new_max_positions, dim, device=device, dtype=dtype)
    with torch.no_grad():
        new_emb.weight.copy_(new_weight)
    new_emb.weight.requires_grad = requires_grad
    return new_emb


def apply_text_position_interpolation(
    model: nn.Module,
    attr_path: str,
    new_max_positions: int,
    *,
    mode: str = "linear",
    preserve_ends: bool = True,
) -> None:
    """Replace a model's text positional embedding with an interpolated one.

    Parameters
    ----------
    model: nn.Module
        The model instance containing the positional embedding.
    attr_path: str
        Dot-separated path to the positional ``nn.Embedding`` attribute,
        e.g., ``"text_model.embeddings.position_embeddings"``.
    new_max_positions: int
        Target number of positions (e.g., 128).
    mode: str, default "linear"
        Interpolation mode.
    preserve_ends: bool, default True
        Preserve the first/last vectors exactly.

    Notes
    -----
    - Attempts to update common config fields (``max_position_embeddings``)
      if present on ``model.config`` or nested ``text_config``.
    - Operates in-place on ``model``.
    """

    if not attr_path:
        raise ValueError("attr_path must be a non-empty dot path to an Embedding")

    parts = attr_path.split(".")
    parent = model
    for name in parts[:-1]:
        if not hasattr(parent, name):
            raise AttributeError(f"Path segment '{name}' not found under model")
        parent = getattr(parent, name)

    leaf_name = parts[-1]
    if not hasattr(parent, leaf_name):
        raise AttributeError(f"Leaf attribute '{leaf_name}' not found at path")

    leaf = getattr(parent, leaf_name)
    if not isinstance(leaf, nn.Embedding):
        raise TypeError(
            f"Attribute at path is type {type(leaf).__name__}, expected nn.Embedding"
        )

    new_emb = extend_text_positional_embedding(
        leaf,
        new_max_positions=new_max_positions,
        mode=mode,
        preserve_ends=preserve_ends,
    )
    setattr(parent, leaf_name, new_emb)

    # Best-effort config updates for HuggingFace-style models
    cfg = getattr(model, "config", None)
    if cfg is not None:
        for key in ("max_position_embeddings", "max_positions"):
            if hasattr(cfg, key):
                setattr(cfg, key, new_max_positions)
        # Sometimes nested under text_config
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None:
            for key in ("max_position_embeddings", "max_positions"):
                if hasattr(text_cfg, key):
                    setattr(text_cfg, key, new_max_positions)

