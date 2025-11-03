"""
Extract unique `input_concat_description` per product for tops indices from the Fashion-Gen HDF5.

- Reads indices from a .npy file (FASHIONGEN_TOPS_IDX_PATH or --indices)
- Loads descriptions from HDF5 at FASHIONGEN_HDF5_PATH or --hdf5
- De-duplicates by `input_productID`: for sequential rows that share the same
  product ID, only the first occurrence is exported.
- Decoding rules per element type:
  * bytes / numpy.bytes_: decode as UTF-8
  * numpy.ndarray: convert to bytes, UTF-8 decode, strip (fallback to str(...))
  * other: str(...)
- Saves JSONL records, one per line:
  {"idx": <int>, "h5_index": <int>, "input_concat_description": "..."}
  - Fields:
    - idx: sequential row number in this exported file (0-based)
    - h5_index: actual HDF5 row index referenced by tops_indices.npy item value

Usage:
  python scripts/preprocess/extract_tops_descriptions.py \
    --indices data/preprocess/fashion_gen/tops_indices.npy \
    --hdf5 D:/.../fashiongen_256_256_train.h5 \
    --out data/preprocess/desc_raw/tops_concat_descriptions.jsonl \
    --limit 10

Environment (.env) overrides:
  - FASHIONGEN_TOPS_IDX_PATH  path to tops indices .npy
  - FASHIONGEN_HDF5_PATH      path to Fashion-Gen HDF5 file
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import h5py
import numpy as np
from dotenv import load_dotenv


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def decode_description(value: Any) -> str:
    """Decode a single HDF5 description element to UTF-8 text.

    Rules:
    - bytes / numpy.bytes_: UTF-8 decode, strip
    - numpy.ndarray: tobytes()->UTF-8 decode (ignore errors), strip; fallback to str(...)
    - otherwise: str(...), strip
    """
    # bytes-like
    if isinstance(value, (bytes, bytearray, np.bytes_)):
        return bytes(value).decode("utf-8", errors="ignore").replace("\x00", "").strip()

    # numpy array containers
    if isinstance(value, np.ndarray):
        try:
            b = value.tobytes()
            s = b.decode("utf-8", errors="ignore").replace("\x00", "").strip()
            if s:
                return s
        except Exception:
            pass
        # Fallback: flatten and join as string
        try:
            flat = value.flatten().tolist()
            s = " ".join(str(x) for x in flat).strip()
            if s:
                return s
        except Exception:
            pass
        return str(value).strip()

    # everything else
    return str(value).strip()


def normalize_product_id(value: Any) -> Optional[Union[int, str]]:
    """Return a normalized product ID (int if possible, else non-empty string)."""
    if value is None:
        return None
    # h5py may return numpy scalar types
    if isinstance(value, (np.integer,)):
        try:
            return int(value)
        except Exception:
            pass
    # direct int
    if isinstance(value, int):
        return value
    # bytes-like
    if isinstance(value, (bytes, bytearray, np.bytes_)):
        s = bytes(value).decode("utf-8", errors="ignore").replace("\x00", "").strip()
        if not s:
            return None
        # try to coerce to int
        try:
            return int(s)
        except Exception:
            return s
    # numpy arrays
    if isinstance(value, np.ndarray):
        # try scalar array
        try:
            if value.shape == ():
                return normalize_product_id(value.item())
        except Exception:
            pass
        # fallback to string
        s = str(value).strip()
        return s if s else None
    # other types
    s = str(value).strip()
    if not s:
        return None
    try:
        return int(s)
    except Exception:
        return s


def parse_args() -> argparse.Namespace:
    load_dotenv(override=False)
    p = argparse.ArgumentParser(description="Extract tops input_concat_description to JSON mapping")
    p.add_argument("--indices", type=Path, default=os.getenv("FASHIONGEN_TOPS_IDX_PATH"), help="Path to tops indices .npy")
    p.add_argument("--hdf5", type=Path, default=os.getenv("FASHIONGEN_HDF5_PATH"), help="Path to Fashion-Gen HDF5 file")
    p.add_argument("--out", type=Path, default=Path("data/preprocess/desc_raw/tops_concat_descriptions.jsonl"), help="Output JSONL path")
    p.add_argument("--desc-key", type=str, default="input_concat_description", help="HDF5 key for descriptions")
    p.add_argument("--product-key", type=str, default="input_productID", help="HDF5 key for product ID")
    p.add_argument("--limit", type=int, default=None, help="Limit number of records written (e.g., 10)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.indices is None or not Path(args.indices).exists():
        raise SystemExit(f"Missing tops indices .npy: {args.indices!s}")
    if args.hdf5 is None or not Path(args.hdf5).exists():
        raise SystemExit(f"Missing HDF5 file: {args.hdf5!s}")

    # Load indices
    indices = np.load(str(args.indices))
    if indices.ndim != 1:
        indices = indices.reshape(-1)

    # No stride; iterate all indices. Limit applies to number of outputs.

    # Open HDF5 and dataset
    with h5py.File(str(args.hdf5), "r") as h5f:
        if args.desc_key not in h5f:
            raise SystemExit(
                f"Description key '{args.desc_key}' not found in HDF5. Available: {list(h5f.keys())}"
            )
        ds = h5f[args.desc_key]
        if args.product_key not in h5f:
            raise SystemExit(
                f"Product key '{args.product_key}' not found in HDF5. Available: {list(h5f.keys())}"
            )
        prod_ds = h5f[args.product_key]

        # Write JSONL as we iterate to reduce memory
        ensure_parent(args.out)
        written = 0
        with Path(args.out).open("w", encoding="utf-8") as out_f:
            seen_products = set()
            seq = 0
            for idx in indices.tolist():
                # product ID
                try:
                    pid_val = prod_ds[idx]
                except Exception:
                    pid_val = None
                pid = normalize_product_id(pid_val)
                key = f"pid:{pid}" if pid is not None else f"row:{int(idx)}"
                if key in seen_products:
                    continue
                seen_products.add(key)

                # description
                try:
                    value = ds[idx]
                    text = decode_description(value)
                except Exception:
                    text = ""

                record = {
                    "idx": int(seq),              # sequential (0,1,2,...)
                    "h5_index": int(idx),         # HDF5 index (item value)
                    "input_concat_description": text,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                seq += 1

                if args.limit is not None and written >= int(args.limit):
                    break

    print(f"[done] wrote {written} JSONL records to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
