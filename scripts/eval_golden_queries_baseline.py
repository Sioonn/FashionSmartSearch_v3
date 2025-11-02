"""Evaluate golden queries against the SigLIP-2 image index (no PI).

This is the same logic as scripts/eval_golden_queries.py but WITHOUT applying
positional embedding interpolation. Useful as a baseline or when you want to
use the model's default maximum sequence length.

Usage (defaults match repo layout):
  python scripts/eval_golden_queries_baseline.py \
    --golden tests/golden_data/golden_queries_by_index.json \
    --index-dir scripts/data/preprocess/siglip2_base_p16_384 \
    --model-name google/siglip2-base-patch16-384 \
    --topk 100

You may also directly specify an embeddings file and matching paths.json:
  python scripts/eval_golden_queries_baseline.py \
    --golden tests/golden_data/golden_queries_by_index.json \
    --embeddings-file scripts/scripts/data/preprocess/siglip2_base_p16_384/embeddings.npy \
    --paths-json scripts/scripts/data/preprocess/siglip2_base_p16_384/paths.json \
    --model-name google/siglip2-base-patch16-384 \
    --topk 100
"""

from __future__ import annotations

# Ensure repository root is importable when running this script directly
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor


def canonicalize_golden_path(p: str) -> str:
    p = p.replace("\\", "/")
    while p.startswith("./") or p.startswith("../"):
        p = p[2:] if p.startswith("./") else p[3:]
    if "." in Path(p).name:
        stem = Path(p).with_suffix("").name
        parent = str(Path(p).parent).replace("\\", "/")
        p = f"{parent}/{stem}" if parent != "." else stem
    return p


def canonicalize_index_path(p: str) -> str:
    p = p.replace("\\", "/")
    while p.startswith("./") or p.startswith("../"):
        p = p[2:] if p.startswith("./") else p[3:]
    stem = Path(p).with_suffix("").name
    parent = str(Path(p).parent).replace("\\", "/")
    return f"{parent}/{stem}" if parent != "." else stem


def _load_truncated_npy(path: Path) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """Best-effort loader for potentially truncated .npy arrays.

    Returns (flat_array_float32, header_shape). The flat array length may be
    smaller than the product of header_shape due to truncation.
    """
    import numpy.lib.format as npfmt

    with open(path, "rb") as f:
        version = npfmt.read_magic(f)
        if version == (1, 0):
            header = npfmt.read_array_header_1_0(f)
        elif version == (2, 0):
            header = npfmt.read_array_header_2_0(f)
        else:
            try:
                header = npfmt.read_array_header_2_0(f)
            except Exception:
                header = npfmt.read_array_header_1_0(f)
        shape, fortran_order, dtype = header
        data = np.fromfile(f, dtype=dtype)
        data = data.astype("float32", copy=False)
        return data, tuple(int(x) for x in shape)


def load_index(
    index_dir: Path | None,
    *,
    embeddings_file: Path | None = None,
    paths_json: Path | None = None,
) -> Tuple[np.ndarray, List[str]]:
    """Load image embeddings and path mapping (with legacy fallback and recovery)."""

    def _try_load(dir_path: Path, emb_path: Path | None = None, paths_path: Path | None = None):
        nonlocal_paths: List[str]
        if paths_path is None:
            paths_path = dir_path / "paths.json"
        if emb_path is None:
            emb_path = dir_path / "embeddings.npy"
        nonlocal_paths = json.loads(paths_path.read_text(encoding="utf-8"))
        try:
            emb_arr = np.load(str(emb_path)).astype("float32", copy=False)
            return emb_arr, nonlocal_paths
        except Exception as e:
            print(f"[warn] Failed to load {emb_path.name} via np.load: {e}")
            print("[warn] Attempting best-effort recovery from truncated file...")
            flat, hdr_shape = _load_truncated_npy(emb_path)
            if len(hdr_shape) != 2:
                raise RuntimeError(
                    f"Unsupported embeddings shape in header: {hdr_shape} (expected 2D)"
                )
            N_exp, D = hdr_shape
            rows = int(flat.size // D)
            if rows == 0:
                raise RuntimeError(
                    "Embeddings file appears empty or unrecoverably truncated."
                )
            if rows < N_exp:
                print(
                    f"[warn] Truncated embeddings detected: header rows={N_exp}, recovered rows={rows}."
                )
            emb_arr = flat[: rows * D].reshape(rows, D)
            if rows < len(nonlocal_paths):
                print(
                    f"[warn] Trimming paths to match recovered embeddings: {len(nonlocal_paths)} -> {rows}"
                )
                nonlocal_paths = nonlocal_paths[:rows]
            return emb_arr, nonlocal_paths

    if embeddings_file is not None:
        emb_path = Path(embeddings_file)
        dir_path = emb_path.parent
        if paths_json is not None:
            paths_path = Path(paths_json)
        else:
            paths_path = dir_path / "paths.json"
        if not paths_path.exists():
            raise FileNotFoundError(
                f"paths.json not found for embeddings file. Provide with --paths-json: {paths_path}"
            )
        return _try_load(dir_path, emb_path=emb_path, paths_path=paths_path)

    if index_dir is None:
        raise ValueError("index_dir must be provided when --embeddings-file is not used")
    if (index_dir / "embeddings.npy").exists() and (index_dir / "paths.json").exists():
        return _try_load(index_dir)

    legacy = Path("scripts/scripts/data/preprocess/siglip2_base_p16_384")
    if index_dir.as_posix() == "scripts/data/preprocess/siglip2_base_p16_384" and legacy.exists():
        print(f"[info] Falling back to legacy index dir: {legacy}")
        if (legacy / "embeddings.npy").exists() and (legacy / "paths.json").exists():
            return _try_load(legacy)

    raise FileNotFoundError(
        f"Could not find embeddings/paths under {index_dir}. Use --embeddings-file and --paths-json to specify files explicitly."
    )


def load_golden(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    return data


def encode_text(query: str, processor, model, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        inputs = processor(text=[query], return_tensors="pt", truncation=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        try:
            feats = model.get_text_features(**inputs)  # [1, D]
        except AttributeError:
            outs = model(**inputs)
            feats = outs.text_embeds  # [1, D]
        feats = F.normalize(feats, dim=-1)
    return feats.squeeze(0).cpu().numpy().astype("float32", copy=False)


def topk_indices_dot(mat: np.ndarray, vec: np.ndarray, k: int) -> np.ndarray:
    scores = mat @ vec
    if k >= len(scores):
        return np.argsort(-scores)
    idx = np.argpartition(scores, -k)[-k:]
    idx_sorted = idx[np.argsort(-scores[idx])]
    return idx_sorted


def evaluate(
    golden_path: Path,
    index_dir: Path | None,
    model_name: str,
    topk: int,
    *,
    embeddings_file: Path | None = None,
    paths_json: Path | None = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    img_emb, img_paths = load_index(index_dir, embeddings_file=embeddings_file, paths_json=paths_json)
    print(f"Loaded index: embeddings={img_emb.shape}, paths={len(img_paths)} from {index_dir}")

    canon_to_rows: Dict[str, List[int]] = {}
    for i, p in enumerate(img_paths):
        key = canonicalize_index_path(p)
        canon_to_rows.setdefault(key, []).append(i)

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    golden = load_golden(golden_path)

    success = 0
    total = 0
    misses: List[Tuple[int, str]] = []

    for item in golden:
        gt_path = str(item.get("img_path", ""))
        query = str(item.get("user_query", ""))
        idx = int(item.get("index", 0))

        if not gt_path or not query:
            continue
        total += 1

        text_vec = encode_text(query, processor, model, device)
        top_idx = topk_indices_dot(img_emb, text_vec, topk)

        gt_key = canonicalize_golden_path(gt_path)
        top_keys = {canonicalize_index_path(img_paths[i]) for i in top_idx.tolist()}
        ok = gt_key in top_keys
        if ok:
            success += 1
        else:
            misses.append((idx, gt_path))

    print(f"Success@{topk}: {success}/{total}")
    if misses:
        preview = ", ".join(f"#{i}:{p}" for i, p in misses[:10])
        print(f"First misses: {preview}{' ...' if len(misses) > 10 else ''}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate golden queries retrieval@TopK using SigLIP-2 text encoder (no PI).")
    p.add_argument("--golden", default="tests/golden_data/golden_queries_by_index.json")
    p.add_argument("--index-dir", default="scripts/data/preprocess/siglip2_base_p16_384")
    p.add_argument("--embeddings-file", default=None, help="Path to embeddings.npy if not under --index-dir")
    p.add_argument("--paths-json", default=None, help="Path to paths.json matching embeddings file")
    p.add_argument("--model-name", default="google/siglip2-base-patch16-384")
    p.add_argument("--topk", type=int, default=100)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    evaluate(
        golden_path=Path(args.golden),
        index_dir=Path(args.index_dir) if args.index_dir else None,
        model_name=args.model_name,
        topk=args.topk,
        embeddings_file=Path(args.embeddings_file) if args.embeddings_file else None,
        paths_json=Path(args.paths_json) if args.paths_json else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

