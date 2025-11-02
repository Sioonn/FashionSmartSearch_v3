"""Evaluate golden queries against the SigLIP-2 image index.

Loads the image embedding index from `--index-dir`, encodes each user query
from `--golden` using the SigLIP-2 text encoder (optionally with positional
interpolation to extend max positions), computes cosine similarity, and checks
whether the ground-truth image path appears in the top-K results.

Usage (defaults match repo layout):
  python scripts/eval_golden_queries.py \
    --golden tests/golden_data/golden_queries_by_index.json \
    --index-dir scripts/data/preprocess/siglip2_base_p16_384 \
    --model-name google/siglip2-base-patch16-384 \
    --topk 100 \
    --text-pi 128 \
    --pi-attr-path text_model.embeddings.position_embeddings

Notes
-----
- The index stores L2-normalized image embeddings; text features are also
  normalized before similarity computation (dot product == cosine).
- Path matching is done with a canonicalized form that ignores leading "../"
  and file extensions, to match the format in golden JSON.
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

from utils.positional_interpolation import apply_text_position_interpolation


def canonicalize_golden_path(p: str) -> str:
    p = p.replace("\\", "/")
    # strip leading ./ or ../
    while p.startswith("./") or p.startswith("../"):
        p = p[2:] if p.startswith("./") else p[3:]
    # remove extension if present
    if "." in Path(p).name:
        stem = Path(p).with_suffix("").name
        parent = str(Path(p).parent).replace("\\", "/")
        p = f"{parent}/{stem}" if parent != "." else stem
    return p


def canonicalize_index_path(p: str) -> str:
    p = p.replace("\\", "/")
    # strip leading ./ or ../ for index paths
    while p.startswith("./") or p.startswith("../"):
        p = p[2:] if p.startswith("./") else p[3:]
    # drop extension
    stem = Path(p).with_suffix("").name
    parent = str(Path(p).parent).replace("\\", "/")
    return f"{parent}/{stem}" if parent != "." else stem


def _load_truncated_npy(path: Path) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """Best-effort loader for potentially truncated .npy arrays.

    Returns (flat_array_float32, header_shape). The flat array length may be
    smaller than the product of header_shape due to truncation.
    """
    import numpy.lib.format as npfmt  # local import to avoid global dependency

    with open(path, "rb") as f:
        version = npfmt.read_magic(f)
        if version == (1, 0):
            header = npfmt.read_array_header_1_0(f)
        elif version == (2, 0):
            header = npfmt.read_array_header_2_0(f)
        else:
            # Try latest available; fall back to 2.0
            try:
                header = npfmt.read_array_header_2_0(f)
            except Exception:
                header = npfmt.read_array_header_1_0(f)
        shape, fortran_order, dtype = header
        # Read remaining bytes as flat array
        data = np.fromfile(f, dtype=dtype)
        # Convert to float32 for downstream use
        data = data.astype("float32", copy=False)
        return data, tuple(int(x) for x in shape)


def load_index(
    index_dir: Path | None,
    *,
    embeddings_file: Path | None = None,
    paths_json: Path | None = None,
) -> Tuple[np.ndarray, List[str]]:
    """Load image embeddings and path mapping.

    Priority:
    1) If `embeddings_file` is provided, use it. Paths are loaded from
       `paths_json` if provided, otherwise from the parent folder's paths.json.
    2) Else, load from `index_dir/embeddings.npy` and `index_dir/paths.json`.
       If missing and `index_dir` is the default location, try a fallback
       at `scripts/scripts/data/preprocess/siglip2_base_p16_384`.
    """

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

    # Case 1: explicit embeddings file
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

    # Case 2: index_dir
    if index_dir is None:
        raise ValueError("index_dir must be provided when --embeddings-file is not used")
    if (index_dir / "embeddings.npy").exists() and (index_dir / "paths.json").exists():
        return _try_load(index_dir)

    # Fallback: legacy location with scripts/scripts/ prefix
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


def _iter_named_embedding_paths(model) -> List[str]:
    paths: List[str] = []
    try:
        import torch.nn as nn  # local import for safety
    except Exception:
        return paths
    for name, module in model.named_modules():
        try:
            if isinstance(module, nn.Embedding):
                paths.append(name)
        except Exception:
            continue
    return paths


def maybe_apply_text_pi(
    model, processor, new_max_positions: int, attr_path: str | None
) -> None:
    if new_max_positions is None or new_max_positions <= 0:
        return

    tried: List[str] = []
    candidates: List[str] = []
    if attr_path:
        candidates.append(attr_path)
    # Common SigLIP/SigLIP2 paths
    candidates.extend([
        "text_model.embeddings.position_embedding",
        "text_model.embeddings.position_embeddings",
        # Some variants nest under model.text_model
        "model.text_model.embeddings.position_embedding",
        "model.text_model.embeddings.position_embeddings",
    ])

    # Add auto-detected embedding module paths that look positional and under text branch
    for p in _iter_named_embedding_paths(model):
        if "position" in p and (p.startswith("text_model") or ".text_model" in p or p.startswith("model.text_model")):
            if p not in candidates:
                candidates.append(p)

    last_err: Exception | None = None
    for p in candidates:
        if p in tried:
            continue
        tried.append(p)
        try:
            apply_text_position_interpolation(
                model, attr_path=p, new_max_positions=new_max_positions
            )
            print(f"[info] Applied text PI at '{p}' -> {new_max_positions}")
            break
        except Exception as e:
            last_err = e
            continue
    else:
        # No candidate succeeded
        msg = f"[warn] failed to apply PI; tried {tried}"
        if last_err is not None:
            msg += f"; last error: {last_err}"
        print(msg)
        return

    # Try to bump tokenizer model_max_length to avoid pre-truncation
    try:
        tok = getattr(processor, "tokenizer", None) or getattr(processor, "text_tokenizer", None)
        if tok is not None and hasattr(tok, "model_max_length"):
            tok.model_max_length = int(new_max_positions)
    except Exception:
        pass


def encode_text(query: str, processor, model, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        # Avoid tokenizer-level truncation if possible; padding not needed for single example
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
    # mat: [N, D] L2-normalized, vec: [D] L2-normalized
    scores = mat @ vec  # [N]
    if k >= len(scores):
        return np.argsort(-scores)
    idx = np.argpartition(scores, -k)[-k:]
    # sort these topk by score desc
    idx_sorted = idx[np.argsort(-scores[idx])]
    return idx_sorted


def evaluate(
    golden_path: Path,
    index_dir: Path | None,
    model_name: str,
    topk: int,
    text_pi: int,
    pi_attr_path: str,
    *,
    embeddings_file: Path | None = None,
    paths_json: Path | None = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    img_emb, img_paths = load_index(index_dir, embeddings_file=embeddings_file, paths_json=paths_json)
    print(f"Loaded index: embeddings={img_emb.shape}, paths={len(img_paths)} from {index_dir}")

    # Build canonical map path->row index
    canon_to_rows: Dict[str, List[int]] = {}
    for i, p in enumerate(img_paths):
        key = canonicalize_index_path(p)
        canon_to_rows.setdefault(key, []).append(i)

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    if text_pi and text_pi > 0:
        maybe_apply_text_pi(model, processor, text_pi, pi_attr_path)

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

        # Check membership by canonical form ignoring extension and leading ../
        gt_key = canonicalize_golden_path(gt_path)
        top_keys = {canonicalize_index_path(img_paths[i]) for i in top_idx.tolist()}
        ok = gt_key in top_keys
        if ok:
            success += 1
        else:
            misses.append((idx, gt_path))

    print(f"Success@{topk}: {success}/{total}")
    if misses:
        # Print a short list of misses for debugging
        preview = ", ".join(f"#{i}:{p}" for i, p in misses[:10])
        print(f"First misses: {preview}{' ...' if len(misses) > 10 else ''}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate golden queries retrieval@TopK using SigLIP-2 text encoder.")
    p.add_argument("--golden", default="tests/golden_data/golden_queries_by_index.json")
    p.add_argument("--index-dir", default="scripts/data/preprocess/siglip2_base_p16_384")
    p.add_argument("--embeddings-file", default=None, help="Path to embeddings.npy if not under --index-dir")
    p.add_argument("--paths-json", default=None, help="Path to paths.json matching embeddings file")
    p.add_argument("--model-name", default="google/siglip2-base-patch16-384")
    p.add_argument("--topk", type=int, default=100)
    p.add_argument("--text-pi", type=int, default=128, help="New max positions for text encoder (PI)")
    p.add_argument(
        "--pi-attr-path",
        default="text_model.embeddings.position_embeddings",
        help="Dot path to text positional nn.Embedding inside the HF model",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    evaluate(
        golden_path=Path(args.golden),
        index_dir=Path(args.index_dir) if args.index_dir else None,
        model_name=args.model_name,
        topk=args.topk,
        text_pi=args.text_pi,
        pi_attr_path=args.pi_attr_path,
        embeddings_file=Path(args.embeddings_file) if args.embeddings_file else None,
        paths_json=Path(args.paths_json) if args.paths_json else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
