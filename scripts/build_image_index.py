"""
Build or append a SigLIP-2 image index.

Usage:

- Fresh build for ALL categories
  python scripts/build_image_index.py --images-root database/images --out-dir scripts/data/preprocess/siglip2_base_p16_384

- Append mode: add all images from the 13 categories below, skipping duplicates already in the index
  python scripts/build_image_index.py --append --images-root database/images --out-dir scripts/data/preprocess/siglip2_base_p16_384

Default append categories:
  long_padding, other_tops, long_sleeve_tshirt, safari_jacket, shearling,
  short_sleeve_tshirt, sleeveless_tshirt, spring_coat, stadium_jacket,
  training_jacket, trucker_jacket, vest, zip_up_hoodie
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor
import numpy as np


def find_images(root: Path, categories: List[str]) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    paths: List[Path] = []
    for cat in categories:
        base = root / cat
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                paths.append(p)
    return sorted(paths)


def discover_categories(root: Path) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    cats: List[str] = []
    if not root.exists():
        return cats
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        has_img = False
        for fp in p.rglob("*"):
            if fp.is_file() and fp.suffix.lower() in exts:
                has_img = True
                break
        if has_img:
            cats.append(p.name)
    return cats


def load_existing_index(out_dir: Path) -> Tuple[np.ndarray, List[str], dict]:
    """Load existing embeddings/paths/meta from out_dir if present.

    Returns (embs[N,D] float32, paths[str], meta dict). If missing, returns empty placeholders.
    """
    emb_path = out_dir / "embeddings.npy"
    map_path = out_dir / "paths.json"
    meta_path = out_dir / "meta.json"

    if emb_path.exists() and map_path.exists() and meta_path.exists():
        embs = np.load(str(emb_path)).astype("float32", copy=False)
        paths: List[str] = json.loads(map_path.read_text(encoding="utf-8"))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return embs, paths, meta
    return np.empty((0, 0), dtype="float32"), [], {}


def load_model(model_name: str, device: torch.device):
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    return processor, model


def encode_images(
    paths: List[Path],
    processor,
    model,
    device: torch.device,
    batch_size: int,
    progress_every: int,
) -> tuple[torch.Tensor, List[Path], int]:
    """Encode images into L2-normalized embeddings.

    Returns (embeddings, success_paths, skipped_count).
    Prints 진행 현황(진행률/속도/ETA)을 배치마다 출력.
    """
    embeds: List[torch.Tensor] = []
    batch_imgs: List[Image.Image] = []
    batch_paths: List[Path] = []
    success_paths: List[Path] = []
    skipped = 0

    total = len(paths)
    t0 = time.time()

    for i, p in enumerate(paths, 1):
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            skipped += 1
            continue
        batch_imgs.append(img)
        batch_paths.append(p)

        if len(batch_imgs) == batch_size or i == total:
            with torch.no_grad():
                inputs = processor(images=batch_imgs, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                try:
                    feats = model.get_image_features(**inputs)  # [B, D]
                except AttributeError:
                    outs = model(**inputs)
                    feats = outs.image_embeds  # [B, D]
                feats = F.normalize(feats, dim=-1)
                embeds.append(feats.cpu())
                success_paths.extend(batch_paths)

            # progress
            done = len(success_paths) + skipped
            elapsed = time.time() - t0
            rate = (len(success_paths) / elapsed) if elapsed > 0 else 0.0
            eta = ((total - done) / rate) if rate > 0 else float("inf")
            if progress_every <= 0 or (len(success_paths) % progress_every == 0) or done == total:
                dim = int(feats.shape[1]) if feats.numel() else 0
                print(
                    f"[encode] {len(success_paths):,}/{total:,} ok, {skipped:,} skipped | dim={dim} | ",
                    f"elapsed={elapsed:.1f}s, rate={rate:.2f} img/s, eta~{eta:.1f}s",
                    flush=True,
                )

            batch_imgs.clear()
            batch_paths.clear()

    if embeds:
        all_embeds = torch.cat(embeds, dim=0)
    else:
        all_embeds = torch.empty((0, 0))

    return all_embeds, success_paths, skipped


def save_artifacts(
    out_dir: Path,
    model_name: str,
    categories: List[str],
    image_paths: List[Path],
    embeddings: torch.Tensor,
) -> Tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_path = out_dir / "embeddings.npy"
    map_path = out_dir / "paths.json"
    meta_path = out_dir / "meta.json"

    # Save embeddings as numpy for universal fallback
    np.save(emb_path, embeddings.numpy())
    map_path.write_text(json.dumps([str(p).replace("\\", "/") for p in image_paths], ensure_ascii=False, indent=2), encoding="utf-8")

    meta = {
        "model_name": model_name,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "vector_dim": int(embeddings.shape[1]) if embeddings.numel() else 0,
        "image_count": len(image_paths),
        "categories": categories,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # Try to save FAISS index (optional)
    try:
        import faiss  # type: ignore

        if embeddings.numel():
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings.numpy().astype("float32"))
            faiss_path = out_dir / "index.faiss"
            faiss.write_index(index, str(faiss_path))
        else:
            faiss_path = out_dir / "index.faiss"
    except Exception:
        faiss_path = out_dir / "index.faiss"  # not created; placeholder path

    return emb_path, map_path, meta_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build SigLIP-2 image vector index for database images.")
    p.add_argument("--images-root", default="database/images", help="Root directory containing category subfolders")
    p.add_argument(
        "--categories",
        nargs="*",
        default=[
            "long_padding",
            "other_tops",
            "long_sleeve_tshirt",
            "safari_jacket",
            "shearling",
            "short_sleeve_tshirt",
            "sleeveless_tshirt",
            "spring_coat",
            "stadium_jacket",
            "training_jacket",
            "trucker_jacket",
            "vest",
            "zip_up_hoodie",
            "sweatshirt",
            "knit_sweater",
            "hoodie",
        ],
        help="Category subfolders under images-root to index",
    )
    p.add_argument(
        "--append",
        action="store_true",
        help="Append vectors for --append-categories to an existing index in --out-dir (skips already indexed paths)",
    )
    p.add_argument(
        "--append-categories",
        nargs="*",
        default=[
            "long_padding",
            "other_tops",
            "long_sleeve_tshirt",
            "safari_jacket",
            "shearling",
            "short_sleeve_tshirt",
            "sleeveless_tshirt",
            "spring_coat",
            "stadium_jacket",
            "training_jacket",
            "trucker_jacket",
            "vest",
            "zip_up_hoodie",
        ],
        help="Categories to append when --append is enabled",
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--progress-every", type=int, default=100, help="진행 현황 출력 주기(성공 이미지 수 기준)")
    p.add_argument("--model-name", default="google/siglip2-base-patch16-384")
    p.add_argument("--out-dir", default="data/preprocess/siglip2_base_p16_384")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    images_root = Path(args.images_root)
    out_dir = Path(args.out_dir)
    
    if args.append:
        # Load existing artifacts
        ex_embs, ex_paths, ex_meta = load_existing_index(out_dir)
        print(
            f"Loaded existing index: embeddings={ex_embs.shape}, paths={len(ex_paths)} from {out_dir}",
            flush=True,
        )

        # Gather images from append categories
        target_cats = list(dict.fromkeys(args.append_categories))
        image_paths = find_images(images_root, target_cats)
        if not image_paths:
            print(f"No images found under {images_root} for categories {target_cats}", flush=True)
            return 0

        # Filter out already indexed paths
        ex_set = set(p.replace("\\", "/") for p in ex_paths)
        new_paths = [p for p in image_paths if str(p).replace("\\", "/") not in ex_set]
        if not new_paths:
            print("All images already indexed; nothing to append.", flush=True)
            return 0
        print(f"Found {len(new_paths):,} new images to append (out of {len(image_paths):,}).", flush=True)

        # Ensure model matches existing meta if present
        model_name = args.model_name
        if ex_meta.get("model_name") and ex_meta["model_name"] != model_name:
            print(
                f"[warn] Using existing model_name '{ex_meta['model_name']}' instead of CLI '{model_name}' to keep dim consistent.",
                flush=True,
            )
            model_name = ex_meta["model_name"]

        processor, model = load_model(model_name, device)
        embeddings, success_paths, skipped = encode_images(
            new_paths, processor, model, device, args.batch_size, args.progress_every
        )
        print(
            f"[summary] encoded {len(success_paths):,}/{len(new_paths):,} images, skipped {skipped:,}. vector_dim={int(embeddings.shape[1]) if embeddings.numel() else 0}",
            flush=True,
        )

        # Concatenate
        if ex_embs.size and embeddings.numel():
            if ex_embs.shape[1] != int(embeddings.shape[1]):
                print(
                    f"[error] Dimension mismatch: existing {ex_embs.shape[1]} vs new {int(embeddings.shape[1])}",
                    flush=True,
                )
                return 2
            all_embs = np.concatenate([ex_embs, embeddings.numpy()], axis=0)
        elif ex_embs.size:
            all_embs = ex_embs
        else:
            all_embs = embeddings.numpy()

        all_paths: List[Path] = [Path(p) for p in ex_paths] + success_paths

        # Merge categories for meta
        existing_cats = set(ex_meta.get("categories", []))
        merged_cats = sorted(existing_cats.union(target_cats))

        # Save back
        emb_tensor = torch.from_numpy(all_embs.astype("float32", copy=False))
        emb_path, map_path, meta_path = save_artifacts(out_dir, model_name, merged_cats, all_paths, emb_tensor)
        print(f"Saved appended embeddings: {emb_path}", flush=True)
        print(f"Saved appended path mapping: {map_path}", flush=True)
        print(f"Saved appended metadata: {meta_path}", flush=True)
        print("If FAISS is available, index.faiss was rebuilt including appended vectors.", flush=True)
        return 0

    # Fresh build path (overwrite)
    categories_to_use = list(dict.fromkeys(args.categories))
    image_paths = find_images(images_root, categories_to_use)
    if not image_paths:
        auto_cats = discover_categories(images_root)
        if auto_cats:
            print(
                f"No images for explicit categories {categories_to_use}. Auto-discovered categories: {auto_cats}",
                flush=True,
            )
            categories_to_use = auto_cats
            image_paths = find_images(images_root, categories_to_use)
        if not image_paths:
            print(
                f"No images found under {images_root} for categories {categories_to_use}",
                flush=True,
            )
            return 1

    processor, model = load_model(args.model_name, device)
    embeddings, success_paths, skipped = encode_images(
        image_paths, processor, model, device, args.batch_size, args.progress_every
    )

    print(
        f"[summary] encoded {len(success_paths):,}/{len(image_paths):,} images, skipped {skipped:,}. vector_dim={int(embeddings.shape[1]) if embeddings.numel() else 0}",
        flush=True,
    )

    emb_path, map_path, meta_path = save_artifacts(out_dir, args.model_name, categories_to_use, success_paths, embeddings)
    print(f"Saved embeddings: {emb_path}", flush=True)
    print(f"Saved path mapping: {map_path}", flush=True)
    print(f"Saved metadata: {meta_path}", flush=True)
    print("If FAISS is available, index.faiss will also be created in the same folder.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
