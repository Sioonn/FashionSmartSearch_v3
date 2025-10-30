"""
Download golden-data images to a local folder and ensure JSON has indices.

Usage
-----
python data/golden_data/download_golden_images.py \
  --input-json data/golden_data/fashion_golden_queries_ko.json \
  --output-dir data/golden_data/image \
  [--force] [--timeout 15] [--concurrency 4]

Notes
- Adds or normalizes a 1-based `index` field per item (in file order).
- Saves each image as `{output_dir}/{index}.jpg` strictly in JSON order
  (indices are normalized to 1..N in file order).
- Safe to re-run; existing files are skipped unless `--force` is set.

This module touches data I/O; keep paths configurable via CLI/env.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import mimetypes
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests


def _target_path(out_dir: Path, idx: int) -> Tuple[Path, Path]:
    # Save strictly as {index}.jpg
    final_path = out_dir / f"{idx}.jpg"
    tmp_path = out_dir / f"{idx}.part"
    return tmp_path, final_path


def _download_one(item: Dict[str, Any], out_dir: Path, timeout: int, force: bool) -> Tuple[int, str | None]:
    idx = int(item["index"])
    url = item["image_url"]
    try:
        if not force:
            final_exists = (out_dir / f"{idx}.jpg").is_file()
            if final_exists:
                return idx, None
        resp = requests.get(url, stream=True, timeout=timeout)
        resp.raise_for_status()
        tmp_path, final_path = _target_path(out_dir, idx)
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        os.replace(tmp_path, final_path)
        return idx, None
    except Exception as e:  # noqa: BLE001
        return idx, str(e)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-json",
        default=os.getenv(
            "GOLDEN_DATA_JSON", "data/golden_data/fashion_golden_queries_ko.json"
        ),
        help="Path to the golden JSON list.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("GOLDEN_DATA_IMAGE_DIR", "data/golden_data/image"),
        help="Where to save images.",
    )
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    in_path = Path(args.input_json)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(in_path, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    # Normalize/add 1-based indices strictly following current file order
    for i, item in enumerate(data, start=1):
        item["index"] = i

    # Persist updated JSON (pretty, UTF-8)
    with open(in_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Download images concurrently
    tasks = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        # Schedule downloads in ascending index order for predictability
        for item in sorted(data, key=lambda d: int(d.get("index", 0))):
            tasks.append(
                ex.submit(_download_one, item, out_dir, args.timeout, args.force)
            )
        errors: List[Tuple[int, str]] = []
        for fut in concurrent.futures.as_completed(tasks):
            idx, err = fut.result()
            if err:
                errors.append((idx, err))

    if errors:
        print("Some downloads failed:", file=sys.stderr)
        for idx, err in sorted(errors):
            print(f"  index={idx}: {err}", file=sys.stderr)
        return 1

    print(f"Saved images to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
