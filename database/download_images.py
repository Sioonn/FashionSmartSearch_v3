"""Download all image_url entries from musinsa_products_db.json.

This script scans the aggregated Musinsa JSON and downloads each image to
`database/images/<category>/` saving files as `<category>_<index>.<ext>`.
The index is taken from item["index"] if present; otherwise, the 1-based
position within the category list is used. File extension is derived from the
URL, defaulting to `.jpg` if missing.

Usage:
  python -m database.download_images

Notes:
  - Skips files that already exist.
  - Continues on individual download errors and prints a brief message.
  - Keeps things intentionally simple (no concurrency).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import requests


BASE_DIR = Path(__file__).resolve().parent
DB_JSON = BASE_DIR / "musinsa_products_db.json"
OUT_DIR = BASE_DIR / "images"


def iter_images(db: Dict[str, list[dict]]) -> Iterable[Tuple[str, int, str]]:
    """Yield (category, index, image_url) for each item.

    - index: item["index"] if present and int-like, else 1-based position.
    - Skips entries without a valid `image_url` string.
    """

    for category, items in db.items():
        if not isinstance(items, list):
            continue
        for pos, it in enumerate(items, start=1):
            url = it.get("image_url")
            if not (isinstance(url, str) and url):
                continue
            idx = it.get("index", pos)
            try:
                idx = int(idx)
            except Exception:
                idx = pos
            yield category, idx, url


def filename_from_url(url: str) -> str:
    return url.rstrip("/").split("/")[-1].split("?")[0]


def download_all() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with DB_JSON.open("r", encoding="utf-8") as f:
        db = json.load(f)

    session = requests.Session()
    total, saved = 0, 0

    for category, idx, url in iter_images(db):
        total += 1

        # derive extension from URL (fallback to .jpg)
        url_name = filename_from_url(url)
        ext = Path(url_name).suffix or ".jpg"

        cat_dir = OUT_DIR / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        out_path = cat_dir / f"{category}_{idx}{ext}"

        if out_path.exists():
            continue

        try:
            resp = session.get(url, timeout=20)
            resp.raise_for_status()
            out_path.write_bytes(resp.content)
            saved += 1
        except Exception as e:  # keep simple and resilient
            print(f"[skip] {category} :: {url} :: {e}")

    print(f"total_items={total} downloaded={saved}")


def main() -> None:
    download_all()


if __name__ == "__main__":
    main()
