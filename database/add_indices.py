"""Add per-item indices within each category list in musinsa_products_db.json.

Usage:
  python -m database.add_indices
  # or
  python database/add_indices.py

This overwrites the JSON in place and adds an integer field `index`
starting from 1 for each item in every list value.
"""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    json_path = Path(__file__).with_name("musinsa_products_db.json")
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # For each category (key), if the value is a list, attach 1-based indices.
    changed = 0
    for key, value in (data.items() if isinstance(data, dict) else []):
        if isinstance(value, list):
            for idx, item in enumerate(value, start=1):
                if isinstance(item, dict):
                    item["index"] = idx
                    changed += 1

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"indexed_items={changed}")


if __name__ == "__main__":
    main()

