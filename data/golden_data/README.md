FFS-v3 Golden Data

- File: `data/golden_data/fashion_golden_queries_ko.json`
- Schema:
  - `상품 소개/구매 url`: product detail/purchase URL
  - `image_url`: direct product image URL
  - `user_query_ko`: real user-style Korean query description

Notes
- This slice is intended for retrieval sanity checks and small-scale evaluation.
- Keep additions append-only; avoid editing existing rows to preserve reproducibility.
- When referencing this file in configs, prefer environment-driven paths.
