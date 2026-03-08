# Contract: .review.json File Format

**Producer**: `SAEGenRec.data_process.preprocess`
**Consumer**: `SAEGenRec.sid_builder.text2emb`

## Format

JSON array, UTF-8 encoding. Each element is one interaction record that survived k-core filtering. Array ordering matches `.emb-{model}-review.npy` row indices.

## Schema

```json
[
  {
    "user_id": "A1234",
    "item_id": 0,
    "item_asin": "B00001",
    "timestamp": 1420070400,
    "rating": 5.0,
    "review_text": "Great product, highly recommend.",
    "summary": "Excellent"
  },
  {
    "user_id": "A1234",
    "item_id": 42,
    "item_asin": "B00042",
    "timestamp": 1420156800,
    "rating": 4.0,
    "review_text": "",
    "summary": ""
  }
]
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| user_id | str | Yes | Reviewer ID |
| item_id | int | Yes | System-assigned integer ID |
| item_asin | str | Yes | Amazon ASIN |
| timestamp | int | Yes | Unix timestamp |
| rating | float | Yes | Rating (1.0-5.0) |
| review_text | str | Yes | Review body (empty string if missing) |
| summary | str | Yes | Review summary (empty string if missing) |

## Naming Convention

- `{dataset}.review.json`

## Key Constraint

- Row index `i` in this file corresponds to row `i` in `{dataset}.emb-{model}-review.npy`
- Keyed by `(user_id, item_asin, timestamp)` triple — unique per interaction
- Current downstream modules (SFT/RL/eval) do NOT consume this file; it is an extension point
