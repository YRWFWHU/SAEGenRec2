# Contract: .item.json File Format

**Producer**: `SAEGenRec.data_process.preprocess`
**Consumer**: `SAEGenRec.sid_builder.text2emb`, `SAEGenRec.data_process.convert_dataset`

## Format

JSON object, UTF-8 encoding. Keys are string item_id, values are item metadata objects.

## Schema

```json
{
  "0": {
    "item_asin": "B00001",
    "title": "Industrial Widget",
    "description": "A high-quality widget for industrial use."
  },
  "42": {
    "item_asin": "B00042",
    "title": "Safety Goggles",
    "description": ""
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| item_asin | str | Yes | Amazon ASIN |
| title | str | Yes | Product title (1-20 words, no HTML) |
| description | str | Yes | Product description (may be empty) |

## Naming Convention

- `{dataset}.item.json`

## Notes

- Keys are string representations of integer item_id (0-indexed)
- Items without valid titles are excluded during preprocessing
- Titles are cleaned: `&quot;` → `"`, `&amp;` → `&`, stripped
