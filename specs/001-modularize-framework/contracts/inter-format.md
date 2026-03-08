# Contract: .inter File Format

**Producer**: `SAEGenRec.data_process.preprocess`
**Consumer**: `SAEGenRec.data_process.convert_dataset`

## Format

TSV (tab-separated values), UTF-8 encoding, first row is header.

## Schema

| Column | Type | Description |
|--------|------|-------------|
| user_id | str | Reviewer ID from Amazon data |
| item_id | int | System-assigned integer ID (0-indexed) |
| item_asin | str | Amazon product ASIN |
| timestamp | int | Unix timestamp of interaction |
| rating | float | User rating (1.0 - 5.0) |

## Naming Convention

- `{dataset}.train.inter`
- `{dataset}.valid.inter`
- `{dataset}.test.inter`

## Split Strategies

- **TO**: Rows sorted globally by timestamp, split 8:1:1 by position
- **LOO**: Per-user split (last → test, second-to-last → valid, rest → train)

## Example

```tsv
user_id	item_id	item_asin	timestamp	rating
A1234	0	B00001	1420070400	5.0
A1234	42	B00042	1420156800	4.0
A5678	7	B00007	1420243200	3.0
```
