# Contract: Info TXT File Format

**Producer**: `SAEGenRec.data_process.convert_dataset`
**Consumer**: `SAEGenRec.training.sft` (TokenExtender), `SAEGenRec.training.rl`, `SAEGenRec.evaluation` (prefix tree construction)

## Format

Tab-separated text, UTF-8 encoding, NO header row.

## Schema

| Column | Type | Description |
|--------|------|-------------|
| semantic_id | str | Full SID token sequence (e.g., `[a_42][b_128][c_7]`) |
| item_title | str | Product title |
| item_id | int | System-assigned integer ID |

## Example

```text
[a_42][b_128][c_7]	Industrial Widget	0
[a_15][b_200][c_33]	Safety Goggles	42
```

## Naming Convention

- `{dataset}.txt` (in `info/` directory)

## Notes

- One row per item in the dataset
- Used to construct the prefix tree (hash_dict) for constrained beam search
- Used by TokenExtender to extract all unique SID tokens for vocabulary extension
- Order matches the shuffled item order from preprocessing (random.seed(42))
