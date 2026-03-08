# Contract: Training CSV File Format

**Producer**: `SAEGenRec.data_process.convert_dataset`
**Consumer**: `SAEGenRec.datasets` (all dataset classes), `SAEGenRec.training.sft`, `SAEGenRec.training.rl`

## Format

CSV (comma-separated), UTF-8 encoding, first row is header.

## Schema

| Column | Type | Description |
|--------|------|-------------|
| user_id | str | Reviewer ID |
| history_item_sid | str | Space-separated SID sequences for history items (e.g., `[a_1][b_2][c_3] [a_4][b_5][c_6]`) |
| target_item_sid | str | SID sequence for target item (e.g., `[a_7][b_8][c_9]`) |
| history_item_title | str | Comma-separated titles for history items |
| target_item_title | str | Title of target item |
| history_item_id | str | Comma-separated integer item_ids for history |
| target_item_id | int | Integer item_id of target |

## Naming Convention

- `{dataset}.train.csv`
- `{dataset}.valid.csv`
- `{dataset}.test.csv`

## Notes

- List fields use comma separation within the CSV cell (escaped as needed)
- History length varies per sample (1 to max_history_len items)
- Split follows the same strategy (TO or LOO) as the source `.inter` files
