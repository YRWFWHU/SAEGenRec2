# Contract: .index.json File Format

**Producer**: `SAEGenRec.sid_builder.generate_indices`
**Consumer**: `SAEGenRec.data_process.convert_dataset`, `SAEGenRec.training.sft` (TokenExtender), `SAEGenRec.evaluation`

## Format

JSON object, UTF-8 encoding. Keys are string item_id, values are lists of SID token strings.

## Schema

```json
{
  "0": ["[a_42]", "[b_128]", "[c_7]"],
  "42": ["[a_15]", "[b_200]", "[c_33]"],
  "7": ["[a_42]", "[b_128]", "[c_7]"]
}
```

| Field | Type | Description |
|-------|------|-------------|
| key | str | String item_id (0-indexed) |
| value | list[str] | Ordered list of SID tokens, length = num_quantization_levels |

## SID Token Format

- Pattern: `[{level_prefix}_{code_index}]`
- Level prefixes are single letters: `a`, `b`, `c`, ... (for levels 0, 1, 2, ...)
- Code index: integer 0 to codebook_size-1
- Default: 3 levels, 256 codes each → tokens like `[a_0]` through `[a_255]`, `[b_0]` through `[b_255]`, `[c_0]` through `[c_255]`

## Naming Convention

- `{dataset}.index.json`

## Notes

- Collisions allowed: multiple items may map to the same SID token sequence
- All SID tokens from this file are added to the LLM tokenizer via TokenExtender
- Total unique tokens = num_levels × codebook_size (default: 3 × 256 = 768)
