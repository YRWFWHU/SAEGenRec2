"""CLI 入口：python -m SAEGenRec.models <command> [args]"""

try:
    import fire
except ImportError as e:
    raise ImportError(
        "fire is required for CLI usage. Install with: pip install fire"
    ) from e

from SAEGenRec.models.sasrec import sasrec_train

if __name__ == "__main__":
    fire.Fire({"sasrec_train": sasrec_train})
