"""CLI 入口：python -m SAEGenRec.evaluation <command> [args]"""

try:
    import fire
except ImportError as e:
    raise ImportError(
        "fire is required for CLI usage. Install with: pip install fire"
    ) from e

from SAEGenRec.evaluation.evaluate import evaluate

if __name__ == "__main__":
    fire.Fire({"evaluate": evaluate})
