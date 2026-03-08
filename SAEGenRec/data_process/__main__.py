"""CLI 入口：python -m SAEGenRec.data_process <command> [args]"""

try:
    import fire
except ImportError as e:
    raise ImportError(
        "fire is required for CLI usage. Install with: pip install fire"
    ) from e

from SAEGenRec.data_process.convert_dataset import convert_dataset
from SAEGenRec.data_process.preprocess import preprocess

if __name__ == "__main__":
    fire.Fire(
        {
            "preprocess": preprocess,
            "convert_dataset": convert_dataset,
        }
    )
