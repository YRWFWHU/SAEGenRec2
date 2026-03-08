"""CLI 入口：python -m SAEGenRec.training <command> [args]"""

try:
    import fire
except ImportError as e:
    raise ImportError(
        "fire is required for CLI usage. Install with: pip install fire"
    ) from e

from SAEGenRec.training.rl import rl
from SAEGenRec.training.sft import sft

if __name__ == "__main__":
    fire.Fire({"sft": sft, "rl": rl})
