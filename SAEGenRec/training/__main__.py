"""CLI 入口：python -m SAEGenRec.training <command> [args]"""

try:
    import fire
except ImportError as e:
    raise ImportError(
        "fire is required for CLI usage. Install with: pip install fire"
    ) from e

from SAEGenRec.training.prepare_sft import list_sft_tasks, prepare_sft
from SAEGenRec.training.rewards import list_rewards
from SAEGenRec.training.rl import rl
from SAEGenRec.training.sft import sft


def _list_rewards():
    """列出所有已注册的 reward 函数。"""
    rewards = list_rewards()
    print("Available Reward Functions:")
    for name, doc in rewards.items():
        print(f"  {name:<12} - {doc}")


if __name__ == "__main__":
    fire.Fire({
        "sft": sft,
        "rl": rl,
        "prepare_sft": prepare_sft,
        "list_sft_tasks": list_sft_tasks,
        "list_rewards": _list_rewards,
    })
