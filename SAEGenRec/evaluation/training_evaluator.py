"""TrainingEvaluator：训练期间可选的推荐指标评估回调。"""

from typing import Dict, List, Optional

from loguru import logger
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class TrainingEvaluator(TrainerCallback):
    """在 SFT/RL 训练过程中周期性计算 HR@K/NDCG@K 指标。

    Parameters:
        eval_rec_steps: 每隔多少步评估一次（0 表示禁用）
        eval_rec_beams: 波束搜索宽度
        eval_rec_samples: 评估样本数（-1 表示全部）
        info_file: info TXT 路径
        test_csv: 测试 CSV 路径
        k_values: K 值列表，默认 [1, 5, 10]
        batch_size: 评估批大小
    """

    def __init__(
        self,
        eval_rec_steps: int = 0,
        eval_rec_beams: int = 10,
        eval_rec_samples: int = 200,
        info_file: str = "",
        test_csv: str = "",
        k_values: Optional[List[int]] = None,
        batch_size: int = 2,
        category: str = "",
        tokenizer=None,
    ):
        self.eval_rec_steps = eval_rec_steps
        self.eval_rec_beams = eval_rec_beams
        self.eval_rec_samples = eval_rec_samples
        self.info_file = info_file
        self.test_csv = test_csv
        self.k_values = k_values or [1, 5, 10]
        self.batch_size = batch_size
        self.category = category
        self._tokenizer = tokenizer  # 由 sft.py 在初始化时注入，避免依赖 callback kwargs

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        tokenizer=None,
        **kwargs,
    ):
        if self.eval_rec_steps <= 0:
            return
        if model is None:
            return
        tok = tokenizer if tokenizer is not None else self._tokenizer
        logger.info(f"[TrainingEvaluator] Epoch end — running recommendation eval...")
        self._run_eval(model, tok)

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        tokenizer=None,
        **kwargs,
    ):
        if self.eval_rec_steps <= 0:
            return
        if state.global_step % self.eval_rec_steps != 0:
            return
        if model is None:
            return
        tok = tokenizer if tokenizer is not None else self._tokenizer
        self._run_eval(model, tok)

    def _run_eval(self, model, tokenizer):
        """执行评估并记录指标。"""

        from SAEGenRec.evaluation.evaluate import evaluate_subset

        if not self.info_file or not self.test_csv:
            logger.warning("TrainingEvaluator: info_file or test_csv not set, skipping eval.")
            return

        logger.info(
            f"[TrainingEvaluator] Running evaluation on {self.eval_rec_samples} samples "
            f"with {self.eval_rec_beams} beams..."
        )
        try:
            metrics = evaluate_subset(
                model=model,
                tokenizer=tokenizer,
                test_csv=self.test_csv,
                info_file=self.info_file,
                num_beams=self.eval_rec_beams,
                n_samples=self.eval_rec_samples,
                k_values=self.k_values,
                batch_size=self.batch_size,
                category=self.category,
            )
            self._log_metrics(
                step=getattr(model, "_current_step", 0),
                total_steps=getattr(model, "_total_steps", 0),
                metrics=metrics,
            )
        except Exception as e:
            logger.warning(f"[TrainingEvaluator] Evaluation failed: {e}")

    def _log_metrics(self, step: int, total_steps: int, metrics: Dict[str, float]):
        """格式化输出指标。"""
        ks = self.k_values
        parts = []
        for k in ks:
            hr = metrics.get(f"HR@{k}", 0.0)
            ndcg = metrics.get(f"NDCG@{k}", 0.0)
            parts.append(f"HR@{k}={hr:.4f}")
            parts.append(f"NDCG@{k}={ndcg:.4f}")
        metric_str = ", ".join(parts)
        logger.info(f"[TrainingEvaluator] Step {step}/{total_steps}: {metric_str}")
