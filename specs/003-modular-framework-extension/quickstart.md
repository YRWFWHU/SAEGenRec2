# Quickstart: 模块化框架扩展

**Branch**: `003-modular-framework-extension` | **Date**: 2026-03-09

## 场景 1: 端到端多模态推荐（文本 + 视觉 SID）

从原始数据到 RL 训练的完整流程。

```bash
# 1. 数据预处理（现有）
make preprocess CATEGORY=Beauty

# 2. 下载物品图像
make download_images CATEGORY=Beauty CONCURRENCY=8

# 3. 提取文本嵌入
make embed_text CATEGORY=Beauty TEXT_MODEL=sentence-transformers/all-MiniLM-L6-v2

# 4. 提取视觉特征
make extract_visual CATEGORY=Beauty VISION_MODEL=openai/clip-vit-base-patch32

# 5. 使用文本嵌入构建 RQ-VAE SID
make build_sid CATEGORY=Beauty METHOD=rqvae EMB_PATH=data/interim/Beauty.emb-all-MiniLM-L6-v2-text.npy

# 6. 转换数据集格式
make convert CATEGORY=Beauty

# 7. 预构建 SFT 数据
make prepare_sft CATEGORY=Beauty SID_TYPE=rqvae TASK=sid_seq

# 8. SFT 训练
make sft CATEGORY=Beauty MODEL_PATH=models/llm \
    SFT_DATA_DIR=data/processed/rqvae/Amazon/Beauty/sid_seq

# 9. RL 训练
make rl CATEGORY=Beauty MODEL_PATH=models/sft/Beauty/final_checkpoint

# 10. 评估
make evaluate CATEGORY=Beauty MODEL_PATH=models/rl/Beauty
```

**验证点**:
- 步骤 2 后：`data/interim/Beauty/images/` 包含 JPEG 文件
- 步骤 3 后：`data/interim/Beauty.emb-all-MiniLM-L6-v2-text.npy` 存在
- 步骤 4 后：`data/interim/Beauty.emb-clip-vit-base-patch32-visual.npy` 存在
- 步骤 5 后：`data/interim/Beauty.index.json` 存在
- 步骤 7 后：`data/processed/rqvae/Amazon/Beauty/sid_seq/train.jsonl` 存在

---

## 场景 2: GatedSAE SID 对比实验

使用 GatedSAE 替代 RQ-VAE 生成 SID，对比效果。

```bash
# 使用已有的文本嵌入
make build_sid CATEGORY=Beauty METHOD=gated_sae K=8

# 预构建 SFT 数据（使用 gated_sae SID）
make prepare_sft CATEGORY=Beauty SID_TYPE=gated_sae TASK=sid_seq

# SFT 训练
make sft CATEGORY=Beauty MODEL_PATH=models/llm \
    SFT_DATA_DIR=data/processed/gated_sae/Amazon/Beauty/sid_seq \
    OUTPUT_DIR=models/sft/Beauty_gated_sae

# 评估
make evaluate CATEGORY=Beauty MODEL_PATH=models/sft/Beauty_gated_sae/final_checkpoint
```

---

## 场景 3: 多阶段渐进式 SFT（Curriculum Learning）

先学习 SID 语义映射，再精调序列推荐。

```bash
# 预构建各阶段数据
make prepare_sft CATEGORY=Beauty SID_TYPE=rqvae TASK=item_feat
make prepare_sft CATEGORY=Beauty SID_TYPE=rqvae TASK=sid_to_title
make prepare_sft CATEGORY=Beauty SID_TYPE=rqvae TASK=fusion
make prepare_sft CATEGORY=Beauty SID_TYPE=rqvae TASK=sid_seq

# 阶段 1: item_feat（title → SID，学习 SID token 语义）
make sft MODEL_PATH=models/llm \
    SFT_DATA_DIR=data/processed/rqvae/Amazon/Beauty/item_feat \
    OUTPUT_DIR=models/sft/stage1 \
    FREEZE_LLM=True

# 阶段 2: fusion（历史 title+SID → 目标 SID，学习融合推荐）
make sft MODEL_PATH=models/sft/stage1/final_checkpoint \
    SFT_DATA_DIR=data/processed/rqvae/Amazon/Beauty/fusion \
    OUTPUT_DIR=models/sft/stage2

# 阶段 3: sid_seq（历史 SID → 目标 SID，精调序列推荐）
make sft MODEL_PATH=models/sft/stage2/final_checkpoint \
    SFT_DATA_DIR=data/processed/rqvae/Amazon/Beauty/sid_seq \
    OUTPUT_DIR=models/sft/stage3 \
    EVAL_REC=True EVAL_REC_STEPS=0.1
```

**验证点**:
- 阶段 1 使用 `FREEZE_LLM=True`，仅训练 SID token embedding
- 阶段 2 从阶段 1 checkpoint 加载，自动跳过 SID token 扩展
- 阶段 3 启用训练期间评估，日志每 10% 步输出 HR@K/NDCG@K

---

## 场景 4: 自定义奖励函数实验

注册新奖励函数并用于 RL 训练。

```python
# my_rewards.py
from SAEGenRec.training.rewards import register_reward

@register_reward("length_penalty")
def length_penalty_reward(predictions, target, **kwargs):
    """惩罚过长或过短的预测。"""
    target_len = len(target)
    return [
        max(0.0, 1.0 - abs(len(p) - target_len) / target_len)
        for p in predictions
    ]
```

```bash
# 使用组合奖励：rule(0.7) + length_penalty(0.3)
make rl CATEGORY=Beauty MODEL_PATH=models/sft/Beauty/final_checkpoint \
    REWARD_TYPE=rule+length_penalty REWARD_WEIGHTS=0.7,0.3

# 查看所有可用奖励函数
make list_rewards
```

---

## 场景 5: 训练期间评估监控

在 SFT 和 RL 训练中启用推荐指标实时监控。

```bash
# SFT + 训练期间评估（每 10% 步，200 样本，10 beams）
make sft CATEGORY=Beauty MODEL_PATH=models/llm \
    SFT_DATA_DIR=data/processed/rqvae/Amazon/Beauty/sid_seq \
    EVAL_REC=True EVAL_REC_STEPS=0.1 EVAL_REC_BEAMS=10 EVAL_REC_SAMPLES=200

# RL + 训练期间评估
make rl CATEGORY=Beauty MODEL_PATH=models/sft/Beauty/final_checkpoint \
    EVAL_REC=True EVAL_REC_STEPS=0.2

# 不启用评估（默认行为，与当前完全一致）
make sft CATEGORY=Beauty MODEL_PATH=models/llm \
    SFT_DATA_DIR=data/processed/rqvae/Amazon/Beauty/sid_seq
```

**预期日志输出**:
```
[TrainingEvaluator] Step 500/5000: HR@1=0.008, HR@5=0.032, HR@10=0.055, NDCG@5=0.019, NDCG@10=0.026
[TrainingEvaluator] Step 1000/5000: HR@1=0.012, HR@5=0.045, HR@10=0.078, NDCG@5=0.028, NDCG@10=0.035
```

---

## 场景 6: 自定义 SFT 任务类型

注册新的 SFT 任务并生成数据。

```python
# my_tasks.py
from SAEGenRec.datasets.task_registry import register_sft_task

@register_sft_task("review_predict")
class ReviewPredictTask:
    """根据用户评论预测物品 SID。"""
    default_template = "templates/review_predict.txt"
    required_inputs = ["csv", "index_json", "review_json"]
    required_placeholders = ["{review}", "{category}"]

    def build_examples(self, csv_data, index_json, **resources):
        examples = []
        for _, row in csv_data.iterrows():
            # ... 构造 prompt-completion 对
            examples.append({"prompt": prompt, "completion": completion})
        return examples
```

```bash
# 生成自定义任务数据
make prepare_sft CATEGORY=Beauty SID_TYPE=rqvae TASK=review_predict

# 查看所有可用任务
make list_sft_tasks
```
