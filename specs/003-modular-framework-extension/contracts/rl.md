# CLI Contract: RL

**Module**: `python -m SAEGenRec.training`

## Commands

### rl

GRPO RL 训练（扩展现有命令）。

```bash
python -m SAEGenRec.training rl \
    --model_path=models/sft/Beauty/final_checkpoint \
    --train_csv=data/processed/Beauty.train.csv \
    [--eval_csv=data/processed/Beauty.valid.csv] \
    [--info_file=data/processed/info/Beauty.txt] \
    [--category=Beauty] \
    [--output_dir=models/rl/Beauty] \
    [--reward_type=rule] \
    [--reward_weights=1.0] \
    [--prompt_template=templates/sid_seq.txt] \
    [--num_generations=16] \
    [--beta=0.04] \
    [--sample=-1] \
    [--eval_rec=False] \
    [--eval_rec_steps=0.1] \
    [--eval_rec_beams=10] \
    [--eval_rec_samples=200]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model_path | str | **required** | SFT checkpoint 路径 |
| train_csv | str | **required** | 训练数据 CSV |
| eval_csv | str | `""` | 评估数据 CSV |
| info_file | str | `""` | SID → title 映射文件 |
| category | str | `""` | 商品类别名 |
| output_dir | str | `models/rl/{category}` | 输出目录 |
| reward_type | str | `rule` | 奖励函数名（支持 `+` 组合：`rule+semantic`） |
| reward_weights | str | `""` | 组合权重（逗号分隔：`0.6,0.4`） |
| prompt_template | str | `""` | prompt 模板路径（复用 SFT 模板机制） |
| num_generations | int | 16 | GRPO 每 prompt 生成数 |
| beta | float | 0.04 | KL 惩罚系数 |
| sample | int | -1 | 采样数（-1 = 全部） |
| eval_rec | bool | False | 训练期间推荐评估 |
| eval_rec_steps | float | 0.1 | 评估间隔 |
| eval_rec_beams | int | 10 | 评估 beam 数 |
| eval_rec_samples | int | 200 | 评估样本数 |

**Reward resolution**:
1. 解析 `reward_type`：按 `+` 分割获得奖励函数名列表
2. 从 `_REWARD_REGISTRY` 查找每个函数
3. 单个函数：直接调用
4. 多个函数：创建 `CombinedReward`，按 `reward_weights` 加权求和
5. 函数不存在：报错并列出所有可用函数

**Prompt template**:
- 若指定 `--prompt_template`，使用模板文件构造 prompt（替换 `{history}` 等占位符）
- 若未指定，保持现有行为（硬编码 prompt）

**Backward compatibility**:
- 所有现有参数保持不变
- 新增参数均有默认值，不影响现有调用方式

---

### list_rewards

列出所有已注册的奖励函数。

```bash
python -m SAEGenRec.training list_rewards
```

**Output** (stdout):
```
Available Reward Functions:
  rule      - 精确匹配奖励 (requires: -)
  prefix    - SID 前缀匹配部分奖励 (requires: -)
  ranking   - NDCG-aware 排名奖励 (requires: -)
  semantic  - 嵌入余弦相似度奖励 (requires: embeddings, item2idx)
  sasrec    - SASRec 模型评分奖励 (requires: sasrec_model, item2id)

Combination syntax: --reward_type=rule+semantic --reward_weights=0.6,0.4
```

## Reward Function Registration API

```python
from SAEGenRec.training.rewards import register_reward

@register_reward("my_reward")
def my_reward(
    predictions: List[str],
    target: str,
    **kwargs,
) -> List[float]:
    """Custom reward function."""
    return [1.0 if pred == target else 0.0 for pred in predictions]
```

**Requirements**:
- 签名必须接受 `predictions: List[str]`, `target: str`, `**kwargs`
- 返回 `List[float]`，长度等于 `len(predictions)`
- 每个值的范围建议 [0, 1]，但不强制
- 异常会被捕获，使用默认值 0.0
