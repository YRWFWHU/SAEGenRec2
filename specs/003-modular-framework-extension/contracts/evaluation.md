# CLI Contract: Evaluation

**Module**: `python -m SAEGenRec.evaluation`

## Commands

### evaluate

独立评估（扩展现有命令，增加参数灵活性）。

```bash
python -m SAEGenRec.evaluation evaluate \
    --model_path=models/rl/Beauty \
    --test_csv=data/processed/Beauty.test.csv \
    --info_file=data/processed/info/Beauty.txt \
    [--output_dir=results/Beauty] \
    [--batch_size=4] \
    [--num_beams=50] \
    [--k_values=1,3,5,10,20] \
    [--max_new_tokens=256] \
    [--device=cuda]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model_path | str | **required** | 模型 checkpoint 路径 |
| test_csv | str | **required** | 测试数据 CSV |
| info_file | str | **required** | SID → title 映射文件 |
| output_dir | str | `results/{category}` | 输出目录 |
| batch_size | int | 4 | 推理批量大小 |
| num_beams | int | 50 | 约束波束搜索 beam 数 |
| k_values | str/tuple | `1,3,5,10,20` | HR@K/NDCG@K 的 K 值 |
| max_new_tokens | int | 256 | 最大生成 token 数 |
| device | str | `cuda` | 计算设备 |

**Output**:
- `{output_dir}/predictions.json`：每个样本的预测结果
- `{output_dir}/metrics.json`：HR@K、NDCG@K 指标

**Backward compatibility**: 现有参数和行为完全保持不变。

---

## Training-Time Evaluation (Callback)

训练期间推荐指标评估通过 HuggingFace `TrainerCallback` 实现，不作为独立 CLI 命令。

### TrainingEvaluator Callback

由 SFT/RL 训练的 `--eval_rec=True` 参数触发注册。

**Parameters** (从训练命令传入):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| eval_rec | bool | False | 启用训练期间评估 |
| eval_rec_steps | float | 0.1 | 评估间隔（总步数百分比，0.1 = 每 10%） |
| eval_rec_beams | int | 10 | 约束波束搜索 beam 数 |
| eval_rec_samples | int | 200 | 评估样本数（从 test.csv 随机抽样） |

**Behavior**:
1. 在 `Trainer.add_callback(TrainingEvaluator(...))` 注册
2. 每达到 `eval_rec_steps` 比例时触发评估
3. 从 test.csv 随机抽取 `eval_rec_samples` 条数据
4. 使用 `eval_rec_beams` 个 beam 执行约束波束搜索
5. 计算 HR@K、NDCG@K（K=[1, 5, 10]）
6. 日志输出指标（loguru + trainer.log）
7. 不影响 best model selection（仅供观测）

**Performance budget**:
- 200 samples × 10 beams ≈ 2-5 分钟（单 GPU）
- 建议 `eval_rec_steps=0.1`（每 10% 步一次）避免过多开销

**Log format**:
```
[TrainingEvaluator] Step 1000/10000: HR@1=0.012, HR@5=0.045, HR@10=0.078, NDCG@5=0.028, NDCG@10=0.035
```
