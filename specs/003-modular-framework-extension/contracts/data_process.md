# CLI Contract: data_process

**Module**: `python -m SAEGenRec.data_process`

## Commands

### download_images

下载物品图像（Amazon 元数据中的最高分辨率 MAIN 图像）。

```bash
python -m SAEGenRec.data_process download_images \
    --category=Beauty \
    [--data_dir=data/interim] \
    [--raw_data_dir=data/raw] \
    [--concurrency=8] \
    [--timeout=30]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| category | str | **required** | 商品类别名（如 Beauty） |
| data_dir | str | `data/interim` | 输出目录基路径 |
| raw_data_dir | str | `data/raw` | 原始数据目录（含 meta JSON） |
| concurrency | int | 8 | 最大并发下载数 |
| timeout | int | 30 | 单个请求超时秒数 |

**Input**: `data/raw/meta_{category}.json`（Amazon 元数据）
**Output**: `data/interim/{category}/images/{item_asin}.jpg`

**Behavior**:
- 从 `imageURLHighRes` 字段提取第一张（MAIN）图像 URL
- 跳过已存在的文件（断点续传）
- 记录下载成功/失败/跳过统计
- 网络错误（超时、429）自动重试 3 次

**Exit codes**: 0 成功（即使部分失败）| 1 全部失败或参数错误

---

### embed_text

提取文本嵌入（现有 `text2emb` 的重构版本）。

```bash
python -m SAEGenRec.data_process embed_text \
    --category=Beauty \
    [--model_name=sentence-transformers/all-MiniLM-L6-v2] \
    [--data_dir=data/interim] \
    [--batch_size=256] \
    [--device=cuda]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| category | str | **required** | 商品类别名 |
| model_name | str | `sentence-transformers/all-MiniLM-L6-v2` | 文本嵌入模型 |
| data_dir | str | `data/interim` | 数据目录（含 .item.json） |
| batch_size | int | 256 | 批量大小 |
| device | str | `cuda` | 计算设备 |

**Input**: `data/interim/{category}.item.json`
**Output**: `data/interim/{category}.emb-{model_slug}-text.npy`

**Behavior**:
- 使用 sentence-transformers 编码 title + description
- 输出 .npy 行序与 item.json 中 item_id 顺序一致
- model_slug 从 model_name 提取（去除 `sentence-transformers/` 前缀）

---

### extract_visual

提取视觉特征。

```bash
python -m SAEGenRec.data_process extract_visual \
    --category=Beauty \
    [--vision_model=openai/clip-vit-base-patch32] \
    [--data_dir=data/interim] \
    [--batch_size=64] \
    [--device=cuda]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| category | str | **required** | 商品类别名 |
| vision_model | str | `openai/clip-vit-base-patch32` | 视觉模型（HuggingFace ID） |
| data_dir | str | `data/interim` | 数据目录 |
| batch_size | int | 64 | 批量大小（GPU 显存敏感） |
| device | str | `cuda` | 计算设备 |

**Input**: `data/interim/{category}/images/{item_asin}.jpg` + `data/interim/{category}.item.json`（获取 item 顺序）
**Output**: `data/interim/{category}.emb-{model_slug}-visual.npy`

**Behavior**:
- 使用 HuggingFace `AutoModel` + `AutoProcessor` 加载视觉模型
- 按 item.json 中 item_id 顺序逐项提取特征
- 缺失图像的 item 使用零向量填充
- 日志记录缺失图像数量和百分比
- model_slug 从 vision_model 提取（去除 org 前缀 + 斜杠转短横线）

**Error handling**:
- 图像文件损坏：该 item 使用零向量，记录警告
- 模型加载失败：立即报错退出
