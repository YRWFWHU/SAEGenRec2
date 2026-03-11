# 强静态SID Baseline设计：SID-MSL + SAE端到端联合训练

**日期**: 2026-03-11
**目标**: 在动态SID之前，先构建一个强力静态SID baseline，融合MSL和端到端训练两个trick

---

## 核心洞察

### MSL在SID范式中的重新审视

批判性分析指出MSL"仅适用于文本标题推荐范式"。**这个结论需要修正。**

关键事实：
- SID token (`[f_0]`~`[f_1535]`) 被加入到LLM完整词表中
- Qwen2.5-0.5B 原始词表 = **151,936** tokens
- 加上SID后 = 151,936 + ~1,536 = **~153,472** tokens
- SFT训练时，softmax在**全部153K tokens**上计算
- 但SID生成位置的合法token仅有**几十到几百个**（由前缀树决定）

因此：
- **L1梯度主导问题在SID范式中同样严重**
- 模型花大量梯度学习"不要生成中文字/英文单词"这种无用知识
- MSL通过mask无关token，直接让梯度聚焦在"区分正确SID vs 错误SID"

### 端到端训练：SAE vs RQ-VAE的本质差异

UniGRec用RQ-VAE做端到端，但SAE有**根本性的结构差异**：

| 特性 | RQ-VAE | GatedSAE |
|------|--------|----------|
| 量化方式 | 多级残差，每级选1个码字 | 扁平化，选top-k个特征 |
| 可微化难点 | 离散码字选择 → soft probability | hard top-k → soft activation |
| 天然优势 | 层次结构利于自回归 | 特征激活值本身就是连续的 |
| 碰撞问题 | 每级256选1，组合有限 | d_sae维中选k个，组合空间大 |

**SAE端到端训练的独特优势**：
- SAE的feature_acts ∈ R^{d_sae} **本身就是连续的**，不需要像RQ-VAE那样做soft relaxation
- SAE的decoder权重W_dec直接给出每个特征的"语义方向"，可以与LLM embedding空间对齐
- top-k选择可以用Gumbel-Softmax或直通估计器(STE)平滑化

---

## 方案设计

### Trick 1: SID-Masked Softmax Loss (SID-MSL)

#### 原理

```
标准SFT Loss（当前实现）：
  L_CE = -log P(y_t | y_{<t}, x)
       = -log [exp(z_{y_t}) / Σ_{j=1}^{|V|} exp(z_j)]
  |V| = 153,472（全词表），但仅 ~100 个 token 是合法SID token

SID-MSL（改进）：
  L_MSL = -log P_valid(y_t | y_{<t}, x)
        = -log [exp(z_{y_t}) / Σ_{j ∈ V_valid(t)} exp(z_j)]
  V_valid(t): 由前缀树决定的当前步合法token集合
```

#### 关键设计细节

**问题1：何时应用MSL？**
- 仅在SID token位置应用（completion中的SID生成部分）
- prompt部分和非SID token仍用标准CE loss
- 实现：通过检测token是否在SID token集合中来决定

**问题2：如何获取V_valid(t)？**
- 已有 `build_prefix_tree()` 构建前缀树
- 训练时：对每个SID token位置，用已生成的prefix查前缀树得到合法next tokens
- 与约束解码不同：训练时用的是**ground truth prefix**（teacher forcing），不是模型预测

**问题3：梯度消失？**
- 合法token数量从153K缩减到~100，softmax更集中
- 可能出现P_valid(y_t)过高导致梯度消失（MSL论文的ATS问题）
- 解决：引入ATS温度策略，但需要适配SID场景

#### 实现方案

```python
# training/sid_msl.py (新增)

class SIDMaskedLoss(torch.nn.Module):
    """SID-Masked Softmax Loss: 在SID token位置仅对合法token计算softmax。"""

    def __init__(self, prefix_tree, sid_token_ids, tau=1.0, use_ats=True):
        """
        Args:
            prefix_tree: build_prefix_tree() 返回的 {prefix_tuple → [valid_next_ids]}
            sid_token_ids: set，所有SID token的id集合
            tau: softmax温度（ATS时为初始值）
            use_ats: 是否使用自适应温度
        """
        super().__init__()
        self.prefix_tree = prefix_tree
        self.sid_token_ids = set(sid_token_ids)
        self.tau = tau
        self.use_ats = use_ats

    def forward(self, logits, labels, input_ids):
        """
        Args:
            logits: (B, T, V) 模型输出logits
            labels: (B, T) ground truth token ids (-100 = ignore)
            input_ids: (B, T) 输入token ids（用于查prefix tree）
        Returns:
            loss: scalar
        """
        B, T, V = logits.shape
        loss = 0.0
        count = 0

        for b in range(B):
            sid_prefix = []  # 当前SID序列的prefix
            for t in range(T):
                if labels[b, t] == -100:
                    continue

                target_id = labels[b, t].item()

                if target_id in self.sid_token_ids:
                    # SID token位置 → 用MSL
                    prefix_key = tuple(sid_prefix)
                    valid_tokens = self.prefix_tree.get(prefix_key, None)

                    if valid_tokens and len(valid_tokens) > 1:
                        # 只在合法token上计算softmax
                        valid_logits = logits[b, t, valid_tokens]  # (n_valid,)

                        if self.use_ats:
                            tau = self._compute_ats_tau(valid_logits)
                        else:
                            tau = self.tau

                        valid_log_probs = F.log_softmax(valid_logits / tau, dim=0)

                        # target在valid_tokens中的位置
                        target_idx = valid_tokens.index(target_id)
                        loss -= valid_log_probs[target_idx]
                    else:
                        # 只有1个合法token或找不到prefix → 标准CE
                        loss -= F.log_softmax(logits[b, t] / self.tau, dim=0)[target_id]

                    sid_prefix.append(target_id)
                    count += 1
                else:
                    # 非SID token → 标准CE
                    loss -= F.log_softmax(logits[b, t], dim=0)[target_id]
                    count += 1
                    sid_prefix = []  # 重置SID prefix

        return loss / max(count, 1)

    def _compute_ats_tau(self, valid_logits):
        """自适应温度：基于valid logits的统计量。"""
        mu = valid_logits.mean()
        sigma = valid_logits.std() + 1e-8
        n = len(valid_logits)
        eta = 0.25  # 目标概率

        # ATS公式（MSL论文Lemma 2）
        tau = sigma * math.sqrt(2 * math.log(n / eta))
        return max(tau.item(), 0.1)  # 下界防止过小
```

**注意**: 上面是概念伪代码，实际实现需要向量化以避免逐token循环。高效实现思路：

```python
# 高效实现：批量构建mask矩阵
def compute_sid_msl_loss(logits, labels, valid_mask_matrix):
    """
    valid_mask_matrix: (B, T, V) bool tensor
        - SID位置：仅对应合法token为True
        - 非SID位置：全True（标准CE）
        - labels=-100位置：忽略
    """
    # 将非法位置设为-inf
    masked_logits = logits.masked_fill(~valid_mask_matrix, float('-inf'))
    # 标准CE在masked logits上
    loss = F.cross_entropy(
        masked_logits.view(-1, V),
        labels.view(-1),
        ignore_index=-100,
    )
    return loss
```

**预计算valid_mask_matrix**的方法：
- SFT数据中每个sample的SID序列是固定的（teacher forcing）
- 可在DataCollator中预计算每个位置的合法token mask
- 存储：稀疏矩阵（每个位置仅几十~几百个合法token）

#### 与现有代码的集成

需要修改的文件：
1. `training/sft.py` — 添加 `use_msl` 参数，替换loss计算
2. `evaluation/logit_processor.py` — 复用 `build_prefix_tree()`
3. 新增 `training/sid_msl.py` — MSL loss实现

```python
# sft.py 修改点
def _sft_jsonl(..., use_msl: bool = False, msl_tau: float = 1.0, ...):
    if use_msl:
        # 需要自定义Trainer覆盖compute_loss
        class MSLSFTTrainer(SFTTrainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                outputs = model(**inputs)
                logits = outputs.logits
                labels = inputs["labels"]

                # 构建valid mask（从prefix tree）
                loss = compute_sid_msl_loss(logits, labels, self.valid_mask_fn)
                return (loss, outputs) if return_outputs else loss
```

#### 预期效果

| 场景 | 标准CE | SID-MSL | 改进原因 |
|------|--------|---------|---------|
| SID位置第1个token | softmax over 153K | softmax over ~256 | 梯度不再浪费在区分SID vs 中文字 |
| SID位置中间token | softmax over 153K | softmax over ~50-100 | 前缀约束进一步缩小空间 |
| SID位置最后token | softmax over 153K | softmax over ~1-10 | 可能梯度消失 → 需要ATS |
| 非SID token | softmax over 153K | softmax over 153K | 不变 |

**保守估计**：由于SID词表比文本标题词表小得多（768 vs 100K+），改进幅度预计不如MSL论文的42%。但由于LLM全词表仍有153K，SID token仅占1%，MSL应仍能带来**显著改进**（预估10-30%）。

---

### Trick 2: SAE-LLM端到端联合训练

#### 核心思路

不照搬UniGRec（T5 + RQ-VAE），设计SAE特有的端到端训练方案：

**利用SAE的独特优势**：
1. SAE激活值本身是连续的（不需要soft relaxation）
2. SAE decoder权重给出每个特征的语义方向
3. SAE的top-k稀疏性与推荐的"关注少量关键特征"目标天然对齐

#### 架构：Tied SAE-LLM Embedding

```
核心创新：将SAE feature与LLM的SID token embedding绑定

SAE decoder:
  W_dec ∈ R^{d_sae × d_in}
  每个特征 f_i 的语义方向: w_i = W_dec[i] ∈ R^{d_in}

LLM SID token embedding:
  E_sid ∈ R^{d_sae × d_model}
  token [f_i] 的embedding: e_i = E_sid[i] ∈ R^{d_model}

绑定方式 — Projection Tie:
  E_sid = W_dec @ P   其中 P ∈ R^{d_in × d_model} 是可学习投影矩阵

效果：
  SAE特征方向 → 投影 → LLM embedding空间
  推荐loss反传 → 更新P → 间接更新SAE特征的LLM表示
  同时更新SAE权重 → 改变特征分解 → 改变SID分配
```

#### 训练流程

```
Phase 0: SAE预训练（已有，不变）
  item_emb → GatedSAE → feature_acts → L_recon + L_sparse
  → 得到初始SAE checkpoint

Phase 1: Embedding对齐预热（新增，1-2 epochs）
  目标：学习投影矩阵P，对齐SAE特征空间与LLM embedding空间
  方法：
    - 冻结SAE和LLM
    - 仅训练P
    - 损失：SID alignment tasks (sid→title, title→sid)
    - 确保P的质量

Phase 2: 联合SFT训练（核心创新）
  可训练参数：
    - SAE encoder+decoder（小学习率，lr_sae = 0.1 * lr_llm）
    - LLM LoRA适配器
    - 投影矩阵P

  前向过程：
    1. item_idx_i → E_item.npy[item_idx_i] → item_emb_i ∈ R^{d_in}
       （从全量物品 embedding 矩阵 lookup，训练开始前一次性加载到 GPU）
    2. item_emb_i → SAE.encode() → acts_i ∈ R^{d_sae}（连续激活）
    3. STE top-k（可微离散化）：
         topk_vals, topk_ids = topk(acts_i, k)
         hard_mask = scatter(topk_ids, 1.0)          ← 前向传 hard binary mask
         soft_mask = sigmoid(acts_i / τ)              ← 反向传 soft 梯度
         ste_mask  = soft_mask + (hard_mask − soft_mask).detach()
    4. 计算 k 个独立 soft SID embedding（不做 softmax 加权）：
         topk_embs_i = W_dec[topk_ids] @ P    # shape: (k, d_model)
         # 每个位置独立，与 hard SID 推理时 k 个 token 的 embedding lookup 完全对应
    5. topk_embs_i 替代离散 SID token embedding 输入 LLM（k 个 token 位置）
    6. LLM 生成 → logits → MSL loss（Trick 1）

  反向过程：
    L_total → ∂L/∂topk_embs → ∂topk_embs/∂W_dec → SAE decoder 更新
                             → ∂topk_embs/∂P → 投影矩阵更新
    L_total → ∂L/∂topk_embs → 通过 STE → ∂ste_mask/∂acts_i → ∂acts_i/∂W_enc（SAE encoder 更新）
    L_total → ∂L/∂θ_LoRA（LLM 更新）

  附加损失：
    L_total = L_MSL + λ_recon * L_SAE_recon + λ_entropy * L_feature_entropy

    L_SAE_recon: SAE重建损失（防止联合训练破坏SAE质量）
    L_feature_entropy: 鼓励特征使用均匀性（防止坍塌）

  温度退火（重新定位）：
    τ: 2.0 → 0.1 over training
    作用：控制 STE 中 sigmoid 的锐度，而非 embedding 的加权方式
    早期：sigmoid 较软，梯度均匀流向各特征，鼓励广泛探索
    后期：sigmoid 趋于阶跃函数，STE 更接近 hard binary，强化离散化
    注意：训练→推理 gap 由 Tied Embedding（E_sid[f_i] ≡ W_dec[i] @ P）消除，无需靠退火桥接

Phase 3: Hard SID固化 + 继续SFT（可选）
  - 用训练后的SAE重新生成hard SID（top-k + 去重）
  - 用新SID做1-2 epochs的标准SFT（fine-tune对齐）
  - 推理时使用discrete SID token
```

#### Soft SID Embedding的具体实现

```python
class SoftSIDEmbedding(torch.nn.Module):
    """将物品 embedding 经 SAE 转换为 soft SID embedding（k 个独立 token 位置）。

    item_emb 来源：全量物品 embedding 矩阵（从 .npy 加载），训练开始前一次性载入 GPU。
    top-k 离散化：使用 Straight-Through Estimator（STE），前向 hard binary，反向 soft sigmoid。
    soft/hard 无 gap：Tied Embedding 保证训练时每个位置向量 = 推理时对应 token embedding。
    """

    def __init__(
        self,
        sae,
        all_item_embeddings: torch.Tensor,
        projection_dim: int,
        k: int = 8,
        tau_init: float = 2.0,
        tau_min: float = 0.1,
    ):
        """
        Args:
            sae: GatedSAE（可训练，SAELens GatedTrainingSAE 或 inference SAE）
            all_item_embeddings: (n_items, d_in) 全量物品 embedding，从 .npy 预加载
            projection_dim: LLM embedding 维度 d_model
            k: top-k 特征数
            tau_init: STE sigmoid 初始温度（越大越软）
            tau_min: STE sigmoid 最小温度下界
        """
        super().__init__()
        self.sae = sae
        self.k = k
        self.tau = tau_init
        self.tau_min = tau_min

        # 全量物品 embedding 矩阵（register_buffer → 随模型转移 device，不参与梯度）
        self.register_buffer("item_embeddings", all_item_embeddings)

        d_in = sae.cfg.d_in
        self.proj = nn.Linear(d_in, projection_dim, bias=False)  # P 矩阵

    def forward(self, item_indices: torch.LongTensor):
        """
        Args:
            item_indices: (B,) 物品在 item_embeddings 中的行索引
        Returns:
            soft_sid_embs: (B, k, d_model) k 个独立 token 位置的 embedding
            topk_ids: (B, k) 所选特征的索引（用于 Phase 3 固化 SID）
        """
        # 1. 全量 embedding 矩阵 lookup（无梯度，仅前向）
        item_emb = self.item_embeddings[item_indices]  # (B, d_in)

        # 2. SAE 编码
        feature_acts = self.sae.encode(item_emb)  # (B, d_sae)

        # 3. STE top-k：前向 hard binary mask，反向 soft sigmoid 梯度
        _, topk_ids = torch.topk(feature_acts, self.k, dim=1)  # (B, k)

        hard_mask = torch.zeros_like(feature_acts)
        hard_mask.scatter_(1, topk_ids, 1.0)                          # 前向：binary 0/1

        soft_mask = torch.sigmoid(feature_acts / self.tau)             # 反向：连续可导

        ste_mask = soft_mask + (hard_mask - soft_mask).detach()        # STE 组合

        # STE mask 乘以激活值，使梯度能流回 SAE encoder
        masked_acts = feature_acts * ste_mask                          # (B, d_sae)

        # 4. 取 top-k 特征的 decoder 方向（W_dec: d_sae × d_in）
        W_dec = self.sae.W_dec                                         # (d_sae, d_in)
        topk_directions = W_dec[topk_ids]                              # (B, k, d_in)

        # 5. 投影到 LLM embedding 空间（Tied Embedding：E_sid[f_i] ≡ W_dec[i] @ P）
        soft_sid_embs = self.proj(topk_directions)                     # (B, k, d_model)

        # 不做 softmax 加权！每个位置独立，与推理时 hard SID 的 k 个 token embedding 完全对应
        # LLM 通过自注意力自由组合 k 个特征位置的信息

        _ = masked_acts  # 确保 SAE encoder 的 STE 梯度路径被保留

        return soft_sid_embs, topk_ids

    def anneal_tau(self, progress: float) -> None:
        """线性退火 STE sigmoid 温度。progress: 0.0→1.0"""
        self.tau = max(
            self.tau_min,
            self.tau_init * (1 - progress) + self.tau_min * progress,
        )
```

#### 与离散SID推理的桥接

```
item_emb 来源（训练 & 推理均相同）：
  全量物品 embedding 矩阵 E_item.npy（n_items × d_in）
  → 按 item_idx lookup，无需运行时重新计算

训练时（soft, STE）：
  item_idx → E_item.npy lookup → SAE.encode() → STE top-k
  → k 个 W_dec[f_i] @ P 向量（无 softmax 加权，k 个独立 token 位置）
  → 输入 LLM 的是 k 个连续 d_model 向量

推理时（hard）：
  item_idx → E_item.npy lookup → SAE.encode() → hard top-k
  → 离散 SID tokens [f_42][f_128]...[f_7]
  → 输入 LLM 的是标准 token embedding lookup：E_sid[f_i]
  → 约束 beam search（已有）

桥接关键（无 gap）：
  Tied Embedding 定义：E_sid[f_i] ≡ W_dec[i] @ P
  训练时第 j 个 STE 位置向量 = W_dec[topk_ids[j]] @ P
                               = E_sid[topk_ids[j]]
  → 两者逐 token 完全一致，soft 与 hard 之间无任何 gap
  → Phase 3 的标准 SFT 可选，用于在固化后的 SID 上进一步对齐
  → τ 退火控制 STE 梯度锐度，与 soft/hard gap 无关
```

#### 处理SAE SID重新分配问题

联合训练会改变SAE权重 → 物品的SID可能在训练过程中变化。

```
挑战：训练数据中的SID是基于初始SAE生成的，联合训练后SAE变了，SID可能不一致

解决方案A：Online SID Update（推荐）
  - 每 N 步用当前SAE重新编码所有物品
  - 更新训练数据中的SID
  - 类似UniGRec的SID演化分析

解决方案B：Soft-Only Mode
  - Phase 2完全不使用离散SID
  - 所有item embedding实时通过SAE → soft embedding
  - 避免SID不一致问题
  - 代价：需要在forward中实时运行SAE编码

解决方案C：Detached SID + Soft Residual
  - 保持原始离散SID不变
  - 加一个soft residual：h = embed([f_42]) + α * soft_correction(SAE(item_emb))
  - SAE通过residual提供额外信息
  - 最安全但改进可能最小
```

**推荐方案B（Soft-Only Mode）**：
- 最干净，无SID不一致问题
- 内存可接受：SAE编码是轻量操作
- Phase 3再固化为离散SID

---

## 两个Trick的协同效应

```
                    ┌─────────────────────────────────────┐
                    │         Joint Training Loop         │
                    │                                     │
  E_item.npy[idx] ► │  SAE.encode() → feature_acts        │
                    │       ↓                              │
                    │  STE top-k (hard fwd / soft bwd)    │
                    │       ↓                              │
                    │  W_dec[topk] @ P → k indep. embs     │
                    │       ↓                              │
  user_history ───► │  LLM([history; soft_SID_embs])       │
                    │       ↓                              │
                    │  logits ──────────────►  SID-MSL     │ ◄── prefix_tree
                    │       ↓                  (Trick 1)   │
                    │  L_MSL + L_recon + L_entropy         │
                    │       ↓                              │
                    │  backward → SAE + P + LoRA           │
                    │                                     │
                    └─────────────────────────────────────┘

Trick 1 (SID-MSL): 让梯度聚焦在"哪个SID更好"
Trick 2 (E2E):     让SAE的特征分解被推荐目标优化

协同：
  - MSL移除了153K词表中99%的噪声梯度
  - 剩余的"有效梯度"通过soft embedding流回SAE
  - SAE学到的特征分解直接由推荐效果驱动
  - 温度退火确保训练→推理平滑过渡
```

---

## 实施路线

### 阶段1：SID-MSL（1-2周）

```
T01: 实现 SID-MSL loss 模块
  - 复用 build_prefix_tree()
  - 实现高效的mask矩阵预计算
  - 在DataCollator中生成valid_token_mask

T02: 集成到SFT训练
  - 自定义SFTTrainer.compute_loss()
  - 添加 --use_msl 参数
  - 支持 --msl_tau 和 --use_ats 参数

T03: 消融实验（Beauty数据集）
  - Baseline: 标准CE SFT
  - + SID-MSL (固定τ=1.0)
  - + SID-MSL + ATS
  - 对比 HR@K, NDCG@K
```

### 阶段2：SAE端到端联合训练（3-4周）

```
T04: 实现 SoftSIDEmbedding 模块
  - SAE feature → projection → LLM embedding
  - 温度退火

T05: 修改SFT训练流程
  - Phase 1: Embedding对齐预热
  - Phase 2: 联合训练（SAE + LoRA + P）

T06: 实现SAE重建损失 + 特征均匀性损失

T07: Phase 3: Hard SID固化 + 继续SFT

T08: 消融实验（Beauty数据集）
  - Baseline: 标准CE SFT
  - + SID-MSL
  - + E2E Joint Training
  - + SID-MSL + E2E（Full）
```

### 阶段3：评估与对比（1-2周）

```
T09: 与文献baseline对比
  - vs 标准GatedSAE SID (我们的当前baseline)
  - vs RQ-VAE SID (TIGER方式)
  - vs MiniOneRec (开源框架)

T10: 分析
  - SID在联合训练前后的变化（参考UniGRec的SID演变分析）
  - SAE特征使用分布的变化
  - 碰撞率变化
  - 训练稳定性（loss曲线、梯度范数）
```

---

## 技术风险评估

| 风险 | 严重度 | 缓解策略 |
|------|--------|---------|
| MSL实现的效率 | 中 | 批量预计算mask矩阵；稀疏存储 |
| 联合训练不稳定 | 高 | SAE用小学习率(0.1x)；梯度裁剪；Phase 1预热 |
| Soft→Hard gap | 中 | 温度退火 + Phase 3 hard SFT |
| SAE质量退化 | 中 | L_recon 正则化；监控重建误差 |
| GPU内存不足 | 中 | SAE很小(~10M参数)；LLM用LoRA；gradient checkpointing |
| SID在训练中剧烈变化 | 中 | 方案B(Soft-Only)避免不一致；监控SID变化率 |

---

## 预期成果

### 最低预期（仅SID-MSL）
- HR@10 提升 10-20%（相对当前baseline）
- 训练收敛速度加快（梯度更高效）
- 无需额外模型或数据

### 理想预期（SID-MSL + E2E）
- HR@10 提升 20-40%
- SAE特征分解被推荐目标优化，更有推荐价值
- 碰撞率下降（联合训练优化SID分配）
- 为后续动态SID提供更强的基础

### 论文价值
- **SID-MSL**：首次将MSL思想适配到SID范式，分析SID词表结构下的L1主导效应
- **SAE E2E**：首次实现SAE tokenizer与LLM的端到端联合训练
- 两者结合 = 一个完整的strong baseline story

---

*方案设计日期: 2026-03-11*
