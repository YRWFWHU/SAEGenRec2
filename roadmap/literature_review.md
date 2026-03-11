# Literature Review: 基于稀疏自编码器的动态语义ID生成式推荐

## 研究主题

在基于稀疏自编码器（SAE）的生成式推荐模型上，解决当前 RQ-VAE / RQ-KMeans 方案中物品 SID 静态不变的问题，通过显式反馈（如评论）动态适配物品语义ID，以捕捉不同用户对物品的差异化偏好。

---

## 1. 生成式推荐系统（Generative Recommendation）

### 1.1 基础范式与经典模型

| # | 论文 | 会议/期刊 | 年份 | 核心贡献 |
|---|------|-----------|------|----------|
| 1 | **P5: Pretrain, Personalized Prompt, and Predict Paradigm for Recommendation** (Geng et al.) | RecSys | 2022 | 首个统一生成式推荐框架，多任务训练，使用数字 item ID |
| 2 | **TIGER: Recommender Systems with Generative Retrieval** (Rajput et al.) | NeurIPS | 2023 | 提出 Semantic ID 概念，使用 SentenceT5 + RQ-VAE 将物品编码为离散token序列 |
| 3 | **LC-Rec: Adapting Large Language Models by Integrating Collaborative Semantics for Recommendation** (Zheng et al.) | ICDE | 2024 | 学习式VQ，统一语义映射 + 协同语义注入LLM |
| 4 | **LETTER: A LEarnable Tokenizer for generaTivE Recommendation** (Wang et al.) | arXiv | 2024 | 集成层次语义、协同信号、code分配多样性的可学习tokenizer |
| 5 | **ColaRec: Content-Based Collaborative Generation for Recommender Systems** (Wang et al.) | arXiv | 2024 | 基于GNN的CF模型生成协同GID，内容+协同双路生成 |
| 6 | **TokenRec: Learning to Tokenize ID for LLM-based Generative Recommendations** (Fan et al.) | TKDE | 2024 | 从CF掩码表示量化离散token，高效top-K生成检索 |
| 7 | **BIGRec: Grounding Generated Identifiers for Generative Recommendation** | arXiv | 2024 | L2距离将生成token序列映射到有效物品 |
| 8 | **PAP-REC: Personalized Automatic Prompt for Recommendation Language Model** (Li et al.) | arXiv | 2024 | 梯度搜索自动个性化prompt |
| 9 | **SEATER: Tree-based Numeric IDs for Generative Recommendation** | arXiv | 2023 | 树结构数字ID改进 |
| 10 | **IDGenRec: Zero-shot Generative Recommendation with Pre-trained Foundation Model** | arXiv | 2024 | 零样本生成式推荐，预训练基座模型 |

### 1.2 近期前沿方法 (2025)

| # | 论文 | 会议/期刊 | 年份 | 核心贡献 |
|---|------|-----------|------|----------|
| 11 | **EAGER: Two-Stream Generative Recommender with Behavior-Semantic Collaboration** (Ye et al.) | KDD | 2024 | 双流生成框架，行为+语义并行解码，全局对比 + 语义迁移 |
| 12 | **UNGER: Generative Recommendation with a Unified Code via Semantic and Collaborative Integration** (Xiao et al.) | TOIS | 2025 | 统一编码融合协同+语义，跨模态对齐 + 知识蒸馏，参数量仅EAGER一半 |
| 13 | **GRAM: Generative Recommendation via Semantic-aware Augmented Multi-modal** | ACL | 2025 | 语义感知多模态增强的生成式推荐 |
| 14 | **OneRec: Unifying Retrieve and Rank with Generative Recommender and Preference Alignment** (Deng et al.) | arXiv | 2025 | 首个在真实场景显著超越传统级联系统的端到端生成式推荐 |
| 15 | **MiniOneRec: An Open-Source Framework for Scaling Generative Recommendation** (Kong et al.) | arXiv | 2025 | 首个完全开源的生成式推荐框架，含SID构建+SFT+RL全流程 |
| 16 | **SynerGen: Contextualized Generative Recommender for Unified Search and Recommendation** (Amazon) | arXiv | 2025 | 统一搜索+推荐的decoder-only生成式推荐，时间感知旋转位置编码 |
| 17 | **GRID: Generative Recommendation with Semantic IDs** (Ju et al., Snap Research) | CIKM | 2025 | 开源benchmark框架，系统性评估生成式推荐各组件影响 |
| 18 | **Generative Recommendation with Semantic IDs: A Practitioner's Handbook** (Snap Research) | CIKM | 2025 | 实践指南，含完整的SID生成式推荐流水线 |

### 1.3 综述文章

| # | 论文 | 会议/期刊 | 年份 | 核心贡献 |
|---|------|-----------|------|----------|
| 19 | **A Survey of Generative Search and Recommendation in the Era of Large Language Models** (Li et al.) | arXiv | 2024 | 生成式搜索与推荐全面综述 |
| 20 | **Large Language Models for Generative Recommendation: A Survey and Visionary Discussions** | LREC-COLING | 2024 | LLM生成式推荐综述 |
| 21 | **Generative Recommendation: A Survey of Models, Systems, and Applications** | arXiv | 2025 | 模型、系统、应用全面综述 |
| 22 | **From Feature-Based, Generative to Agentic Paradigms** | arXiv | 2025 | 推荐系统范式演变：特征→生成→智能体 |
| 23 | **From Matching to Generation: A Survey on Generative Information Retrieval** | TOIS | 2025 | 生成式信息检索综述 |

---

## 2. 语义ID与向量量化 (Semantic ID & Vector Quantization)

### 2.1 量化方法

| # | 论文 | 会议/期刊 | 年份 | 核心贡献 |
|---|------|-----------|------|----------|
| 24 | **VQ-VAE: Neural Discrete Representation Learning** (Van den Oord et al.) | NeurIPS | 2017 | 向量量化变分自编码器，离散表示学习基础 |
| 25 | **SoundStream / RQ-VAE: Residual Quantization** (Zeghidour et al.) | TASLP | 2022 | 残差量化VAE，多级codebook逐级量化残差 |
| 26 | **VQ-Rec: Learning Vector-Quantized Item Representation for Transferable Sequential Recommenders** (Hou et al.) | WWW | 2023 | OPQ量化文本编码，text→code→representation的可迁移方案 |
| 27 | **CoST: Contrastive Quantization based Semantic Tokenization** (Fan et al.) | RecSys | 2024 | 对比量化语义token化，捕获语义+关系信息，NDCG@5提升44% |
| 28 | **HiD-VAE: Interpretable Generative Recommendation via Hierarchical and Disentangled Semantic IDs** | arXiv | 2025 | 层次解耦SID，用语义标签监督各量化层（类别→子类别），解决ID碰撞 |
| 29 | **SIDE: Semantic ID Embedding for Effective Learning from Sequences** | AdKDD | 2025 | VQ Fusion多任务框架 + DPCA量化 + 无参数SID-to-embedding转换，工业级2.4x NE提升 |
| 30 | **SimCIT: A Simple Contrastive Framework of Item Tokenization for Generative Recommendation** | arXiv | 2025 | 简单对比tokenization框架，多模态知识对齐+语义量化 |
| 31 | **ACERec: Unleash the Potential of Long Semantic IDs for Generative Recommendation** | arXiv | 2025 | 注意力token合并器将长SID压缩为紧凑latent，双粒度目标 |
| 32 | **RPG: Generating Long Semantic IDs in Parallel** | arXiv | 2025 | 并行生成所有SID token，解耦解码步数与SID长度 |
| 33 | **Rethinking Generative Recommender Tokenizer: Recsys-Native Encoding and Semantic Quantization Beyond LLMs** | arXiv | 2025 | 重新思考推荐原生编码与语义量化 |

### 2.2 向量量化综述

| # | 论文 | 会议/期刊 | 年份 | 核心贡献 |
|---|------|-----------|------|----------|
| 34 | **Vector Quantization for Recommender Systems: A Review and Outlook** (Liu et al.) | arXiv | 2024 | VQ在推荐系统的全面综述，分pre/in/post-processing三类 |

---

## 3. ★ 动态/个性化语义ID（核心相关）

> **与本研究idea最直接相关的文献**

| # | 论文 | 会议/期刊 | 年份 | 核心贡献 |
|---|------|-----------|------|----------|
| 35 | **Pctx: Tokenizing Personalized Context for Generative Recommendation** | ICLR | 2025 | ★ **最相关**：突破静态tokenization限制，同一物品在不同用户上下文下生成不同SID，条件化token化 |
| 36 | **MMQ: Multimodal Mixture-of-Quantization Tokenization for Semantic ID Generation and User Behavioral Adaptation** | arXiv | 2025 | ★ 多专家量化 + 行为感知微调，动态适配SID以反映用户交互信号 |
| 37 | **AgentDR: Dynamic Recommendation with Implicit Item-Item Relations via LLM-based Agents** | arXiv | 2025 | LLM智能体动态推理物品间隐式关系，替代/互补候选生成 |
| 38 | **EmerFlow: LLM-Empowered Representation Learning for Emerging Item Recommendation** | arXiv | 2025 | LLM增强新兴物品表示：特征增强→空间对齐→元学习精炼 |
| 39 | **Sensory-Aware Sequential Recommendation via Review-Distilled Representations** (ASEGR) | arXiv | 2025 | ★ LLM从评论中提取感官属性→紧凑感官embedding注入序列推荐 |
| 40 | **DeepInterestGR: Mining Deep Multi-Interest Using Multi-Modal LLMs for Generative Recommendation** | arXiv | 2025 | 多模态LLM挖掘深层多兴趣 |

---

## 4. 稀疏自编码器 (Sparse Autoencoder)

### 4.1 SAE 架构与可解释性

| # | 论文 | 会议/期刊 | 年份 | 核心贡献 |
|---|------|-----------|------|----------|
| 41 | **Towards Monosemanticity: Decomposing Language Models With Dictionary Learning** (Anthropic) | Anthropic | 2023 | 字典学习分解LM，单义特征提取 |
| 42 | **Sparse Autoencoders Find Highly Interpretable Features in Language Models** (Cunningham et al.) | ICLR | 2024 | SAE发现LM中高度可解释特征 |
| 43 | **Improving Sparse Decomposition of Language Model** (Gated SAE, Rajamanoharan & Nanda) | NeurIPS | 2024 | Gated SAE架构，改进稀疏性/重建精度权衡 |
| 44 | **Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU SAE** (DeepMind) | NeurIPS | 2024 | JumpReLU SAE，优于Gated SAE的重建保真度 |
| 45 | **BatchTopK Sparse Autoencoders** (Bussmann et al.) | arXiv | 2024 | BatchTopK SAE，改进稀疏性/重建权衡 + 训练稳定性 |
| 46 | **Scaling and Evaluating Sparse Autoencoders** (TopK SAE, OpenAI) | arXiv | 2024 | OpenAI TopK SAE，大规模训练与评估 |
| 47 | **SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in LM Interpretability** | arXiv | 2025 | SAE综合评测基准 |

### 4.2 SAE 应用于推荐系统

| # | 论文 | 会议/期刊 | 年份 | 核心贡献 |
|---|------|-----------|------|----------|
| 48 | **SAE4Rec: Sparse Autoencoders for Sequential Recommendation Models: Interpretation and Flexible Control** (Klenitskiy et al.) | RecSys | 2025 | ★ SAE应用于序列推荐模型，可解释特征发现 + 灵活行为控制 |
| 49 | **Understanding Internal Representations of Recommendation Models with Sparse Autoencoders** | arXiv | 2024 | SAE理解推荐模型内部表示 |

### 4.3 SAE 用于模型控制与编辑

| # | 论文 | 会议/期刊 | 年份 | 核心贡献 |
|---|------|-----------|------|----------|
| 50 | **SALVE: Sparse Autoencoder-Latent Vector Editing for Mechanistic Control** | arXiv | 2025 | SAE + Grad-FAM + 权重空间干预，发现→验证→控制统一框架 |
| 51 | **Control Reinforcement Learning: Interpretable Token-Level Steering via SAE Features** | arXiv | 2025 | 将SAE特征转向转化为可解释干预分析 |
| 52 | **Use Sparse Autoencoders to Discover Unknown Concepts, Not to Act on Known Concepts** | arXiv | 2025 | SAE适用于发现未知概念，而非已知概念操控 |
| 53 | **Sparse Autoencoder Features for Classifications and Transferability** | EMNLP | 2025 | SAE特征的分类与迁移能力 |
| 54 | **Transcoders Beat Sparse Autoencoders for Interpretability** | arXiv | 2025 | Transcoder作为SAE替代的可解释性比较 |

---

## 5. 评论/反馈驱动的推荐系统 (Review-Based Recommendation)

### 5.1 经典深度学习方法

| # | 论文 | 会议/期刊 | 年份 | 核心贡献 |
|---|------|-----------|------|----------|
| 55 | **DeepCoNN: Joint Deep Modeling of Users and Items Using Reviews** (Zheng et al.) | WSDM | 2017 | 首个用CNN从评论建模用户/物品的推荐模型 |
| 56 | **NARRE: Neural Attentional Rating Regression with Review-level Explanations** (Chen et al.) | WWW | 2018 | 注意力机制挖掘评论有用性，提供评论级解释 |
| 57 | **ANR: Aspect-based Neural Recommender** (Chin et al.) | CIKM | 2018 | 基于方面的神经推荐 |
| 58 | **ABAE: An Unsupervised Neural Attention Model for Aspect Extraction** (He et al.) | ACL | 2017 | 无监督神经注意力方面提取 |
| 59 | **NRPA: Neural Recommendation with Personalized Attention** (Liu et al.) | SIGIR | 2019 | 个性化注意力神经推荐 |

### 5.2 LLM增强的评论理解

| # | 论文 | 会议/期刊 | 年份 | 核心贡献 |
|---|------|-----------|------|----------|
| 60 | **Understanding Before Recommendation: Semantic Aspect-Aware Review Exploitation via LLMs** (SAER/SAGCN) | TOIS | 2024 | ★ 链式提示策略从评论提取语义方面，SAGCN建模方面级用户行为 |
| 61 | **Sentiment-Aware Recommendation Systems in E-Commerce: A Review from NLP Perspective** | arXiv | 2025 | 电商情感推荐综述 |
| 62 | **BERT-Based Multi-Embedding Fusion Method Using Review Text for Recommendation** | Expert Systems | 2025 | BERT多嵌入融合评论推荐 |

### 5.3 评论驱动推荐综述

| # | 论文 | 会议/期刊 | 年份 | 核心贡献 |
|---|------|-----------|------|----------|
| 63 | **Review-based Recommender Systems: A Survey of Approaches, Challenges and Future Perspectives** | ACM Computing Surveys | 2024 | 评论推荐全面综述（2015-2024） |
| 64 | **Review-Aware Recommender Systems (RARSs): Recent Advances, Experimental Comparative Analysis, and New Directions** | ACM Computing Surveys | 2025 | 评论感知推荐最新进展与实验对比 |

### 5.4 方面级情感与动态偏好

| # | 论文 | 会议/期刊 | 年份 | 核心贡献 |
|---|------|-----------|------|----------|
| 65 | **FedAspect-GNN: Integrating Aspect-Level Sentiment Analysis and GNN for Federated Recommendation** | Expert Systems with Applications | 2025 | 方面级情感 + GNN联邦推荐 |
| 66 | **Capturing Dynamic User Preferences: A Recommendation System Model with Non-Linear Forgetting and Evolving Topics** | Systems | 2025 | 非线性遗忘 + 评论语义演化建模动态偏好 |
| 67 | **Sentimentally Enhanced Conversation Recommender System** | Complex & Intelligent Systems | 2024 | 情感增强对话推荐 |

---

## 6. LLM驱动的推荐系统

### 6.1 LLM增强推荐

| # | 论文 | 会议/期刊 | 年份 | 核心贡献 |
|---|------|-----------|------|----------|
| 68 | **Large Language Model Enhanced Recommender Systems: A Survey** | KDD | 2025 | LLM增强推荐综述：知识/交互/模型增强三类 |
| 69 | **A Survey on Large Language Models for Recommendation** (Wu et al.) | WWW Journal | 2024 | DLLM4Rec vs GLLM4Rec分类 |
| 70 | **LLM4Rec: A Comprehensive Survey** | Future Internet | 2025 | 150+论文综合综述 |
| 71 | **A Survey on LLM-powered Agents for Recommender Systems** | EMNLP Findings | 2025 | LLM智能体推荐综述 |
| 72 | **Towards Next-Generation LLM-based Recommender Systems: A Survey and Beyond** | arXiv | 2024 | 下一代LLM推荐综述 |
| 73 | **Do LLMs Benefit from User and Item Embeddings in Recommendation Tasks?** | arXiv | 2025 | 用户/物品embedding对LLM推荐的影响 |

### 6.2 强化学习与偏好对齐

| # | 论文 | 会议/期刊 | 年份 | 核心贡献 |
|---|------|-----------|------|----------|
| 74 | **DeepSeekMath / GRPO: Group Relative Policy Optimization** (Shao et al.) | arXiv | 2024 | GRPO算法，组内相对优势估计，无需价值网络 |
| 75 | **Rank-GRPO: Training LLM-based Conversational Recommender Systems with RL** (Netflix) | ICLR | 2026 | ★ rank-level优势分配 + 几何均值重要性比，推荐专用GRPO |
| 76 | **MiniRec: Data-Efficient RL for LLM-based Recommendation** | arXiv | 2025 | 数据高效的LLM推荐RL训练 |
| 77 | **Direct Preference Optimization (DPO)** (Rafailov et al.) | NeurIPS | 2023 | 无需RL的偏好优化，LLM对齐基础 |
| 78 | **A Survey of Direct Preference Optimization** | arXiv | 2025 | DPO综述 |
| 79 | **Group Robust Preference Optimization in Reward-free RLHF** | NeurIPS | 2024 | 组鲁棒偏好优化 |
| 80 | **Training-Free Group Relative Policy Optimization** | arXiv | 2025 | 无训练GRPO，上下文空间优化 |
| 81 | **SimPO: Simple Preference Optimization** | arXiv | 2024 | 简化偏好优化 |

---

## 7. 序列推荐基础模型

| # | 论文 | 会议/期刊 | 年份 | 核心贡献 |
|---|------|-----------|------|----------|
| 82 | **SASRec: Self-Attentive Sequential Recommendation** (Kang & McAuley) | ICDM | 2018 | 自注意力序列推荐，左到右单向 |
| 83 | **BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations** (Sun et al.) | CIKM | 2019 | 双向Transformer + Cloze任务 |
| 84 | **GRU4Rec: Session-based Recommendations with RNN** (Hidasi et al.) | ICLR Workshop | 2016 | 首个将RNN应用于序列推荐 |
| 85 | **RecFormer: Text Is All You Need for Sequential Recommendation** (Li et al.) | KDD | 2023 | 纯文本表示的序列推荐，无需预训练LM或item ID |
| 86 | **TransRec: Modeling Sequential Prediction as Translation** (He et al.) | RecSys | 2017 | 用户作为翻译向量的序列预测 |
| 87 | **Embedding in Recommender Systems: A Survey** | arXiv | 2023 | 推荐系统嵌入综述 |

---

## 8. 用户感知表示与对比学习

### 8.1 解耦表示与对比学习

| # | 论文 | 会议/期刊 | 年份 | 核心贡献 |
|---|------|-----------|------|----------|
| 88 | **IDCL: Intent-aware Recommendation via Disentangled Graph Contrastive Learning** | IJCAI | 2023 | 意图感知解耦图对比学习 |
| 89 | **GDCCDR: Graph Disentangled Contrastive Learning with Personalized Transfer for Cross-Domain Recommendation** | AAAI | 2024 | 跨域解耦对比学习 + 个性化迁移 |
| 90 | **MDCCF: Multi-level Disentangled Contrastive Collaborative Filtering** | arXiv | 2025 | 多级解耦对比协同过滤 |
| 91 | **DCLKR: Disentangled Contrastive Learning for Knowledge-aware Recommender System** | ISWC | 2023 | 知识图谱视角的解耦对比学习 |

### 8.2 多模态物品表示

| # | 论文 | 会议/期刊 | 年份 | 核心贡献 |
|---|------|-----------|------|----------|
| 92 | **MUFASA: Multimodal Fusion and Sparse Attention-based Alignment Model** | arXiv | 2025 | 多模态融合 + 稀疏注意力对齐 |
| 93 | **MMFN: Multifactorial Modality Fusion Network for Multimodal Recommendation** | Applied Intelligence | 2024 | 多因子模态融合网络 |
| 94 | **Gotta Embed Them All: Knowledge-Aware Heterogeneous Multimodal Item Embeddings** | JIIS | 2025 | 知识感知异构多模态物品嵌入 |
| 95 | **Personalized Item Embeddings in Federated Multimodal Recommendation** | arXiv | 2024 | 联邦多模态个性化物品嵌入 |

---

## 9. 知识图谱与动态兴趣建模

| # | 论文 | 会议/期刊 | 年份 | 核心贡献 |
|---|------|-----------|------|----------|
| 96 | **DRSKG: Dynamic Preference Recommendation Based on Spatiotemporal Knowledge Graphs** | Complex & Intelligent Systems | 2024 | 时空知识图谱动态偏好建模 |
| 97 | **Learning Fine-Grained User Preference for Personalized Recommendation** | TST | 2024 | 细粒度用户偏好学习 |
| 98 | **Enhanced Knowledge Graph Recommendation with Multi-level Contrastive Learning** | Scientific Reports | 2024 | 知识图谱 + 多级对比学习 |

---

## 10. 生成式信息检索基础

| # | 论文 | 会议/期刊 | 年份 | 核心贡献 |
|---|------|-----------|------|----------|
| 99 | **DSI: Transformer Memory as a Differentiable Search Index** (Tay et al.) | NeurIPS | 2022 | 可微搜索索引，自回归生成文档ID |
| 100 | **NCI: A Neural Corpus Indexer for Document Retrieval** (Wang et al.) | NeurIPS | 2022 | 神经语料库索引，语义结构化docID + 查询生成增强 |
| 101 | **SE-DSI: Semantic-Enhanced Differentiable Search Index** (Guo et al.) | KDD | 2023 | 语义增强的可微搜索索引 |
| 102 | **How Does Generative Retrieval Scale to Millions of Passages?** | EMNLP | 2023 | 生成式检索的规模化挑战 |

---

## 11. 冷启动与新物品推荐

| # | 论文 | 会议/期刊 | 年份 | 核心贡献 |
|---|------|-----------|------|----------|
| 103 | **Hybrid Attribute-based Recommender System with Emphasis on Cold Start Problem** | Frontiers in Computer Science | 2024 | 混合属性推荐，纯冷启动 |
| 104 | **Improving Cold-Start Recommendations Using Item-Based Stereotypes** | UMUAI | 2021 | 基于物品刻板印象的冷启动改进 |

---

## 12. 连续token与扩散模型推荐

| # | 论文 | 会议/期刊 | 年份 | 核心贡献 |
|---|------|-----------|------|----------|
| 105 | **Diffusion Generative Recommendation with Continuous Tokens** | arXiv | 2025 | 扩散模型 + 连续token的生成式推荐 |
| 106 | **Masked Diffusion for Generative Recommendation** (Shah et al.) | arXiv | 2025 | 掩码扩散生成推荐 |

---

## 主题交叉分析

### 与本研究最相关的论文 (Top 10)

1. **Pctx** [35] — 同一物品在不同用户上下文下生成不同SID（条件化token化）
2. **MMQ** [36] — 行为感知微调动态适配SID
3. **SAE4Rec** [48] — SAE在序列推荐中的可解释性与控制
4. **SAER/SAGCN** [60] — LLM提取评论语义方面，增强推荐
5. **ASEGR** [39] — 评论中提取感官属性嵌入序列推荐
6. **LETTER** [4] — 可学习tokenizer融合协同+语义+多样性
7. **HiD-VAE** [28] — 层次解耦SID，解决碰撞问题
8. **CoST** [27] — 对比量化融合语义+关系信息
9. **OneRec** [14] — 端到端生成式推荐 + 偏好对齐
10. **Rank-GRPO** [75] — 推荐场景专用GRPO训练

### 研究空白分析

| 空白领域 | 说明 | 相关论文 |
|---------|------|---------|
| **评论驱动的动态SID** | 现有Pctx/MMQ使用行为信号适配SID，但未利用评论文本的丰富语义信息 | [35,36,60,39] |
| **SAE特征引导的SID生成** | SAE能发现可解释特征，但未被用于指导SID的个性化生成 | [48,49,41-46] |
| **用户-物品交互的方面级SID** | 方面级情感分析已用于推荐，但未与SID生成结合 | [57,58,60,65] |
| **评论感知的量化空间适配** | VQ/RQ-VAE的codebook是静态的，未根据用户反馈动态调整 | [25,26,27,28] |
| **可解释的动态SID** | 动态SID的可解释性（为何为该用户生成此SID）尚未研究 | [35,48,50] |

---

## 关键技术路线建议

基于文献分析，建议的技术路线：

```
用户评论 → LLM方面提取 [60] → 方面级情感嵌入
    ↓
物品内容嵌入 (text/visual) → SAE分解 [48,49] → 可解释特征维度
    ↓
条件化量化 [35,36]:
  - 方面级权重调整 SAE 特征
  - 用户上下文依赖的 codebook 选择
  - 动态 SID 生成
    ↓
生成式推荐训练 (SFT + GRPO [75])
    ↓
评估: HR@K, NDCG@K + 可解释性指标
```

### 与现有方法的对比定位

| 方法 | 信息来源 | SID类型 | 个性化 | 可解释 |
|------|---------|---------|--------|--------|
| TIGER | 文本内容 | 静态 | ✗ | ✗ |
| LETTER | 文本+协同 | 静态 | ✗ | ✗ |
| Pctx | 用户行为序列 | 动态 | ✓ | ✗ |
| MMQ | 多模态+行为 | 动态 | ✓ | ✗ |
| **Ours (提议)** | **文本+评论+SAE** | **动态** | **✓** | **✓** |

---

## 参考文献列表（按主题排序）

### 生成式推荐
1. Geng et al. "Recommendation as Language Processing (RLP): A Unified Pretrain, Personalized Prompt & Predict Paradigm (P5)." RecSys, 2022.
2. Rajput et al. "Recommender Systems with Generative Retrieval." NeurIPS, 2023.
3. Zheng et al. "Adapting Large Language Models by Integrating Collaborative Semantics for Recommendation (LC-Rec)." ICDE, 2024.
4. Wang et al. "LETTER: A LEarnable Tokenizer for generaTivE Recommendation." arXiv:2405.07314, 2024.
5. Wang et al. "Content-Based Collaborative Generation for Recommender Systems (ColaRec)." arXiv:2403.18480, 2024.
6. Fan et al. "TokenRec: Learning to Tokenize ID for LLM-based Generative Recommendations." TKDE, 2024.
7. "BIGRec: Grounding Generated Identifiers for Generative Recommendation." arXiv, 2024.
8. Li et al. "PAP-REC: Personalized Automatic Prompt for Recommendation Language Model." arXiv:2402.00284, 2024.
9. "SEATER: Tree-based Numeric IDs for Generative Recommendation." arXiv, 2023.
10. "IDGenRec: Zero-shot Generative Recommendation." arXiv, 2024.
11. Ye et al. "EAGER: Two-Stream Generative Recommender with Behavior-Semantic Collaboration." KDD, 2024.
12. Xiao et al. "UNGER: Generative Recommendation with a Unified Code." TOIS, 2025.
13. "GRAM: Generative Recommendation via Semantic-aware Augmented Multi-modal." ACL, 2025.
14. Deng et al. "OneRec: Unifying Retrieve and Rank with Generative Recommender." arXiv:2502.18965, 2025.
15. Kong et al. "MiniOneRec: An Open-Source Framework for Scaling Generative Recommendation." arXiv:2510.24431, 2025.
16. "SynerGen: Contextualized Generative Recommender for Unified Search and Recommendation." arXiv:2509.21777, 2025.
17. Ju et al. "GRID: Generative Recommendation with Semantic IDs." CIKM, 2025.
18. Ju et al. "Generative Recommendation with Semantic IDs: A Practitioner's Handbook." CIKM, 2025.

### 综述
19. Li et al. "A Survey of Generative Search and Recommendation in the Era of LLMs." arXiv:2404.16924, 2024.
20. "Large Language Models for Generative Recommendation: A Survey and Visionary Discussions." LREC-COLING, 2024.
21. "Generative Recommendation: A Survey of Models, Systems, and Applications." arXiv, 2025.
22. "From Feature-Based, Generative to Agentic Paradigms." arXiv:2504.16420, 2025.
23. "From Matching to Generation: A Survey on Generative Information Retrieval." TOIS, 2025.

### 语义ID与向量量化
24. Van den Oord et al. "Neural Discrete Representation Learning (VQ-VAE)." NeurIPS, 2017.
25. Zeghidour et al. "SoundStream: An End-to-End Neural Audio Codec." TASLP, 2022. (RQ-VAE)
26. Hou et al. "Learning Vector-Quantized Item Representation for Transferable Sequential Recommenders (VQ-Rec)." WWW, 2023.
27. Fan et al. "CoST: Contrastive Quantization based Semantic Tokenization." RecSys, 2024.
28. "HiD-VAE: Interpretable Generative Recommendation via Hierarchical and Disentangled Semantic IDs." arXiv:2508.04618, 2025.
29. "SIDE: Semantic ID Embedding for Effective Learning from Sequences." AdKDD, 2025.
30. "SimCIT: A Simple Contrastive Framework of Item Tokenization." arXiv:2506.16683, 2025.
31. "ACERec: Unleash the Potential of Long Semantic IDs." arXiv:2602.13573, 2025.
32. "RPG: Generating Long Semantic IDs in Parallel." arXiv:2506.05781, 2025.
33. "Rethinking Generative Recommender Tokenizer." arXiv:2602.02338, 2025.
34. Liu et al. "Vector Quantization for Recommender Systems: A Review and Outlook." arXiv:2405.03110, 2024.

### 动态/个性化SID
35. "Pctx: Tokenizing Personalized Context for Generative Recommendation." arXiv:2510.21276, 2025.
36. "MMQ: Multimodal Mixture-of-Quantization Tokenization." arXiv:2508.15281, 2025.
37. "AgentDR: Dynamic Recommendation with Implicit Item-Item Relations via LLM-based Agents." arXiv:2510.05598, 2025.
38. "EmerFlow: LLM-Empowered Representation Learning for Emerging Item Recommendation." arXiv:2512.10370, 2025.
39. "Sensory-Aware Sequential Recommendation via Review-Distilled Representations (ASEGR)." arXiv:2603.02709, 2025.
40. "DeepInterestGR: Mining Deep Multi-Interest Using Multi-Modal LLMs." arXiv:2602.18907, 2025.

### 稀疏自编码器
41. Bricken et al. "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning." Anthropic, 2023.
42. Cunningham et al. "Sparse Autoencoders Find Highly Interpretable Features in Language Models." ICLR, 2024.
43. Rajamanoharan & Nanda. "Improving Sparse Decomposition of Language Model (Gated SAE)." NeurIPS, 2024.
44. Rajamanoharan & Nanda. "Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU SAE." NeurIPS, 2024.
45. Bussmann et al. "BatchTopK Sparse Autoencoders." arXiv:2412.06410, 2024.
46. Gao et al. "Scaling and Evaluating Sparse Autoencoders (TopK SAE)." arXiv, 2024.
47. "SAEBench: A Comprehensive Benchmark for SAEs in LM Interpretability." arXiv:2503.09532, 2025.
48. Klenitskiy et al. "SAE4Rec: Sparse Autoencoders for Sequential Recommendation Models." RecSys, 2025.
49. "Understanding Internal Representations of Recommendation Models with SAEs." arXiv:2411.06112, 2024.
50. "SALVE: Sparse Autoencoder-Latent Vector Editing for Mechanistic Control." arXiv:2512.15938, 2025.
51. "Control Reinforcement Learning: Interpretable Token-Level Steering via SAE Features." arXiv:2602.10437, 2025.
52. "Use Sparse Autoencoders to Discover Unknown Concepts, Not to Act on Known Concepts." arXiv:2506.23845, 2025.
53. "Sparse Autoencoder Features for Classifications and Transferability." EMNLP, 2025.
54. "Transcoders Beat Sparse Autoencoders for Interpretability." arXiv:2501.18823, 2025.

### 评论驱动推荐
55. Zheng et al. "DeepCoNN: Joint Deep Modeling of Users and Items Using Reviews." WSDM, 2017.
56. Chen et al. "NARRE: Neural Attentional Rating Regression with Review-level Explanations." WWW, 2018.
57. Chin et al. "ANR: Aspect-based Neural Recommender." CIKM, 2018.
58. He et al. "An Unsupervised Neural Attention Model for Aspect Extraction (ABAE)." ACL, 2017.
59. Liu et al. "NRPA: Neural Recommendation with Personalized Attention." SIGIR, 2019.
60. Chen et al. "Understanding Before Recommendation: Semantic Aspect-Aware Review Exploitation via LLMs (SAGCN)." TOIS, 2024.
61. "Sentiment-Aware Recommendation Systems in E-Commerce: A Review." arXiv:2505.03828, 2025.
62. "BERT-Based Multi-Embedding Fusion Method Using Review Text." Expert Systems, 2025.
63. "Review-based Recommender Systems: A Survey." ACM Computing Surveys, 2024.
64. "Review-Aware Recommender Systems (RARSs): Recent Advances." ACM Computing Surveys, 2025.
65. "FedAspect-GNN: Integrating Aspect-Level Sentiment and GNN for Federated Recommendation." Expert Systems with Applications, 2025.
66. "Capturing Dynamic User Preferences: Non-Linear Forgetting and Evolving Topics." Systems, 2025.
67. "Sentimentally Enhanced Conversation Recommender System." Complex & Intelligent Systems, 2024.

### LLM推荐
68. "Large Language Model Enhanced Recommender Systems: A Survey." KDD, 2025.
69. Wu et al. "A Survey on Large Language Models for Recommendation." WWW Journal, 2024.
70. "LLM4Rec: A Comprehensive Survey." Future Internet, 2025.
71. "A Survey on LLM-powered Agents for Recommender Systems." EMNLP Findings, 2025.
72. "Towards Next-Generation LLM-based Recommender Systems: A Survey and Beyond." arXiv, 2024.
73. "Do LLMs Benefit from User and Item Embeddings in Recommendation Tasks?" arXiv:2601.04690, 2025.

### 强化学习与偏好对齐
74. Shao et al. "DeepSeekMath / Group Relative Policy Optimization (GRPO)." arXiv:2402.03300, 2024.
75. Zhu et al. "Rank-GRPO: Training LLM-based Conversational Recommender Systems with RL." ICLR, 2026.
76. "MiniRec: Data-Efficient RL for LLM-based Recommendation." arXiv:2602.04278, 2025.
77. Rafailov et al. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." NeurIPS, 2023.
78. "A Survey of Direct Preference Optimization." arXiv:2503.11701, 2025.
79. "Group Robust Preference Optimization in Reward-free RLHF." NeurIPS, 2024.
80. "Training-Free Group Relative Policy Optimization." arXiv:2510.08191, 2025.
81. "SimPO: Simple Preference Optimization." arXiv, 2024.

### 序列推荐
82. Kang & McAuley. "Self-Attentive Sequential Recommendation (SASRec)." ICDM, 2018.
83. Sun et al. "BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations." CIKM, 2019.
84. Hidasi et al. "Session-based Recommendations with RNN (GRU4Rec)." ICLR Workshop, 2016.
85. Li et al. "Text Is All You Need for Sequential Recommendation (RecFormer)." KDD, 2023.
86. He et al. "TransRec: Translation-based Sequential Recommendation." RecSys, 2017.
87. "Embedding in Recommender Systems: A Survey." arXiv:2310.18608, 2023.

### 用户感知表示与对比学习
88. "Intent-aware Recommendation via Disentangled Graph Contrastive Learning (IDCL)." IJCAI, 2023.
89. "Graph Disentangled Contrastive Learning with Personalized Transfer (GDCCDR)." AAAI, 2024.
90. "Multi-level Disentangled Contrastive Collaborative Filtering (MDCCF)." arXiv, 2025.
91. "Disentangled Contrastive Learning for Knowledge-aware Recommender System (DCLKR)." ISWC, 2023.
92. "MUFASA: Multimodal Fusion and Sparse Attention-based Alignment Model." arXiv, 2025.
93. "MMFN: Multifactorial Modality Fusion Network for Multimodal Recommendation." Applied Intelligence, 2024.
94. "Gotta Embed Them All: Knowledge-Aware Heterogeneous Multimodal Item Embeddings." JIIS, 2025.
95. "Personalized Item Embeddings in Federated Multimodal Recommendation." arXiv:2410.08478, 2024.

### 知识图谱与动态兴趣
96. "DRSKG: Dynamic Preference Recommendation Based on Spatiotemporal Knowledge Graphs." Complex & Intelligent Systems, 2024.
97. "Learning Fine-Grained User Preference for Personalized Recommendation." TST, 2024.
98. "Enhanced Knowledge Graph Recommendation with Multi-level Contrastive Learning." Scientific Reports, 2024.

### 生成式检索基础
99. Tay et al. "Transformer Memory as a Differentiable Search Index (DSI)." NeurIPS, 2022.
100. Wang et al. "A Neural Corpus Indexer for Document Retrieval (NCI)." NeurIPS, 2022.
101. Guo et al. "Semantic-Enhanced Differentiable Search Index (SE-DSI)." KDD, 2023.
102. "How Does Generative Retrieval Scale to Millions of Passages?" EMNLP, 2023.

### 冷启动
103. "Hybrid Attribute-based Recommender System for Cold Start." Frontiers in Computer Science, 2024.
104. "Improving Cold-Start Recommendations Using Item-Based Stereotypes." UMUAI, 2021.

### 连续token与扩散
105. "Diffusion Generative Recommendation with Continuous Tokens." arXiv:2504.12007, 2025.
106. "Masked Diffusion for Generative Recommendation." arXiv:2511.23021, 2025.

---

*文献综述生成日期：2026-03-11*
*共收录 106 篇文献，覆盖 12 个主题领域*
