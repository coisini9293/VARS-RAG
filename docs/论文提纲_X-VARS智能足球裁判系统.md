## 一、摘要（Abstract）

1. **研究背景与问题**

   - 足球判罚主观性强、争议多；VAR 提供多视角回放，但面向公众/教学的**规则依据型文字解释**仍不足。
   - 现有足球视频分析多侧重动作识别与事件检测；多模态大模型虽能开放问答，但**缺少权威规则文本注入**时易出现「常识化」回答。
   - X-VARS（Held et al., CVPRW 2024）已展示「视频 + 多模态 LLM → 可解释裁判问答」；本文在其工程路线上，系统接入**足球规则 RAG**，并建立**可复现的定量消融评测**。

2. **研究目标与思路**

   - 目标：构建「**视觉理解 + 规则检索增强 + 对话式解释**」的本地可运行原型，并回答：**在相同视觉–语言主干下，RAG 是否提升 Offence+Severity 判罚识别？**
   - 技术路径：
     - **视觉**：CLIP ViT-L/14 提帧特征 → 定长视频 token（沿用 Video-ChatGPT / X-VARS 管线）；
     - **可选分类头**：`14_model.pth.tar` 预测犯规类型/严重程度，以文本形式注入提示（对齐论文与 `custom_dataset`）；
     - **RAG**：规则库分块 + 向量检索 → 「参考规则」拼入用户问题；
     - **生成**：多模态 LLM 回答论文原问（是否犯规？给什么牌？为何？）；
     - **评测**：从自由文本中抽取四分类标签，计算 Accuracy / Balanced Accuracy。

3. **主要贡献**

   - **系统集成**：在 X-VARS 类多模态判罚管线中系统化接入足球规则 RAG（索引、检索、prompt 拼接、Streamlit 演示）。
   - **评测协议对齐**：按 X-VARS 论文设定，在 SoccerNet-XFoul / MVFoul 风格 Test 集上做 **Offence+Severity 四分类**消融（R0–R4），指标以 **Accuracy + Balanced Accuracy** 为主。
   - **RAG 消融设计**：固定解码与问题模板，仅改变检索深度/查询方式（Baseline / Top1 截断 / TopK / Deep / Adaptive），并记录检索诊断（空检索率、Top1 分数、上下文长度）。
   - **边界说明**：不主张全新视觉骨干；增量聚焦规则侧 RAG 与系统/评测层设计。

4. **实验表述注意**

   - 摘要中的数值**仅写入已跑通并落盘的结果**（`rag_experiment_runs/*/summary/`）；若全量 Test 未跑完，写「完成协议与框架 + 部分/计划中的结果」，勿夸大。
   - 生成侧解释质量（BLEU/ROUGE、人工评分）可作为**补充或未来工作**，主表以离散标签识别为准。

---

## 二、介绍（Introduction）

1. **研究背景：为什么要做这个项目？**

   - 足球对抗激烈，单次回放难以形成共识；公众与教学更需要**可读的判罚理由**。
   - VAR / 传统 VARS 侧重检测与多视角决策支持；开放式「为什么」仍依赖专家经验。
   - 纯多模态生成缺少规则锚定；RAG 可将《足球竞赛规则》条款注入生成上下文。

2. **文献综述（建议三段）**

   - **足球视频分析与 VARS**：犯规类型/严重程度识别、SoccerNet-MVFoul 等；强调标签预测 vs. 自然语言解释的差异。
   - **多模态裁判解释（X-VARS）**：Video-ChatGPT 路线；训练/评测问题「Is it a foul or not? Why?」「What card would you give? Why?」；用 ChatGPT 从解释中抽取分类标签。
   - **RAG 与体育多模态**：规则/法律类文本 RAG 的可信度收益；足球规则 + 短视频判罚问答的工程缺口。

3. **研究问题与创新点**

   - **研究问题**：
     1. 如何在尽量不改视觉–语言主干的前提下，将规则库稳定并入多模态判罚问答？
     2. 不同 RAG 强度（截断、Top-K、自适应查询）对四分类识别有何影响？
     3. 如何使评测协议与 X-VARS 论文尽量一致，从而结果可对照？
   - **创新点（宜写法）**：
     - RAG 与多模态判罚流水线的闭环（非纯文本 chatbot）；
     - 专用 `rag_experiment` 消融（R0–R4）与检索诊断；
     - 论文对齐：犯规居中 8+8 帧、分类头文本、原问题措辞、LLM 标签抽取、Accuracy + Balanced Accuracy。

4. **论文结构安排**

   - 第 3 章：相关技术（CLIP、Video-ChatGPT 式 MLLM、RAG、标签抽取）。
   - 第 4 章：系统总体设计（演示管线 + RAG 模块 + 与原 X-VARS 的接口）。
   - 第 5 章：实验设计与结果（评测协议、R0–R4、指标与分析）。
   - 第 6 章：总结与展望。

---

## 三、相关技术（Related Techniques）

> 写清：技术是什么 → 在本系统中的角色 → 与本文增量的关系。

1. **CLIP 与视觉编码**

   - ViT-L/14 视觉塔；帧采样后得到定长视频 token（`Video token len`）。
   - **犯规居中采样**：默认在片段约 3.0s 处取犯规前 8 帧 + 后 8 帧（对齐 `create_features` 约定；标注 Timestamp 为比赛绝对时间，不能直接当地内帧下标）。

2. **视频多模态 LLM（Video-ChatGPT / X-VARS）**

   - 投影层 + 视频占位符注入因果解码器；自由文本生成裁判解释。
   - 本文不重推架构；接口点：用户侧可拼接**分类头预测句**与**参考规则**。

3. **CLIP 多任务分类头（可选注入）**

   - 权重 `visual_encoder/14_model.pth.tar`；输出犯规相关类别与严重程度倾向。
   - 注入格式类似：`The prediction for this video is ...`（置于问题之前，贴近论文 `<P_foul><P_sev>`）。

4. **RAG**

   - 规则文本分块 → 嵌入（如 sentence-transformers）→ Top-K 检索 → 拼入 prompt。
   - 与纯文本 RAG 的差异：检索可基于固定问句或**自适应元数据查询**（动作类型/身体部位等），视频特征仍独立编码。

5. **标签抽取（评测后处理）**

   - 论文：ChatGPT-3.5 从解释中抽取分类预测。
   - 本仓库：`--label-parse llm`（如 DashScope Qwen）或规则/混合解析；目标标签为四类之一。

---

## 四、技术流程（Methodology & System Design）

1. **整体框架**

```text
视频 → 帧采样(犯规居中/均匀) → CLIP 特征 → 视频 token
                ↓
用户问题（论文双问） ← 可选：分类头文本 / RAG 参考规则
                ↓
build_prompt（视频占位 + 对话模板）→ 多模态 LLM generate
                ↓
自由文本回答 →（评测）LLM/规则抽取四分类标签
```

   - **继承**：视频解码、CLIP、特征压缩、模型加载与生成、Streamlit 交互。
   - **本文扩展**：FootballRAGSystem、规则索引缓存、演示侧「开/关 RAG」、专用消融包 `rag_experiment/`。

2. **演示系统（Streamlit）**

   - 入口：`x_vars_demo_with_streamlit.py`。
   - 能力：上传/示例视频、特征缓存、多轮问答、RAG 检索结果展示、设备与长度等参数。
   - 定位：可复现原型与定性展示；**定量主实验不依赖 UI**。

3. **RAG 模块要点**

   - 知识源：如 `vector_store/` 下竞赛规则文本；`embeddings.pkl` 缓存。
   - 检索：向量相似度（设备对齐，避免 CUDA/CPU 不一致导致静默回退关键词匹配）。
   - 拼装顺序（论文对齐默认）：**分类头文本 → 论文问题 → 参考规则 → 视频 token**。
   - 配置：Top-K、截断长度、重建索引、CPU/GPU。

4. **与原项目 / 论文的边界**

   - 训练脚本、多视角池化 SOTA 复现：**非本文主贡献**（可声明未做或多视角为扩展）。
   - 本文主增量：**规则 RAG + 论文协议下的 RAG 消融评测**。

---

## 五、实验（Experiments）

### 5.1 实验目标

- **主问题**：相同模型与解码设置下，RAG 各配置相对 Baseline（R0）是否提升四分类 **Accuracy / Balanced Accuracy**。
- **辅问题**：检索是否健康（空检索率、Top1 分数）；生成文本是否可被稳定解析为四类标签（parse rate）。
- **非主目标（可附录/展望）**：解释文风的 BLEU/ROUGE、大规模人工偏好。

### 5.2 数据与任务（论文对齐）

| 项目 | 设定 |
|------|------|
| 数据 | SoccerNet-XFoul / MVFoul 风格；默认 `dataset/Test` + `annotations.json` |
| 单位 | `--unit action`：每 action 优先 Main camera center（约 248 条有效 action，具体以过滤后为准） |
| 标签 | `No offence` / `Offence + No card` / `Offence + Yellow card` / `Offence + Red card` |
| Severity 映射 | 标注 `1.0→No card`，`3.0→Yellow`，`5.0→Red` |
| 排除 | `Offence==Between`；Severity `2.0`/`4.0`/空 |
| 问题 | 「Is it a foul or not? Why?」+「What card would you give? Why?」 |
| 帧 | 16 帧 @224；默认 foul-centered（中心约 3.0s） |
| 指标 | Accuracy、**Balanced Accuracy**（主排序）、Macro-F1、parse rate |

### 5.3 实验线设计（RAG 专用：R0–R4）

> 入口：`python run_rag_experiment.py`；说明见 `rag_experiment/README.md`、`PAPER_ALIGNMENT.md`。

| 线 | 名称 | 变量（其余固定） |
|----|------|------------------|
| **R0** | Baseline | 无 RAG |
| **R1** | RAG-Base | Top3 检索取 Top1，约 200 字截断（接近演示默认） |
| **R2** | RAG-TopK | 合并 Top3，短截断 |
| **R3** | RAG-Deep | Top3 更长上下文 |
| **R4** | RAG-Adaptive | 同 R3 合并；检索查询加入 Action class / Contact / Bodypart（**禁止**泄漏 Offence 真值） |

公平性：各线共用同一任务问句与解码超参；共享视频特征缓存；记录 `predictions.jsonl` 与检索命中。

### 5.4 标签解析与实现细节

- 默认 `--label-parse llm`（环境变量配置 API）；对比实验可报 rule / hybrid。
- 注意：若模型回答从不提及黄牌/红牌，解析器无法「发明」牌类 → 需从**生成内容**与 **prompt** 两侧分析失败模式。
- 可选消融：`--no-use-clip-classifier`、`--frame-sample uniform`。

### 5.5 结果呈现（写作模板）

1. **主表**：R0–R4 的 Acc / Bal Acc / Macro-F1 / parse rate。
2. **图**：跨线指标柱状图、各类混淆矩阵（英文坐标，便于直接贴图）。
3. **检索诊断表**：空检索率、平均 Top1 分数、平均上下文长度。
4. **错误分析**：检索噪声、规则过长挤占视频信息、只答犯规不问牌、类别极不均衡等。
5. **诚实声明**：若某次 run 中断，标明已完成线数/样本数，或以完整 run 为准更新表格。

### 5.6 可选扩展实验（非必须）

- 混合线 **L0–L6**（`experiment/`：Prompt / CoT / Self-consistency）：说明与 RAG 正交，**主文以 R0–R4 为主**，L 线可放附录。
- 人工小样本：规则引用是否合理、可读性 Likert。

---

## 六、总结与展望（Conclusion & Future Work）

1. **工作总结**

   - 完成 X-VARS 类管线 + 足球规则 RAG + 本地演示；
   - 建立与论文对齐的四分类评测与 R0–R4 消融框架；
   - 贡献落点：集成、协议化评测与可复现实验，而非新基础模型。

2. **局限性**

   - 单视角评测；未复现论文人工研究 / 全文生成指标；
   - 全量推理成本高（生成慢），完整表格依赖算力与稳定跑通；
   - 规则库版本、嵌入模型语种、检索噪声仍影响上限。

3. **未来工作**

   - 跑完全量 Test 并固化主表；混合检索 / 重排 / 结构化规则；
   - 多视角聚合；解释质量自动指标 + 小规模人工评测；
   - 更强 VLM 或领域微调与 Agent 化规则查询。

---

## 七、参考文献与写作提示（对应关系）

为方便写作，下表说明核心文献适合支撑的章节；**数值与表格必须以原文或本人实验落盘结果为准**。

1. **X-VARS（Held et al., CVPRW 2024）**  
   - 背景、任务定义（犯规/牌/为何）、评测（四分类、Balanced Accuracy、从解释中抽取标签）、方法高层流程。  
   - **自己写**：RAG 接入、R0–R4、本地实现与本人实验数字。

2. **VARS / SoccerNet-MVFoul 系列**  
   - 多视角自动判罚与数据集背景；对比「检测标签」vs「解释生成」。  
   - **自己写**：本文单视角 + 解释 + RAG 的定位。

3. **Towards Universal Soccer Video Understanding 等**  
   - 相关工作中的通用足球理解路线；对比任务类型即可。

4. **体育 AI / 多模态综述、SPORTU、Sports-QA 等**  
   - 应用图谱、可解释性动机、未来基准扩展；未实测则不写数值。

---

## 八、仓库材料与章节映射（写作自检）

| 论文章节 | 主要对应材料 |
|----------|----------------|
| 系统与演示 | `x_vars_demo_with_streamlit.py`，`rag_system.py`，`docs/RAG_使用说明.md` |
| RAG 设计 | `rag_experiment/knowledge_rag.py`，`prompts.py`，`presets.py` |
| 评测协议 | `rag_experiment/PAPER_ALIGNMENT.md`，`labels.py`，`frame_sample.py`，`clip_classifier.py` |
| 实验运行 | `run_rag_experiment.py`，`rag_experiment/README.md` |
| 结果落盘 | `rag_experiment_runs/<run>/summary/REPORT.md`，`plots/`，各线 `metrics` / `predictions.jsonl` |
| 混合扩展（附录） | `docs/实验方案_RAG与推理优化.md`，`experiment/` |

**撰写提示**：摘要贡献句、第 5 章每一句数字与结论，均需能在仓库日志/报告中找到对应文件；协议已对齐论文处写「对齐」，未做处置于局限或展望，避免「提纲超前、实现滞后」。
