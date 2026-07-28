# X-VARS + RAG 推理优化实验方案

---

## 1. 技术关系澄清

CoT、Knowledge RAG、Self-consistency 是**正交技术**，可自由组合：

| 技术                       | 作用层                 | 解决的问题                       |
| -------------------------- | ---------------------- | -------------------------------- |
| **Knowledge RAG**    | 输入端（增强上下文）   | 模型没看到正确的规则 → 信息缺失 |
| **CoT**              | 输出端（改变生成方式） | 模型逻辑跳跃、推理链不完整       |
| **Self-consistency** | 解码端（多次采样聚合） | 模型判断不稳定 → 随机波动       |
| **Prompt 工程**      | 输入端（规范化指令）   | 输出格式不稳定、评测解析失败     |

它们之间没有依赖关系，但**组合时需注意**：

- CoT 的输出更长、更复杂，会增加 Self-consistency 的投票解析难度
- RAG 注入过多上下文会挤压视频 token 预算，需控制总长
- 组合实验应在单变量实验之后，基于错误归因结果选择

---

## 2. 实验线设计

### 2.1 总览

| Line         | 名称      | 改动内容                                      | 依赖       |
| ------------ | --------- | --------------------------------------------- | ---------- |
| **L0** | Baseline  | 原版 X-VARS，无 RAG                           | 无         |
| **L1** | RAG-Base  | 添加 RAG（Top1，短截断 ~200 字符）            | L0         |
| **L2** | RAG-Deep  | RAG 深化：Top-K=3，取消/放宽截断，可选 rerank | L1         |
| **L3** | Prompt    | 固定输出格式 + 角色约束                       | L0         |
| **L4** | CoT       | 链式思考提示，调大 `max_new_tokens`         | L0         |
| **L5** | Self-Con  | N 次采样 + 投票聚合（`do_sample=True`）     | L0         |
| **L6** | Best-Comb | L2 + L3 + L4 最优组合（基于 L1-L5 结果）      | L2, L3, L4 |

### 2.2 各 Line 详细配置

#### L0: Baseline（原版 X-VARS）

- 不启用 RAG 检索
- `enhanced_question = question`（不拼接规则文本）
- 其余参数与原版一致：`max_new_tokens=64`、`do_sample=False`
- **目的**：建立基线，作为所有后续实验的对照

#### L1: RAG-Base（基础 RAG）

- 启用 `FootballRAGSystem.search(question, top_k=3)`，但仅使用 Top1
- 规则文本截断 ~200 字符，以「参考规则：…」形式拼入 `enhanced_question`
- **目的**：验证基础 RAG 对准确率的影响（即当前系统现状）

#### L2: RAG-Deep（RAG 深化）⭐ 预期收益最高

在 L1 基础上改进检索管线：

- **Top-K 扩展**：使用 Top-3 结果，按相关性拼接（控制总长度）
- **取消截断**：移除 200 字符限制，使用完整规则片段
- **可选 rerank**：用 cross-encoder 对检索结果重排序，过滤噪声
- **查询改写（可选）**：HyDE 或同义词扩展，缓解用户问法与规则术语的词汇鸿沟

**注意事项**：

- 监控拼接后 `enhanced_question` 总长度，避免挤压视频 token
- 检索延迟会增加，但对 300 条评测可接受

#### L3: Prompt Engineering（提示词工程）

在 L0 基础上优化提示词模板：

- **角色约束**：明确系统角色（如「你是足球裁判专家」）
- **输出格式固定**：要求先输出判罚标签，再输出一句话依据
- **指令精简**：去除冗余指令，控制长度

**示例模板**：

```
你是一名专业的足球视频裁判分析专家。
请根据视频内容和以下规则，判断该场景的判罚结果。

要求：
1. 首先输出判罚标签（如：犯规/不犯规/黄牌/红牌）
2. 然后用一句话说明判罚依据

参考规则：{rag_context}
```

**目的**：低成本快速验证提示词规范化对准确率和可解析性的提升。

#### L4: CoT（Chain-of-Thought）

在 L0 基础上引入链式思考：

- 提示词中加入「请按以下步骤分析」：
  1. 描述视频中发生的动作
  2. 分析是否构成犯规
  3. 根据规则判断判罚类别
  4. 给出最终结论
- **`max_new_tokens` 需调大**（如 128-256），否则推理链会被截断
- **评测解析**：需从 CoT 输出中抽取最终标签（正则/关键词匹配）

**风险**：

- 小模型可能「虚构推理链」，推理过程不忠实
- 延迟和 token 消耗增加

#### L5: Self-consistency（自洽投票）

- 对同一输入做 N 次采样（`do_sample=True`，`temperature=0.7`，N=3~5）
- 对离散判罚标签做**多数投票**
- 需要统一的标签抽取规则（与 L4 的解析规则保持一致）

**成本**：

- 延迟和算力约为 **N 倍**
- 需记录随机种子以保证可复现性

**适用场景**：判罚类别为离散标签时效果好；长开放式文本聚合困难，收益不确定。

#### L6: Best-Comb（最优组合）

基于 L1-L5 的实验结果，选取效果最好的 2-3 个方向叠加：

**可能的组合**：

- **组合 A**（推荐优先尝试）：L2 + L3 + L4 = RAG 深化 + Prompt 规范 + CoT
- **组合 B**（离散标签友好）：L2 + L3 + L5 = RAG 深化 + Prompt 规范 + Self-consistency
- **组合 C**（全面型）：L2 + L3 + L4 + L5，但成本高，视预算决定

---

## 3. 评测规范

### 3.1 数据集

- **测试集**：`dataset/Test/annotations.json`（SoccerNet-MVFoul / XFoul 风格）。
- **与 X-VARS 论文对齐的推荐协议**（`rag_experiment/`，默认开启）：
  - 任务标签为 **4 类 Offence+Severity**：`No offence` / `Offence + No card` / `Offence + Yellow card` / `Offence + Red card`（Severity `1.0/3.0/5.0`）。
  - **每个 action 取 1 个视角**（优先 Main camera center），不要默认用全部多视角 clip（否则 N 会被放大到 ~700）。
  - 排除 `Between` 与 Severity `2.0/4.0/空`（歧义，不在论文四分类头内）。当前 Test 约 **248** 条进入主指标。
  - **帧采样**：犯规前后各 8 帧（默认中心 3.0s，与 `create_features.py` 一致；标注 Timestamp 为比赛绝对时间，不能直接当片内偏移）。
  - **CLIP 分类头文本**：默认加载 `visual_encoder/14_model.pth.tar`，把 `The prediction for this video is ...` 拼进 LLM 提示（与 `custom_dataset.py` 一致）。
- **避免 few-shot 泄漏**：所有 few-shot 示例不得出现在测试集中。
- 细节见 `rag_experiment/PAPER_ALIGNMENT.md`。

### 3.2 评测指标

| 指标                         | 说明                                            |
| ---------------------------- | ----------------------------------------------- |
| **准确率（Accuracy）** | 判罚结果与标准答案一致的比例            |
| **均衡准确率（Balanced Acc.）** | **论文 Table 4 同款主辅指标**；类别极不均衡时优先报告 |
| **F1 Score**           | 各类别 F1（macro/weighted），关注类别不均衡情况 |
| **检索命中率（可选）** | 检索片段是否包含正确规则条款                    |
| **输出可解析率**       | 输出能被评测脚本正确解析为四类标签的比例   |

### 3.3 消融记录

每次实验需记录以下参数配置：

```
- use_rag: true/false
- top_k: N
- 截断长度: N / 无限制
- 是否 rerank: true/false
- max_new_tokens: N
- do_sample: true/false
- temperature: N（若采样）
- N_samples: N（若 self-consistency）
- prompt_template: 版本号或内容摘要
```

---

## 4. 实验执行流程

### 阶段一：基线建立（1-2 天）

1. 跑 **L0**（Baseline），记录 300 条准确率、F1、混淆矩阵
2. 跑 **L1**（RAG-Base），与 L0 对比，确认 RAG 的基础收益

### 阶段二：单变量实验（3-5 天）

3. 跑 **L2**（RAG-Deep），与 L1 对比，验证检索深化的收益
4. 跑 **L3**（Prompt），与 L0 对比，验证提示词的影响
5. 跑 **L4**（CoT），与 L0 对比，验证链式推理的收益
6. 跑 **L5**（Self-Con），与 L0 对比，验证投票聚合的收益

### 阶段三：错误归因与组合（2-3 天）

7. 抽取 **50-100 条错例**，按错误类型打标：

| 错误类型              | 对应优化方向                 |
| --------------------- | ---------------------------- |
| 检索失败 / 召回为空   | → RAG 深化（L2）            |
| 检索到错误条款        | → Rerank（L2）              |
| 上下文过短 / 模型瞎编 | → 取消截断（L2）            |
| 输出格式不稳定        | → Prompt 工程（L3）         |
| 推理跳跃              | → CoT（L4）                 |
| 随机波动大            | → Self-consistency（L5）    |
| 视觉误判              | → 视觉侧改进（future work） |

8. 根据错误归因结果，选择 L2-L5 中**最高频的 2-3 个方向**叠加为 **L6**
9. 跑 L6，记录最终结果

---


## 5. 工程实现与输出目录（仓库）

> **仅测 RAG 效果时**：优先使用专用包 `rag_experiment/`（入口 `run_rag_experiment.py`，实验线 **R0–R4**），详见 `rag_experiment/README.md`。下文 L0–L6 为混合优化实验。

- **混合实验入口**：`test.py`（调用 `experiment/runner.py`）。
- **RAG 专用入口**：`run_rag_experiment.py`（调用 `rag_experiment/runner.py`）。
- **按功能分模块**（便于在不同 Line 上组合能力）：
  - `experiment/knowledge_rag.py`：检索与片段合并（L1/L2 类 RAG）。
  - `experiment/prompt_engineering.py`：基础任务描述与专家角色/格式（Prompt 工程）。
  - `experiment/cot.py`：链式思考指令块（CoT）。
  - `experiment/prompt_assembly.py`：按 Line 组合上述文本与 RAG 块。
  - `experiment/self_consistency.py`：多次采样 + 多数投票。
  - `experiment/generation.py`：单次贪心/可控采样解码。
  - `experiment/line_presets.py`：L0–L6 与各模块的对应关系。
  - `rag_experiment/`：RAG 消融（查询模式、检索诊断、英文报告与图）。
- **单次运行根目录**（`--output-dir`，例如 `experiment_runs/run1`）结构示例：

```
experiment_runs/run1/
  L0/
    experiment_settings.json   # 本线全部超参、提示词模板原文、CoT 块、RAG 配置等
    checkpoint.json            # 断点续跑（按 action_id/clip_idx）
    logs/
      predictions.jsonl        # 每条样本一条 JSON（含 raw 输出）
    plots/
      confusion_matrix.png
      metrics_line.json        # 本线 accuracy / F1 等
  L1/
    ...
  summary/
    metrics.json               # 各线汇总指标
    plots/
      metrics_by_line.png      # 跨线对比柱状图（图中文字为英文）
```

**说明**：L2 方案中的 **rerank / HyDE** 仍为可选增强，当前代码未接 cross-encoder；需要时可只在 `knowledge_rag.py` 扩展，不影响其它 Line。

---

## 6. 注意事项

1. **可复现性**：固定随机种子，记录所有超参数，确保实验可复现
2. **token 预算管理**：视频 token + RAG 上下文 + CoT 输出的总长度不能超过模型上下文窗口
3. **评测解析统一**：L3、L4、L5 的输出格式不同，需统一标签抽取规则，保证评测一致性
4. **渐进式实验**：先跑低成本实验（L3），再跑高成本实验（L5），避免资源浪费
5. **版本管理**：每次实验记录代码版本、模型权重版本、数据集版本
