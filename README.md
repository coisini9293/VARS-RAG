## X-VARS 智能足球裁判系统（本地 Streamlit 版）

面向足球短视频的**可解释辅助判罚演示**：先从视频中提取**视觉特征**，再交由**多模态大语言模型**结合文本问题生成回答；可选使用 **RAG** 从本地规则文本中检索片段，把检索结果拼进提示词以增强依据。

**唯一推荐入口（本地）：** 根目录下的 `x_vars_demo_with_streamlit.py`。

---

## 目录

1. [如何运行](#如何运行)
2. [环境与依赖](#环境与依赖)
3. [本地须准备的文件](#本地须准备的文件)
4. [权重文件分别做什么](#权重文件分别做什么)
5. [使用流程（界面）](#使用流程界面)
6. [项目整体框架](#项目整体框架)
7. [端到端流程：从问题到回答](#端到端流程从问题到回答)
8. [模型与数据流（输入 / 输出 / 对应代码）](#模型与数据流输入--输出--对应代码)
9. [目录结构速查](#目录结构速查)
10. [延伸阅读](#延伸阅读)
11. [批量实验与评测](#批量实验与评测)
12. [常见问题](#常见问题)

---

## 如何运行

推荐使用 Conda 环境 **`asd`**（Python **3.10.x**）。若尚未创建：

```bash
conda create -n asd python=3.10 -y
conda activate asd
pip install -r requirements.txt
```

之后在**项目根目录**启动界面：

```bash
conda activate asd
streamlit run x_vars_demo_with_streamlit.py
```

浏览器打开终端提示的地址（一般为 `http://localhost:8501`）。

> 若你使用其他环境名，将下文命令中的 `asd` 换成你的环境名即可。

---

## 环境与依赖

| 项目        | 说明                                                                                                                                                                                  |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Python      | 建议 **3.10.x**，与 `requirements.txt` 中 PyTorch / Transformers 版本匹配；文档示例使用 Conda 环境 **`asd`**                                                                                                      |
| 磁盘        | 本地完整权重通常需约 **15GB 级** 预留空间（随分片略有出入）                                                                                                                      |
| 内存 / 显存 | **CPU**：建议大内存（例如 32GB+ 更稳）。**GPU**：可选；需安装与驱动匹配的 CUDA 版 PyTorch；显存紧张时在界面中降低帧数、`Video token len`、CLIP 批大小或把 CLIP 切到 CPU |

主要依赖见 `requirements.txt`（`torch`、`transformers`、`decord`、`streamlit`、`sentence-transformers` 等）。程序入口处会设置 `TRANSFORMERS_NO_TF`、`TRANSFORMERS_NO_FLAX` 等，减少 TensorFlow/Flax 冲突。

---

## 本地须准备的文件

运行 `x_vars_demo_with_streamlit.py` 且走**本地离线权重**时，需存在（路径与代码中常量一致）：

```text
X-VARS/
  X-VARS_weights/          # VideoChatGPTLlamaForCausalLM 权重（config + safetensors 分片等）
  tokenizer/               # AutoTokenizer（tokenizer.model、tokenizer_config.json 等）
visual_encoder/
  clip-vit-large-patch14/  # HuggingFace 风格的 CLIP Vision 目录，供 CLIPVisionModel / CLIPImageProcessor 加载
vector_store/
  2023-2024中文版足球規則.txt   # 规则文本
  embeddings.pkl               # 构建后生成的向量缓存
```

测试或训练用视频可放在 `dataset/`（如 `Train` / `Valid` / `Test`）；界面中的示例视频路径在代码里为 `data/videos/clip_1.mp4`（不存在时可直接上传视频）。

---

## 权重文件分别做什么

下表对应**本地 Streamlit** 主流程里实际会用到的资源（路径以 `x_vars_demo_with_streamlit.py` 中常量为准）。

| 路径 / 文件                          | 作用                                                                                                                                                                                                                       |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `X-VARS/X-VARS_weights/`            | **主多模态大模型**：`VideoChatGPTLlamaForCausalLM`，负责在读入对齐后的 **视频特征序列** 与 **文本 token** 后做多层 Decoder 计算，并 **自回归生成** 回答文本。                                                             |
| `X-VARS/tokenizer/`                 | **分词器**：把「用户问题 + 对话模板 + 视频占位符」编成 `input_ids`；生成后再把新 token **解码成字符串**。                                                                                                                  |
| `visual_encoder/clip-vit-large-patch14/` | **CLIP 视觉主干（ViT）**：对每一帧 `(3,224,224)` 做前向，取 **倒数第二层 hidden states、去掉 CLS**，得到 patch 级特征，再在本项目中被压成固定长度 **`Video token len`** 的序列，作为 **视觉侧输入** 送给大模型（经 `mm_projector` 对齐到语言模型隐层维度）。 |

可简记：**大模型权重 = 生成回答；Tokenizer = 文本↔token；CLIP 目录 = 从画面抽向量。**

---

## 使用流程（界面）

1. 侧边栏填写运行参数，点击「应用参数」。  
2. 上传视频或勾选示例视频，点击「处理视频」，特征会写入 `st.session_state.video_features`。  
3. 在「智能对话」中输入问题或使用快捷问题按钮；若已初始化 RAG，可勾选规则增强。  
4. 「RAG 检索结果」展示最近一次检索的表格摘要；「系统状态」展示处理指标等。

侧边栏主要参数包括：`Temperature`、`Max new tokens`、是否采样、`Video token len`、采样帧数、CLIP 批大小、CLIP/RAG 设备、是否自动优化显存、RAG 初始化与重建等。

---

## 项目整体框架

下面按**层次**说明本仓库在「本地 Streamlit」这条线上的组成（不展开训练脚本细节）。

```text
┌──────────────────────────────────────────────────────────┐
│ x_vars_demo_with_streamlit.py                           │
│ Streamlit UI / session_state / RAG / 组装提示           │
└───────────────┬─────────────────────────────────────────┘
                │
    ┌───────────┴───────────┐
    ▼                       ▼
┌───────────────┐   ┌────────────────────────────────────┐
│ 视觉特征支路   │   │ 语言与多模态支路                   │
│ 读视频 + 采样  │   │ video_chatgpt/eval/model_utils.py │
│ CLIP 提特征   │   │   initialize_model →               │
│ 时空压缩      │   │   VideoChatGPTLlamaForCausalLM     │
└───────────────┘   │   + CLIPVisionModel + Tokenizer    │
                    │ video_chatgpt/model/video_chatgpt  │
                    │   forward / generate 消费视频特征  │
                    └────────────────────────────────────┘
```

- **界面与编排**：`x_vars_demo_with_streamlit.py`（`main()`、视频处理、`generate_answer`、RAG 检索与日志等）。  
- **模型加载（一次性）**：`load_all_models()` → `video_chatgpt.eval.model_utils.initialize_model()`。  
- **视频编码**：`load_torch_frames_without_numpy`、`extract_frames_to_tensor`、`process_clip_features`、`make_video_features_from_clip_hidden`；底层视频读取依赖 `decord`。  
- **多模态 LLM 定义与生成**：`video_chatgpt/model/video_chatgpt.py`（`VideoChatGPTLlamaForCausalLM` 等），生成时传入 `video_spatio_temporal_features`。  
- **对话格式**：`video_chatgpt/video_conversation.py`（如 `video-chatgpt_v1` 模板）、`video_chatgpt/constants.py`（视频占位 token 等）。  
- **独立 RAG 模块（可选扩展）**：`rag_system.py`、`rag_chat.py`、`knowledge_manager.py`，与主界面内嵌的 `FootballRAGSystem` 不同。

---

## 端到端流程：从问题到回答

下面按**时间顺序**描述一次完整交互中「你的视频 / 问题」经过哪些模块。

### 阶段 A：加载权重与组件

1. `load_all_models()` 调用 `initialize_model()`。  
2. 加载 `AutoTokenizer`（`X-VARS/tokenizer`）。  
3. 加载 `VideoChatGPTLlamaForCausalLM`（`X-VARS/X-VARS_weights`）。  
4. 加载 `CLIPVisionModel` 与 `CLIPImageProcessor`（`visual_encoder/clip-vit-large-patch14`）。

### 阶段 B：视频 → 固定形状的视频特征张量

5. 用户点击「处理视频」→ `process_video()`。  
6. `decord` 读视频，均匀采样到 `max_frames` 帧，归一化后得到张量 `(N, 3, 224, 224)`。  
7. `process_clip_features()`：  
   - 利用 `vision_tower_hf` 分批前向，`output_hidden_states=True`；  
   - 取 `hidden_states[-2][:, 1:]`，拼成 `(T, S, C)`。  
8. `make_video_features_from_clip_hidden()`：  
   - 对时间/空间分别做平均并拼接，得到时空混合特征 `(T+S, C)`；  
   - 插值/填充到 `video_token_len` → `(video_token_len, C)`；  
   - `unsqueeze(0)` → `video_spatio_temporal_features`，形状 `(1, N, C)`。  
9. 写入 `st.session_state.video_features`，同一视频多轮对话复用。

### 阶段 C：你的问题 → 文本 prompt（可选 RAG）

10. 用户在输入框或快捷按钮输入问题 `question`。  
11. 若启用 RAG：`FootballRAGSystem.search(question)` 检索规则片段，拼成增强问题 `enhanced_question`。  
12. `build_prompt(enhanced_question, model, video_token_len)`：  
    - 用对话模板包装问题；  
    - 在文本中插入长度为 `video_token_len` 的视频占位符串。  
13. `tokenizer` 对完整模板编码 → `input_ids`，形状 `(1, L)`。

### 阶段 D：多模态 Decoder 内部

14. `model.generate(input_ids, video_spatio_temporal_features, ...)`：  
    - `input_ids` → 文本嵌入；  
    - `video_spatio_temporal_features` 经过 `mm_projector` 映射到与词嵌入同一维度；  
    - 在占位符位置用视觉向量替换对应嵌入；  
    - 多层因果自注意力 + 前馈堆叠，自回归生成新 token。  
15. `tokenizer.decode` / `batch_decode` 得到回答字符串，返回到界面。

---

## 模型与数据流（输入 / 输出 / 对应代码）

### 1）主多模态模型（视频问答）

| 项目               | 内容                                                                                                                                                                                    |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 是什么             | `VideoChatGPTLlamaForCausalLM`，Video-ChatGPT / LLaMA 系多模态结构，基于 `from_pretrained` 从 `X-VARS/X-VARS_weights` 加载                                                            |
| 定义与 forward     | `video_chatgpt/model/video_chatgpt.py`                                                                                                                                                 |
| 如何被加载         | `video_chatgpt/eval/model_utils.py` 中 `initialize_model(...)`，由 `x_vars_demo_with_streamlit.py` 的 `load_all_models()` 调用                                                          |
| 图像预处理与配置   | 同 `initialize_model` 内的 `CLIPImageProcessor.from_pretrained`，源目录优先 `visual_encoder/clip-vit-large-patch14`                                                                    |

典型输入输出：

| 名称                        | 类型 / 形状           | 说明                                                                                                                                                                      |
| --------------------------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `input_ids`                | `(1, L_prompt)`       | `tokenizer` 对完整对话模板编码得到                                                                                                                                       |
| `video_spatio_temporal_features` | `(1, N, C)` | 由 `process_clip_features()` 生成的定长视频特征序列，经 `mm_projector` 投影后与文本 hidden states 对齐。`N` 对应 `Video token len`                                       |
| 输出 `sequences` / 文本     | token 序列 / 字符串   | `model.generate` 结果，经 `tokenizer.batch_decode` 转为自然语言回答                                                                                                      |

---

### 2）CLIP 视觉塔（特征提取支路）

| 阶段      | 输入                 | 输出                          | 代码位置                                                         |
| --------- | -------------------- | ----------------------------- | ---------------------------------------------------------------- |
| 读视频    | 文件路径             | 帧张量 `(N, 3, 224, 224)`    | `load_torch_frames_without_numpy` / `extract_frames_to_tensor`   |
| CLIP 编码 | `(N, 3, 224, 224)`   | `(T, S, C)`                   | `process_clip_features` 中 `vision_tower_hf(..., output_hidden_states=True)` |
| 时空压缩  | `(T, S, C)`          | `(video_token_len, C)`        | `make_video_features_from_clip_hidden`                           |
| 送入 LLM  | `(video_token_len, C)` | `(1, video_token_len, C)`   | `unsqueeze(0)`，在 `process_clip_features` 末尾                  |

---

### 3）RAG（规则检索）

| 项目         | 内容                                                                                                                                                                                           |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 是什么       | `x_vars_demo_with_streamlit.py` 中的 `FootballRAGSystem`，使用 sentence-transformers 将规则段落和查询编码成向量，并保存到 `vector_store/embeddings.pkl`                                       |
| 输入         | 用户问题字符串                                                                                                                                                                                 |
| 输出         | Top-K 规则片段列表（含文本和相似度分数），在 `generate_answer` 中与问题拼接，形成增强问题                                                                                                      |

---

## 目录结构速查

| 路径                                       | 作用                                                  |
| ------------------------------------------ | ----------------------------------------------------- |
| `x_vars_demo_with_streamlit.py`           | 本地 Streamlit 主入口，整合 UI、模型加载、RAG、日志 |
| `video_chatgpt/`                          | 多模态 LLM、对话模板、`initialize_model` 等          |
| `visual_encoder/`                         | CLIP 视觉编码相关脚本与权重目录                     |
| `static/`                                 | 前端样式与脚本                                       |
| `vector_store/`                           | 规则文本与向量缓存                                   |
| `chat_history/`、`detailed_logs/`         | 会话与详细生成日志                                   |
| `dataset/`                                | 数据集视频                                           |
| `X-VARS/`                                 | 本地 LLM 权重与 tokenizer                            |
| `training.py`、`custom_dataset.py`        | 训练相关脚本                                         |
| `rag_system.py`、`rag_chat.py` 等         | 独立 RAG 模块                                        |
| `docs/`                                   | 项目文档、论文提纲等                                |
| `experiment/`                             | 批量实验：RAG / Prompt / CoT / Self-consistency 等模块与 `runner` |
| `test.py`                                 | 混合实验命令行入口（调用 `experiment.runner`，L0–L6） |
| `rag_experiment/`                         | **专用 RAG 消融实验**（后端、进度条、英文图与报告） |
| `run_rag_experiment.py`                   | RAG 消融实验入口（调用 `rag_experiment.runner`，R0–R4） |

---

## 延伸阅读

| 文档                       | 内容                         |
| -------------------------- | ---------------------------- |
| `docs/项目模型与技术总结.md` | 模型组件与维度、路径归纳     |
| `docs/详细日志使用说明.md`   | `detailed_logs` 字段含义   |
| `docs/优化说明.md`         | 加载与 RAG 等优化记录        |
| `docs/training.md`         | 训练与特征脚本入口           |
| `docs/RAG_使用说明.md`     | 独立 RAG 模块使用说明        |
| `docs/实验方案_RAG与推理优化.md` | 混合实验线 L0–L6、指标与输出目录约定 |
| `rag_experiment/README.md`     | **专用 RAG 消融（R0–R4）使用说明与设计** |

---

## RAG 消融实验（推荐：测 RAG 效果）

若目标是验证「在原版 X-VARS 上加 RAG 是否有效」，请用专用目录 **`rag_experiment/`**（无前端、终端进度条、英文图、自动分析报告）。

**入口**：

```bash
conda activate asd
python run_rag_experiment.py --output-dir rag_experiment_runs/run1 --lines R0,R1,R2,R3,R4
```

| 线 | 含义 |
|----|------|
| **R0** | Baseline（无 RAG） |
| **R1** | RAG-Base（接近当前界面：Top1 + 约 200 字截断） |
| **R2** | RAG-TopK（合并 Top3，短截断） |
| **R3** | RAG-Deep（Top3 更长上下文） |
| **R4** | RAG-Adaptive（检索查询加入 Action class 等，**不含** Offence 标签） |

**与 X-VARS 论文对齐（默认）**：评测为 **Offence+Severity 四分类**；指标含 **Accuracy + Balanced Accuracy**；每 action **单主视角**；帧采样为犯规前后各 8 帧（中心约 3s）；并向 LLM 注入 **CLIP 分类头文本**（`14_model.pth.tar`）。详见 [`rag_experiment/PAPER_ALIGNMENT.md`](rag_experiment/PAPER_ALIGNMENT.md)。

主要产出：`summary/REPORT.md`（英文分析）、`summary/plots/*.png`（英文坐标与标题）、各线 `confusion_matrix.png` 与 `logs/predictions.jsonl`（含检索命中）。  
详解与续跑说明见 [`rag_experiment/README.md`](rag_experiment/README.md)。

---

## 批量实验与评测（混合线 L0–L6）

在**不启动 Streamlit** 的前提下，对 `dataset/Test` 批量对比 **Baseline、RAG、Prompt、CoT、Self-consistency**（L0–L6）。若只关心 RAG，优先用上一节的 `rag_experiment/`。

**入口脚本**：根目录 `test.py`（内部调用 `experiment.runner`，亦可 `python -m experiment.runner`）。

**依赖**：与主项目相同（`requirements.txt` 中的 `rich`、`scikit-learn`、`matplotlib` 等）；需已按上文准备好 X-VARS、CLIP、规则向量库。

**示例**（在项目根目录执行）：

```bash
python test.py --output-dir experiment_runs/run1 --lines L0,L1,L2
```

常用参数：`--dataset-root`、`--annotations`（默认 `dataset/Test/annotations.json`）、`--max-action`、`--video-token-len`、`--lines`（逗号分隔）、`--seed`。同一 `--output-dir` 下按 **Line 分子目录** 保存配置与断点，支持续跑。

**输出结构概要**：

- `experiment_runs/<run>/<Lk>/experiment_settings.json`：该线的 RAG/解码超参与提示模板原文  
- `experiment_runs/<run>/<Lk>/checkpoint.json`：断点  
- `experiment_runs/<run>/<Lk>/logs/predictions.jsonl`：逐样本预测与原始输出  
- `experiment_runs/<run>/<Lk>/plots/`：该线混淆矩阵、`metrics_line.json`  
- `experiment_runs/<run>/summary/`：跨线汇总 `metrics.json` 与 `plots/metrics_by_line.png`  

实现按功能拆在 `experiment/` 包内，详见 `docs/实验方案_RAG与推理优化.md` 第 5 节。

---

## 常见问题

**本地报“找不到模型 / tokenizer / CLIP”**  
核对 `X-VARS/X-VARS_weights`、`X-VARS/tokenizer`、`visual_encoder/clip-vit-large-patch14` 是否完整；查看终端里 `initialize_model` 的日志路径。

**显存不足**  
降低采样帧数、`Video token len`、CLIP batch；必要时将 CLIP 切到 CPU。

**RAG 无法初始化 / 看起来有结果但其实是关键词匹配**  
确认 `vector_store/2023-2024中文版足球規則.txt` 与 `embeddings.pkl` 存在，且已安装 `sentence-transformers`。若 RAG 设备选 CUDA 而向量缓存在 CPU，旧版会静默回退到文本匹配；当前代码会在加载/检索时自动对齐设备。界面「系统状态」里应显示检索模式为「向量搜索」。

**CLIP 分类头 `14_model.pth.tar` 好像没加载**  
这是**可选**辅助分类头，主对话 / RAG 实验默认 `load_clip_classifier=False`，不加载也能正常跑 X-VARS+RAG；文件存在但不等于当前主流程在用。

**CUDA 相关报错（Windows）**  
安装与显卡驱动匹配的 PyTorch CUDA 版本；无 GPU 时安装 CPU 版 PyTorch 并在界面中选择 CPU。

