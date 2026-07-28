
# ⚽️Explainable Video Assistant Referee System⚽️ 


| Paper | Demo Clips | Dataset | Model | Training | Offline Demo | Examples |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| [![paper](https://img.shields.io/badge/Paper-<COLOR>.svg)](https://arxiv.org/abs/2404.06332) | [![DemoClip-1](https://img.shields.io/badge/-DemoClip1-blue)](https://www.youtube.com/watch?v=IJAkf94lPO8) [![DemoClip-2](https://img.shields.io/badge/-DemoClip2-yellow)](https://www.youtube.com/watch?v=SEdUjrLMBik) [![DemoClip-3](https://img.shields.io/badge/-DemoClip3-green)](https://www.youtube.com/watch?v=Nkk125MvkJ0) [![DemoClip-4](https://img.shields.io/badge/-DemoClip4-violet)](https://www.youtube.com/watch?v=yG6qV9jsAJI) | [SoccerNet-XFoul](#SoccerNet-XFoul) | [X-VARS](#X-VARS) | [Training](#Training) |  [Offline Demo](https://github.com/heldJan/X-VARS/blob/main/X-VARS/demo.md) |  [Examples](#Examples) |


This repository contains: 
- **SoccerNet-XFoul**, a novel dataset consisting of more than **22k video-question-answer** triplets annotated by over 70 experienced football referees. 🚀
- **X-VARS**, a new vision language model that can **perform multiple multi-modal tasks** such as visual captioning, question-answering, video action recognition, and can generate explanations of its decisions on-par with human level. 🤖
- The code to run an **offline demo on your laptop**. 💻

## 📢 NEWS 📢

The **SoccerNet-XFoul dataset** is now [available](https://drive.google.com/drive/folders/1UbMAQVFrTB-DtEFUSmv8tBXuurrBfMUJ?usp=sharing)! 🔥🔥

## Installation

We recommend setting up a conda environment for the project:

```
conda create --name=xvars python=3.10
conda activate xvars

git clone https://github.com/heldJan/X-VARS.git
cd X-VARS
pip install -r requirements.txt
```

## SoccerNet-XFoul

The SoccerNet-XFoul dataset consists of 22k video-question-answer pairs annotated by more than 70 experienced referees. 
Due to the subjectivity in refereeing, we gathered multiple answers for the same action, rather than collecting a single decision and explanation for each question. In the end, for each action, we have, on average, $1.5$ answers for the same question.

Follow the [link](https://pypi.org/project/SoccerNet/) to easily download the SoccerNet pip package.

If you want to download the video clips, you will need to fill a [NDA](https://docs.google.com/forms/d/e/1FAIpQLSfYFqjZNm4IgwGnyJXDPk2Ko_lZcbVtYX73w5lf6din5nxfmA/viewform) to get the password.

Then use the API to downlaod the data:

```
from SoccerNet.Downloader import SoccerNetDownloader as SNdl
mySNdl = SNdl(LocalDirectory="path/to/SoccerNet")
mySNdl.downloadDataTask(task="mvfouls", split=["train","valid","test","challenge"], password="enter password")
```
To obtain the data in 720p, add version = "720p" to the input arguments. If you face issues extracting data from the train_720p.zip folder, the error may come from using the default unzip extractor. Using the app "The Unarchiver" should enable you to unzip it successfully.

The annotations can be downloaded from [here](https://drive.google.com/drive/folders/1UbMAQVFrTB-DtEFUSmv8tBXuurrBfMUJ?usp=sharing)
🔥

![My Image](Images/dataset_example.png)


## X-VARS

X-VARS is a visual language model based on a **fine-tuned CLIP visual encoder** to extract **spatio-temporal video features** and to obtain **multi-task predictions** regarding the type and severity of fouls. The **linear layer** connects the vision encoder to the language model by projection the video features in the text embedding dimension. We input the **projected spatio-temporal features** alongside **the text predictions** obtained by the two classification heads (for the task of determining the type of foul and the task of determining if it is a foul and the corresponding severity) into the Vicuna-v1.1 model, initialized with weights from LLaVA.
![My Image](Images/architecture.png)


## Training 

We propose a two-stage training approach. **The first stage** fine-tunes CLIP on a multi-task classification to learn prior knowledge about football and refereeing. **The second stage** consists in fine-tuning the projection layer and several layers of the LLM to enhance the model's generation abilities in the sport-specific domain. 

To replicate the training, check out [Training](https://github.com/heldJan/X-VARS/blob/main/X-VARS/training.md)

More information are provided in our [paper](https://arxiv.org/abs/2404.06332).


## Examples
![My Image](Images/holding_no_card_main.png)
![My Image](Images/tackling_yellow_main.png)
![My Image](Images/dive.png)
![My Image](Images/tackling.png)

## 🎨 UI美化功能 (2025-01-13)

### 足球主题界面美化

我们为Streamlit界面添加了完整的足球主题美化，包括：

#### 🎯 主要美化内容：

1. **足球场主题背景**
   - 绿色草地纹理背景
   - 足球场线条装饰
   - 符合足球主题的配色方案

2. **视频播放器美化**
   - 足球场风格的视频容器
   - 白色边框和阴影效果
   - 中心圆圈装饰元素
   - 响应式布局优化

3. **对话框界面美化**
   - 用户消息：蓝色渐变气泡，右侧对齐
   - 助手消息：绿色渐变气泡，左侧对齐，带黄色边框
   - 消息进入动画效果
   - 点击消息可复制内容

4. **按钮和交互元素**
   - 黄色渐变主要按钮
   - 绿色渐变次要按钮
   - 悬停动画和点击波纹效果
   - 统一的圆角设计

5. **人性化提示**
   - 右下角使用提示卡片
   - 视频播放状态通知
   - 加载动画和进度提示
   - 友好的错误提示样式

6. **侧边栏美化**
   - 深绿色渐变背景
   - 黄色边框装饰
   - 统一的按钮样式
   - 清晰的信息层次

#### 🚀 使用方法：

美化功能已自动集成，无需额外配置。只需运行：

```bash
streamlit run x_vars_demo_with_streamlit.py
```

美化文件位置：
- CSS样式：`static/football_theme.css`
- JavaScript交互：`static/football_theme.js`

#### 🎨 自定义美化：

如需自定义样式，可以修改 `static/football_theme.css` 文件中的CSS变量：

```css
:root {
    --football-green: #2d5016;      /* 深绿色 */
    --football-light-green: #4a7c2a; /* 浅绿色 */
    --football-yellow: #ffd700;      /* 金黄色 */
    --football-white: #ffffff;       /* 白色 */
}
```

## 📊 新增功能 (2025-01-13)

### Jupyter Notebook 视频分析工具

我们新增了一个完整的Jupyter Notebook代码文件 `video_analysis_notebook.py`，提供了详细的视频分析功能：

#### 🎯 主要功能：

1. **单视频深度分析**
   - 视频基本信息提取（帧率、分辨率、时长等）
   - 智能帧采样和预览显示
   - 使用X-VARS模型进行特征提取和预测

2. **多维度可视化**
   - 预测结果可视化（动作类型、犯规判定、严重程度）
   - 特征分布分析（直方图、箱线图、热图）
   - 时间vs空间特征对比分析
   - 综合性能评估雷达图

3. **深度特征分析**
   - 特征重要性分析（基于方差）
   - 时空特征融合效果评估
   - 特征质量统计和评分

4. **结果保存和导出**
   - 特征数据保存（.pkl格式）
   - 预测结果导出（.json格式）
   - 详细分析报告（.json和.md格式）

5. **系统性能监控**
   - CPU和内存使用情况
   - GPU资源监控
   - 模型推理性能评估

#### 🚀 使用方法：

```python
# 1. 复制video_analysis_notebook.py的内容到Jupyter Notebook
# 2. 修改配置参数
CONFIG = {
    'video_path': 'path/to/your/video.mp4',
    'model_weights': 'path/to/model/weights.pth',
    # ... 其他配置
}
# 3. 逐个执行代码单元格
```

#### 📈 扩展功能：

- **批量视频分析**: 支持处理多个视频文件
- **结果对比分析**: 比较不同视频的预测结果
- **自定义分析**: 基于提取特征进行自定义机器学习分析

## 🔧 优化记录 (2025-01-11)

### Streamlit阻塞问题修复

我们对 `x_vars_demo_with_streamlit.py` 进行了全面优化，解决了可能导致应用阻塞的关键问题：

#### 📊 主要改进：

1. **模型加载优化**
   - 添加了更好的错误处理和用户反馈
   - 改进了加载状态显示，让用户了解当前进度
   - 增加了自动错误恢复机制

2. **视频处理流程优化**
   - **实时进度条**: 显示详细的CLIP处理进度 (X/Y - Z%)
   - **内存优化**: 中间结果移动到CPU以节省GPU内存
   - **批次间延迟**: 每2批处理添加10ms延迟，防止UI完全阻塞
   - **用户控制**: 添加了"停止处理"按钮，允许用户中止长时间运行的任务

3. **生成过程改进**
   - **超时控制**: 默认60秒生成超时，防止无限等待
   - **状态反馈**: 实时显示生成状态和进度
   - **错误处理**: 更友好的错误消息和自动重试机制

4. **内存管理**
   - **自动清理**: 在开始新处理前清理GPU内存
   - **资源监控**: 可选的系统资源使用情况显示
   - **更好的垃圾回收**: 及时释放不需要的张量

5. **用户体验**
   - **更好的按钮布局**: 分离的"开始分析"和"停止处理"按钮
   - **实时状态更新**: 更详细的处理状态信息
   - **系统信息显示**: 可选的CPU、内存和GPU使用情况监控

#### 🚀 性能提升：

- **减少阻塞**: 通过添加适当的延迟和进度更新，UI响应更流畅
- **更好的错误恢复**: 失败时自动清理资源，无需重启应用
- **内存效率**: 优化的内存使用模式，减少OOM风险
- **用户控制**: 用户可以随时停止长时间运行的任务

#### 🛠️ 使用建议：

- 如果遇到显存不足，请降低"采样帧数"或"CLIP 批大小"
- 可以将"CLIP 设备"切换到CPU以减少GPU内存占用
- 使用"显示系统信息"功能监控资源使用情况
- 长时间处理时可使用"停止处理"按钮安全中止

#### 🔍 故障排除：

如果仍然遇到阻塞问题：
1. 检查系统资源使用情况
2. 降低批处理大小和采样帧数
3. 尝试使用CPU进行CLIP处理
4. 确保有足够的系统内存和存储空间

## Acknowledgements

 - [VARS:](https://github.com/SoccerNet/sn-mvfoul) The first multi-task classification model for predicting if it is a foul or not and the corresponding severity.
 - [Video-ChatGPT:](https://github.com/mbzuai-oryx/Video-ChatGPT) A vision and language model used as a foundation model for X-VARS


If you're using X-VARS in your research or application, please cite our paper: 
```bibtex
    @article{Held2024XVARS-arxiv,
        title = {X-{VARS}: Introducing Explainability in Football Refereeing with Multi-Modal Large Language Model},
        author = {Held, Jan and Itani, Hani and Cioppa, Anthony and Giancola, Silvio and Ghanem, Bernard and Van Droogenbroeck, Marc},
        journal = arxiv,
        volume = {abs/2404.06332},
        year = {2024},
        publisher = {arXiv},
        eprint = {2404.06332},
        keywords = {},
        eprinttype = {arXiv},
        doi = {10.48550/arXiv.2404.06332},
        url = {https://doi.org/10.48550/arXiv.2404.06332}
}

}
```
## Authors 

- [Jan Held](https://www.linkedin.com/in/heldjan/)
- [Hani Itani](https://www.linkedin.com/in/hani-itani-613b01111/)
- [Anthony Cioppa](https://www.linkedin.com/in/anthony-cioppa-65a779128/)
- [Silvio Gincola](https://www.linkedin.com/in/silvio-giancola/)
- [Bernard Ghanem](https://www.linkedin.com/in/bernardghanem/)
- [Marc Van Droogenbroeck](https://scholar.google.com/citations?user=IQm8ljgAAAAJ&hl=en)

[<img src="Images/uliege.png" width="150" height="70">](https://www.ival-mbzuai.com)
[<img src="Images/kaust.png" width="150" height="80">](https://github.com/mbzuai-oryx)

---

## 🧭 本地快速上手（Windows/CPU 简洁版）

> 目标：在不配置GPU的前提下，快速跑通「上传视频 → 提取特征 → 简洁对话」闭环。

### 1) 安装与环境

```bash
conda create -n xvars python=3.10 -y
conda activate xvars
pip install -r requirements.txt
```

- 如需RAG增强（足球规则检索），安装：
```bash
pip install faiss-cpu sentence-transformers
```

### 2) 数据准备

- 演示视频：仓库已附带 `data/videos/clip_1.mp4` 可直接使用。
- 完整数据集（可选）：将 SoccerNet-XFoul 解压到 `dataset/` 目录，结构应为：
```
dataset/
  ├─ Train/ action_x/clip_*.mp4
  ├─ Valid/ action_x/clip_*.mp4
  └─ Test/  action_x/clip_*.mp4
```

### 3) 启动 Streamlit 简洁对话 Demo（CPU）

```bash
streamlit run x_vars_demo_with_streamlit.py
```

在左侧：
- 勾选“使用示例视频”或上传自定义视频
- 点击「🚀 处理视频」完成一次性特征提取
- 右侧输入问题开始对话（默认短答、直给结论）

可调参数（侧边栏）：
- Max new tokens：输出长度（推荐 64-128 简洁模式）
- Temperature：采样温度（0.1-0.3 更稳定）
- Video token len：视频特征序列长度（64-300，默认200）
- 采样帧数/批大小：CPU下建议保持较小以更稳

### 4) 启用足球规则RAG增强（可选）

前提：`vector_store/2023-2024中文版足球規則.txt` 已存在（仓库已提供）。

步骤：
1. 安装依赖：`pip install faiss-cpu sentence-transformers`
2. 在侧边栏点击「初始化知识库」
3. 勾选「启用足球规则增强」

若初始化失败，点击「🔥 强制重建向量存储」重试（会自动在 `vector_store/embeddings.pkl` 里缓存向量）。

### 5) 批量特征提取与预测（离线生成 .pkl + 预测JSON）

使用视觉编码器批量处理 Train/Valid/Test：

```bash
python visual_encoder/create_features.py \
  --path_dataset dataset \
  --path_weights visual_encoder/14_model.pth.tar
```

输出：
- 每个 `action_x/` 下生成 `PRE_CLIP_feature_clip_*.pkl`
- 项目根目录生成 `predictionsTrain_clip.json / predictionsValid_clip.json / predictionsTest_clip.json`

### 6) 常见问题（Windows/CPU）

- decord/视频读帧问题：确保 `decord` 已安装；如仍异常，可尝试 `pip install opencv-python` 备用读取方案（本仓库默认使用 decord）。
- 模型/权重找不到：确认本地目录存在：
  - 语言模型与分词器：`X-VARS/X-VARS_weights/`、`X-VARS/tokenizer/`
  - CLIP视觉权重：`visual_encoder/clip-vit-large-patch14/`、`visual_encoder/14_model.pth.tar`
- 序列长度报错：侧边栏下调 `Video token len`（如 128）或减少采样帧数。
- 速度较慢：本指南默认 CPU，追求速度请在有条件时切换到 GPU（需自行配置CUDA版PyTorch）。

---

## 🔑 关键脚本速览与参数（简明）

- `x_vars_demo_with_streamlit.py`：本地对话式演示（默认CPU）
  - 侧边栏参数：`Max new tokens`、`Temperature`、`Video token len`、`采样帧数`、`CLIP 批大小`、`启用RAG`

- `visual_encoder/create_features.py`：批量提取特征与预测
  - `--path_dataset`：`dataset/` 根目录
  - `--path_weights`：视觉分类权重路径（默认 `visual_encoder/14_model.pth.tar`）

- `video_chatgpt/inference.py`：底层时空特征聚合与生成调用（供开发者参考）

如需更细的研发说明，请参考现有文档：`README_x_vars_demo.md`、`training.md`、`RAG_使用说明.md`。
