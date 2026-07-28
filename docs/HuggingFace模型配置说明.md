# HuggingFace模型配置说明

## 📦 概述

本系统已支持从HuggingFace Hub加载模型，无需在本地存储大型模型文件。

## 🔧 配置步骤

### 1. 设置HuggingFace用户名

有两种方式设置您的HuggingFace用户名：

**方式1：环境变量（推荐）**
```bash
# Windows PowerShell
$env:HF_USERNAME="your-username"

# Windows CMD
set HF_USERNAME=your-username

# Linux/Mac
export HF_USERNAME="your-username"
```

**方式2：直接修改代码**
编辑 `src/streamlit_app.py` 文件，找到以下行：
```python
HF_USERNAME = os.environ.get("HF_USERNAME", "your-username")
```
将 `"your-username"` 替换为您的HuggingFace用户名。

### 2. 安装依赖

确保已安装 `huggingface_hub`：
```bash
pip install huggingface_hub
```

### 3. 配置HuggingFace Token（可选）

如果您的模型是私有的，需要设置访问令牌：
```bash
# Windows PowerShell
$env:HF_TOKEN="your-huggingface-token"

# Linux/Mac
export HF_TOKEN="your-huggingface-token"
```

或者使用HuggingFace CLI登录：
```bash
huggingface-cli login
```

### 4. 启用HuggingFace模型

设置环境变量启用HuggingFace模型加载：
```bash
# Windows PowerShell
$env:USE_HF_MODELS="true"

# Linux/Mac
export USE_HF_MODELS="true"
```

**默认行为**：如果未设置 `USE_HF_MODELS`，默认使用HuggingFace模型（`USE_HF_MODELS=true`）。

要使用本地模型，设置：
```bash
export USE_HF_MODELS="false"
```

## 📁 HuggingFace模型仓库结构

您需要在HuggingFace上创建4个模型仓库：

### 1. X-VARS主模型权重
- **仓库名称**: `{username}/x-vars-weights`
- **内容**: 上传 `X-VARS/X-VARS_weights/` 目录中的所有文件
  - `model-00001-of-00003.safetensors`
  - `model-00002-of-00003.safetensors`
  - `model-00003-of-00003.safetensors`
  - `model.safetensors.index.json`
  - `config.json`
  - `generation_config.json`

### 2. Tokenizer
- **仓库名称**: `{username}/x-vars-tokenizer`
- **内容**: 上传 `X-VARS/tokenizer/` 目录中的所有文件
  - `tokenizer.model`
  - `tokenizer_config.json`
  - `special_tokens_map.json`

### 3. CLIP视觉编码器
- **仓库名称**: `{username}/clip-vit-large-patch14`
- **内容**: 上传 `visual_encoder/clip-vit-large-patch14/` 目录中的所有文件
  - `model.safetensors`
  - `config.json`
  - `preprocessor_config.json`

### 4. CLIP分类头权重
- **仓库名称**: `{username}/visual-encoder-14-model`
- **内容**: 上传 `visual_encoder/14_model.pth.tar` 文件
  - `14_model.pth.tar`

## 🚀 使用方法

### 方式1：使用环境变量（推荐）

```bash
# 设置用户名
export HF_USERNAME="your-username"

# 设置Token（如果模型是私有的）
export HF_TOKEN="your-token"

# 启用HuggingFace模型
export USE_HF_MODELS="true"

# 运行应用
streamlit run src/streamlit_app.py
```

### 方式2：直接修改代码

编辑 `src/streamlit_app.py`，修改以下配置：

```python
# 第103行：设置您的HuggingFace用户名
HF_USERNAME = "your-username"  # 直接设置，不使用环境变量

# 第112行：启用HuggingFace模型
USE_HUGGINGFACE_MODELS = True  # 改为True启用
```

## 🔍 验证配置

运行应用后，在侧边栏的"模型来源配置"部分可以看到：
- ✅ 使用HuggingFace模型
- 显示所有模型ID
- 如果huggingface_hub未安装，会显示警告

## ⚠️ 故障排除

### 问题1：无法从HuggingFace下载模型

**解决方案**：
1. 检查网络连接
2. 确认模型仓库名称正确
3. 确认模型仓库是公开的，或已设置正确的HF_TOKEN
4. 检查huggingface_hub是否已安装：`pip install huggingface_hub`

### 问题2：模型加载失败

**解决方案**：
1. 检查模型仓库中是否包含所有必需文件
2. 查看终端日志了解详细错误信息
3. 如果HuggingFace加载失败，可以回退到本地模型：
   ```bash
   export USE_HF_MODELS="false"
   ```

### 问题3：找不到模型文件

**解决方案**：
1. 确认模型仓库ID格式正确：`{username}/{repo-name}`
2. 确认所有文件都已上传到HuggingFace
3. 检查文件大小是否超过HuggingFace限制（单个文件最大10GB）

## 📝 注意事项

1. **首次加载**：从HuggingFace下载模型可能需要较长时间，取决于网络速度
2. **缓存**：下载的模型会缓存在本地，后续加载会更快
3. **存储空间**：确保有足够的磁盘空间存储下载的模型
4. **网络**：首次下载需要稳定的网络连接
5. **Token权限**：如果使用私有模型，确保Token有访问权限

## 🔄 切换模型来源

可以在运行时通过环境变量切换：

```bash
# 使用HuggingFace模型
export USE_HF_MODELS="true"
streamlit run src/streamlit_app.py

# 使用本地模型
export USE_HF_MODELS="false"
streamlit run src/streamlit_app.py
```

## 📚 相关文档

- [HuggingFace Hub文档](https://huggingface.co/docs/hub)
- [huggingface_hub库文档](https://huggingface.co/docs/huggingface_hub)

