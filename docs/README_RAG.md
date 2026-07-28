# X-VARS RAG功能集成

## 📖 概述

本项目为X-VARS视频对话系统集成了RAG（检索增强生成）功能，使系统能够基于知识库提供更准确、更可靠的回答。

## ✨ 主要功能

### 🔍 智能检索
- 基于语义相似度的文档检索
- 支持多种文档格式（TXT、MD、JSON）
- 可配置的检索参数

### 📚 知识管理
- 直观的文档上传和管理界面
- 手动添加文本内容
- 知识库统计和监控
- 支持批量导入

### 🤖 增强对话
- 自动检索相关上下文
- 基于知识库的回答生成
- 可选的RAG开关控制

## 🚀 快速开始

### 1. 安装依赖

```bash
# 安装基础依赖
pip install -r requirements.txt

# 安装RAG相关依赖
pip install langchain==0.1.0
pip install langchain-community==0.0.10
pip install faiss-cpu==1.7.4
pip install chromadb==0.4.22
pip install sentence-transformers==2.2.2
pip install openai==1.3.0
pip install tiktoken==0.5.2
```

### 2. 运行测试

```bash
# 测试RAG功能
python test_rag.py
```

### 3. 启动演示

```bash
# 使用快速启动脚本
python start_rag_demo.py

# 或直接启动
python x_vars_demo_with_rag.py
```

## 📁 文件结构

```
├── x_vars_demo_with_rag.py    # RAG增强的主演示文件
├── rag_system.py              # RAG系统核心模块
├── rag_chat.py                # RAG增强聊天模块
├── knowledge_manager.py       # 知识管理模块
├── test_rag.py               # RAG功能测试脚本
├── start_rag_demo.py         # 快速启动脚本
├── sample_knowledge.json     # 示例知识库文件
├── requirements.txt          # 依赖包列表
├── RAG_使用说明.md           # 详细使用说明
└── README_RAG.md            # 本文档
```

## 🔧 核心模块

### 1. RAGSystem (`rag_system.py`)
RAG系统的核心类，负责：
- 文档处理和向量化
- 相似度检索
- 上下文生成
- 向量存储管理

### 2. RAGChat (`rag_chat.py`)
RAG增强的聊天模块，提供：
- 查询增强功能
- 相关文档检索
- 知识库管理接口

### 3. KnowledgeManager (`knowledge_manager.py`)
知识管理模块，包含：
- 文档上传功能
- 文本内容添加
- 知识库搜索
- 统计信息管理

## 🎯 使用示例

### 1. 基本使用

```python
from rag_system import RAGSystem
from knowledge_manager import KnowledgeManager

# 初始化RAG系统
rag_system = RAGSystem(
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
    vector_store_path="vector_store",
    max_context_length=2048,
    top_k=5
)

# 初始化知识管理器
km = KnowledgeManager(rag_system)

# 添加知识文档
documents = [
    {
        "content": "这是知识库中的内容...",
        "metadata": {"title": "示例文档", "source": "测试"}
    }
]
rag_system.add_documents(documents)

# 检索相关文档
docs = rag_system.retrieve_relevant_documents("用户问题", top_k=3)
```

### 2. 集成到X-VARS

```python
from rag_chat import RAGEnhancedChat

# 创建RAG增强聊天
rag_enhanced_chat = RAGEnhancedChat(original_chat, rag_system)

# 启用RAG功能
rag_enhanced_chat.enable_rag(True)

# 添加知识
rag_enhanced_chat.add_knowledge(documents)
```

## 🎨 界面功能

### 视频聊天标签页 💬
- 原有的视频上传和对话功能
- 新增RAG参数控制
- 实时RAG开关

### 知识管理标签页 📚
- **文件上传**：支持TXT、MD、JSON格式
- **添加文本**：手动输入知识内容
- **搜索文档**：基于关键词搜索
- **知识库管理**：清空和统计功能

### RAG设置标签页 ⚙️
- 系统状态监控
- 知识库统计信息
- 参数配置

## ⚙️ 配置参数

### RAG系统参数

```python
# 嵌入模型
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

# 向量存储路径
vector_store_path = "vector_store"

# 最大上下文长度
max_context_length = 2048

# 默认检索文档数量
top_k = 5
```

### 文档分割参数

```python
# 文档块大小
chunk_size = 1000

# 块重叠大小
chunk_overlap = 200
```

## 🔍 测试和验证

### 运行测试

```bash
# 运行完整测试套件
python test_rag.py

# 测试特定功能
python -c "from test_rag import test_rag_system; test_rag_system()"
```

### 验证功能

1. **基础功能测试**：RAG系统初始化、文档添加、检索
2. **知识管理测试**：文件上传、文本添加、搜索
3. **集成测试**：与X-VARS的完整集成

## 🐛 故障排除

### 常见问题

1. **依赖包安装失败**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **模型下载失败**
   - 检查网络连接
   - 使用国内镜像源
   - 手动下载模型文件

3. **内存不足**
   - 减少`max_context_length`
   - 降低`top_k`值
   - 使用更小的嵌入模型

4. **检索效果不佳**
   - 优化文档分割参数
   - 改进知识库内容质量
   - 调整相似度阈值

### 日志查看

```bash
# 查看系统日志
tail -f gradio_web_server.log

# 查看错误信息
grep "ERROR" gradio_web_server.log
```

## 📈 性能优化

### 1. 模型选择
- 使用更小的嵌入模型提高速度
- 根据需求选择合适的模型

### 2. 参数调优
- 调整文档块大小
- 优化检索参数
- 合理设置上下文长度

### 3. 硬件配置
- 使用GPU加速嵌入计算
- 增加内存容量
- 使用SSD存储

## 🔮 扩展功能

### 1. 多语言支持
```python
# 使用多语言嵌入模型
embedding_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```

### 2. 自定义嵌入模型
```python
# 使用自定义模型
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('your-custom-model')
```

### 3. 实时更新
- 实现知识库的实时更新
- 支持增量学习
- 动态文档管理

## 📚 参考资料

- [LangChain文档](https://python.langchain.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [FAISS文档](https://faiss.ai/)
- [RAG技术介绍](https://arxiv.org/abs/2005.11401)

## 🤝 贡献

欢迎提交Issue和Pull Request来改进RAG功能！

## 📄 许可证

本项目遵循原X-VARS项目的许可证。

---

**注意**：RAG功能需要额外的计算资源，建议在性能较好的机器上运行。 