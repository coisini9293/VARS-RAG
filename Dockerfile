# X-VARS 智能足球裁判系统 Dockerfile
# 使用Python 3.10.18

FROM python:3.10.18-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TRANSFORMERS_NO_TF=1 \
    TRANSFORMERS_NO_FLAX=1 \
    TOKENIZERS_PARALLELISM=false \
    DISABLE_CODECARBON=1

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir huggingface_hub

# 复制应用代码
COPY . .

# 创建必要的目录
RUN mkdir -p chat_history detailed_logs vector_store analysis_results

# 暴露Streamlit端口
EXPOSE 8501

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5m --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# 运行Streamlit应用
CMD ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]

