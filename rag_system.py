import os
import json
import pickle
from typing import List, Dict, Any, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
import logging

logger = logging.getLogger(__name__)

class RAGSystem:
    """RAG系统核心类，负责文档检索和知识增强"""
    
    def __init__(self, 
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 vector_store_path: str = "vector_store",
                 max_context_length: int = 2048,
                 top_k: int = 5):
        """
        初始化RAG系统
        
        Args:
            embedding_model_name: 嵌入模型名称
            vector_store_path: 向量存储路径
            max_context_length: 最大上下文长度
            top_k: 检索的文档数量
        """
        self.embedding_model_name = embedding_model_name
        self.vector_store_path = vector_store_path
        self.max_context_length = max_context_length
        self.top_k = top_k
        
        # 初始化嵌入模型
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )
        
        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # 初始化向量存储
        self.vector_store = None
        self._load_or_create_vector_store()
        
        logger.info(f"RAG系统初始化完成，使用模型: {embedding_model_name}")
    
    def _load_or_create_vector_store(self):
        """加载或创建向量存储"""
        if os.path.exists(self.vector_store_path):
            try:
                self.vector_store = FAISS.load_local(
                    self.vector_store_path, 
                    self.embeddings
                )
                logger.info(f"从 {self.vector_store_path} 加载向量存储")
            except Exception as e:
                logger.warning(f"加载向量存储失败: {e}，将创建新的向量存储")
                self.vector_store = None
        
        if self.vector_store is None:
            # 创建空的向量存储
            self.vector_store = FAISS.from_texts(
                ["初始化文档"], 
                self.embeddings
            )
            self._save_vector_store()
    
    def _save_vector_store(self):
        """保存向量存储"""
        os.makedirs(self.vector_store_path, exist_ok=True)
        self.vector_store.save_local(self.vector_store_path)
        logger.info(f"向量存储已保存到 {self.vector_store_path}")
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        添加文档到知识库
        
        Args:
            documents: 文档列表，每个文档包含 'content' 和 'metadata' 字段
        """
        try:
            # 处理文档
            processed_docs = []
            for doc in documents:
                content = doc.get('content', '')
                metadata = doc.get('metadata', {})
                
                # 分割长文档
                chunks = self.text_splitter.split_text(content)
                
                for i, chunk in enumerate(chunks):
                    chunk_metadata = metadata.copy()
                    chunk_metadata['chunk_id'] = i
                    chunk_metadata['total_chunks'] = len(chunks)
                    
                    processed_docs.append(Document(
                        page_content=chunk,
                        metadata=chunk_metadata
                    ))
            
            # 添加到向量存储
            if processed_docs:
                self.vector_store.add_documents(processed_docs)
                self._save_vector_store()
                logger.info(f"成功添加 {len(processed_docs)} 个文档块到知识库")
            
        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            raise
    
    def add_text_file(self, file_path: str, metadata: Optional[Dict] = None):
        """
        从文本文件添加文档
        
        Args:
            file_path: 文件路径
            metadata: 元数据
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if metadata is None:
                metadata = {}
            
            metadata['source'] = file_path
            metadata['file_type'] = 'text'
            
            self.add_documents([{
                'content': content,
                'metadata': metadata
            }])
            
        except Exception as e:
            logger.error(f"添加文本文件失败 {file_path}: {e}")
            raise
    
    def add_json_file(self, file_path: str, content_key: str = 'content'):
        """
        从JSON文件添加文档
        
        Args:
            file_path: JSON文件路径
            content_key: 内容字段名
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                documents = []
                for item in data:
                    if content_key in item:
                        documents.append({
                            'content': item[content_key],
                            'metadata': {k: v for k, v in item.items() if k != content_key}
                        })
            else:
                documents = [{
                    'content': data.get(content_key, ''),
                    'metadata': {k: v for k, v in data.items() if k != content_key}
                }]
            
            self.add_documents(documents)
            
        except Exception as e:
            logger.error(f"添加JSON文件失败 {file_path}: {e}")
            raise
    
    def retrieve_relevant_documents(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        """
        检索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回的文档数量
            
        Returns:
            相关文档列表
        """
        if top_k is None:
            top_k = self.top_k
        
        try:
            # 使用向量存储检索相关文档
            docs = self.vector_store.similarity_search(query, k=top_k)
            logger.info(f"检索到 {len(docs)} 个相关文档")
            return docs
            
        except Exception as e:
            logger.error(f"文档检索失败: {e}")
            return []
    
    def get_context_for_query(self, query: str, max_length: Optional[int] = None) -> str:
        """
        为查询获取上下文信息
        
        Args:
            query: 查询文本
            max_length: 最大上下文长度
            
        Returns:
            格式化的上下文字符串
        """
        if max_length is None:
            max_length = self.max_context_length
        
        # 检索相关文档
        relevant_docs = self.retrieve_relevant_documents(query)
        
        if not relevant_docs:
            return ""
        
        # 构建上下文
        context_parts = []
        current_length = 0
        
        for doc in relevant_docs:
            doc_content = doc.page_content
            doc_metadata = doc.metadata
            
            # 添加元数据信息
            metadata_info = ""
            if 'source' in doc_metadata:
                metadata_info = f"[来源: {doc_metadata['source']}] "
            
            formatted_content = f"{metadata_info}{doc_content}"
            
            # 检查长度限制
            if current_length + len(formatted_content) > max_length:
                break
            
            context_parts.append(formatted_content)
            current_length += len(formatted_content)
        
        context = "\n\n".join(context_parts)
        
        if context:
            context = f"基于以下相关信息回答问题:\n\n{context}\n\n请基于以上信息回答用户的问题。"
        
        return context
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """
        获取知识库统计信息
        
        Returns:
            统计信息字典
        """
        try:
            if self.vector_store is None:
                return {"total_documents": 0, "embedding_dimension": 0}
            
            # 获取向量存储信息
            index = self.vector_store.index
            total_docs = index.ntotal if hasattr(index, 'ntotal') else 0
            embedding_dim = index.d if hasattr(index, 'd') else 0
            
            return {
                "total_documents": total_docs,
                "embedding_dimension": embedding_dim,
                "embedding_model": self.embedding_model_name,
                "vector_store_path": self.vector_store_path
            }
            
        except Exception as e:
            logger.error(f"获取知识库统计信息失败: {e}")
            return {"error": str(e)}
    
    def clear_knowledge_base(self):
        """清空知识库"""
        try:
            # 重新创建空的向量存储
            self.vector_store = FAISS.from_texts(
                ["初始化文档"], 
                self.embeddings
            )
            self._save_vector_store()
            logger.info("知识库已清空")
            
        except Exception as e:
            logger.error(f"清空知识库失败: {e}")
            raise 