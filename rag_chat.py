import os
import logging
from typing import List, Dict, Any, Optional
from rag_system import RAGSystem

logger = logging.getLogger(__name__)

class RAGChat:
    """RAG增强的聊天模块"""
    
    def __init__(self, rag_system: RAGSystem):
        """
        初始化RAG聊天模块
        
        Args:
            rag_system: RAG系统实例
        """
        self.rag_system = rag_system
        self.enable_rag = True  # 默认启用RAG
        self.rag_threshold = 0.3  # RAG相似度阈值
        
    def enable_rag_mode(self, enable: bool = True):
        """启用或禁用RAG模式"""
        self.enable_rag = enable
        logger.info(f"RAG模式已{'启用' if enable else '禁用'}")
    
    def set_rag_threshold(self, threshold: float):
        """设置RAG相似度阈值"""
        self.rag_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"RAG相似度阈值设置为: {self.rag_threshold}")
    
    def enhance_query_with_context(self, query: str, max_context_length: Optional[int] = None) -> str:
        """
        使用RAG增强查询
        
        Args:
            query: 原始查询
            max_context_length: 最大上下文长度
            
        Returns:
            增强后的查询
        """
        if not self.enable_rag:
            return query
        
        try:
            # 获取相关上下文
            context = self.rag_system.get_context_for_query(query, max_context_length)
            
            if context:
                # 构建增强的查询
                enhanced_query = f"{context}\n\n用户问题: {query}"
                logger.info(f"查询已使用RAG增强，上下文长度: {len(context)}")
                return enhanced_query
            else:
                logger.info("未找到相关上下文，使用原始查询")
                return query
                
        except Exception as e:
            logger.error(f"RAG增强查询失败: {e}")
            return query
    
    def get_relevant_documents(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        获取相关文档信息
        
        Args:
            query: 查询文本
            top_k: 返回的文档数量
            
        Returns:
            相关文档信息列表
        """
        try:
            docs = self.rag_system.retrieve_relevant_documents(query, top_k)
            
            doc_info = []
            for doc in docs:
                info = {
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'source': doc.metadata.get('source', '未知'),
                    'relevance_score': doc.metadata.get('score', 0.0)
                }
                doc_info.append(info)
            
            return doc_info
            
        except Exception as e:
            logger.error(f"获取相关文档失败: {e}")
            return []
    
    def add_knowledge_documents(self, documents: List[Dict[str, Any]]):
        """
        添加知识文档
        
        Args:
            documents: 文档列表
        """
        try:
            self.rag_system.add_documents(documents)
            logger.info(f"成功添加 {len(documents)} 个知识文档")
        except Exception as e:
            logger.error(f"添加知识文档失败: {e}")
            raise
    
    def get_knowledge_base_info(self) -> Dict[str, Any]:
        """
        获取知识库信息
        
        Returns:
            知识库统计信息
        """
        return self.rag_system.get_knowledge_base_stats()
    
    def clear_knowledge_base(self):
        """清空知识库"""
        try:
            self.rag_system.clear_knowledge_base()
            logger.info("知识库已清空")
        except Exception as e:
            logger.error(f"清空知识库失败: {e}")
            raise


class RAGEnhancedChat:
    """RAG增强的聊天类，集成到现有的Chat类中"""
    
    def __init__(self, original_chat, rag_system: RAGSystem):
        """
        初始化RAG增强聊天
        
        Args:
            original_chat: 原始Chat实例
            rag_system: RAG系统实例
        """
        self.original_chat = original_chat
        self.rag_chat = RAGChat(rag_system)
        
        # 保存原始方法
        self.original_answer = original_chat.answer
        
        # 替换answer方法
        original_chat.answer = self.enhanced_answer
        
        logger.info("RAG增强聊天模块初始化完成")
    
    def enhanced_answer(self, state, img_list, temperature, max_output_tokens, first_run):
        """
        RAG增强的答案生成方法
        
        Args:
            state: 对话状态
            img_list: 图片列表
            temperature: 温度参数
            max_output_tokens: 最大输出token数
            first_run: 是否首次运行
            
        Returns:
            增强后的回答
        """
        try:
            # 获取最后一条用户消息
            if state.messages and len(state.messages) > 0:
                last_message = state.messages[-1]
                if len(last_message) > 1:
                    user_query = last_message[1]
                    
                    # 检查是否是文本查询
                    if isinstance(user_query, str) and user_query.strip():
                        # 使用RAG增强查询
                        enhanced_query = self.rag_chat.enhance_query_with_context(user_query)
                        
                        # 更新状态中的查询
                        if enhanced_query != user_query:
                            state.messages[-1][1] = enhanced_query
                            logger.info("查询已使用RAG增强")
            
            # 调用原始方法
            return self.original_answer(state, img_list, temperature, max_output_tokens, first_run)
            
        except Exception as e:
            logger.error(f"RAG增强答案生成失败: {e}")
            # 回退到原始方法
            return self.original_answer(state, img_list, temperature, max_output_tokens, first_run)
    
    def enable_rag(self, enable: bool = True):
        """启用或禁用RAG功能"""
        self.rag_chat.enable_rag_mode(enable)
    
    def set_rag_threshold(self, threshold: float):
        """设置RAG相似度阈值"""
        self.rag_chat.set_rag_threshold(threshold)
    
    def add_knowledge(self, documents: List[Dict[str, Any]]):
        """添加知识文档"""
        self.rag_chat.add_knowledge_documents(documents)
    
    def get_knowledge_info(self) -> Dict[str, Any]:
        """获取知识库信息"""
        return self.rag_chat.get_knowledge_base_info()
    
    def clear_knowledge(self):
        """清空知识库"""
        self.rag_chat.clear_knowledge_base()
    
    def get_relevant_docs(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取相关文档"""
        return self.rag_chat.get_relevant_documents(query, top_k) 