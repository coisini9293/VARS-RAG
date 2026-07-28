import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import gradio as gr
from rag_system import RAGSystem

logger = logging.getLogger(__name__)

class KnowledgeManager:
    """知识管理模块，提供文档上传和管理功能"""
    
    def __init__(self, rag_system: RAGSystem):
        self.rag_system = rag_system
        self.supported_extensions = {'.txt', '.md', '.json', '.csv'}
        
    def upload_document(self, file) -> Dict[str, Any]:
        """上传文档到知识库"""
        try:
            if file is None:
                return {"success": False, "message": "未选择文件"}
            
            file_path = file.name
            file_extension = Path(file_path).suffix.lower()
            
            if file_extension not in self.supported_extensions:
                return {
                    "success": False, 
                    "message": f"不支持的文件格式: {file_extension}"
                }
            
            if file_extension in ['.txt', '.md']:
                self.rag_system.add_text_file(file_path)
            elif file_extension == '.json':
                self.rag_system.add_json_file(file_path)
            
            stats = self.rag_system.get_knowledge_base_stats()
            return {
                "success": True,
                "message": f"文档 {Path(file_path).name} 上传成功",
                "stats": stats
            }
            
        except Exception as e:
            return {"success": False, "message": f"文档上传失败: {str(e)}"}
    
    def add_text_content(self, content: str, title: str = "", source: str = "") -> Dict[str, Any]:
        """添加文本内容到知识库"""
        try:
            if not content.strip():
                return {"success": False, "message": "内容不能为空"}
            
            metadata = {'title': title, 'source': source or '手动添加', 'file_type': 'text'}
            self.rag_system.add_documents([{'content': content, 'metadata': metadata}])
            
            stats = self.rag_system.get_knowledge_base_stats()
            return {"success": True, "message": "文本内容添加成功", "stats": stats}
            
        except Exception as e:
            return {"success": False, "message": f"添加文本内容失败: {str(e)}"}
    
    def search_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """搜索相关文档"""
        try:
            docs = self.rag_system.retrieve_relevant_documents(query, top_k)
            results = []
            for i, doc in enumerate(docs):
                result = {
                    'rank': i + 1,
                    'content': doc.page_content,
                    'source': doc.metadata.get('source', '未知'),
                    'title': doc.metadata.get('title', ''),
                    'metadata': doc.metadata
                }
                results.append(result)
            return results
        except Exception as e:
            return []
    
    def get_knowledge_stats(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        return self.rag_system.get_knowledge_base_stats()
    
    def clear_knowledge_base(self) -> Dict[str, Any]:
        """清空知识库"""
        try:
            self.rag_system.clear_knowledge_base()
            return {
                "success": True,
                "message": "知识库已清空",
                "stats": self.rag_system.get_knowledge_base_stats()
            }
        except Exception as e:
            return {"success": False, "message": f"清空知识库失败: {str(e)}"}


def create_knowledge_ui(knowledge_manager: KnowledgeManager):
    """创建知识管理UI界面"""
    
    def upload_file_handler(file):
        result = knowledge_manager.upload_document(file)
        stats = result.get("stats", {})
        stats_text = f"总文档数: {stats.get('total_documents', 0)}"
        return result["message"], stats_text
    
    def add_text_handler(content, title, source):
        result = knowledge_manager.add_text_content(content, title, source)
        stats = result.get("stats", {})
        stats_text = f"总文档数: {stats.get('total_documents', 0)}"
        return result["message"], stats_text
    
    def search_handler(query, top_k):
        results = knowledge_manager.search_documents(query, int(top_k))
        if not results:
            return "未找到相关文档"
        
        output = "搜索结果:\n\n"
        for result in results:
            output += f"排名 {result['rank']}:\n"
            output += f"来源: {result['source']}\n"
            output += f"内容: {result['content'][:200]}...\n\n"
        return output
    
    def clear_handler():
        result = knowledge_manager.clear_knowledge_base()
        stats = result.get("stats", {})
        stats_text = f"总文档数: {stats.get('total_documents', 0)}"
        return result["message"], stats_text
    
    # 创建UI组件
    with gr.Blocks(title="知识管理") as knowledge_ui:
        gr.Markdown("# 📚 知识库管理")
        
        with gr.Tabs():
            # 文件上传标签页
            with gr.TabItem("📁 文件上传"):
                file_input = gr.File(label="选择文件", file_types=[".txt", ".md", ".json"])
                upload_btn = gr.Button("上传文档", variant="primary")
                upload_result = gr.Textbox(label="上传结果", interactive=False)
                upload_stats = gr.Textbox(label="知识库统计", interactive=False)
                
                upload_btn.click(upload_file_handler, [file_input], [upload_result, upload_stats])
            
            # 文本添加标签页
            with gr.TabItem("✏️ 添加文本"):
                text_content = gr.Textbox(label="文本内容", placeholder="请输入要添加到知识库的文本内容...", lines=10)
                text_title = gr.Textbox(label="标题（可选）", placeholder="为内容添加标题...")
                text_source = gr.Textbox(label="来源（可选）", placeholder="内容来源...")
                add_text_btn = gr.Button("添加文本", variant="primary")
                add_text_result = gr.Textbox(label="添加结果", interactive=False)
                add_text_stats = gr.Textbox(label="知识库统计", interactive=False)
                
                add_text_btn.click(add_text_handler, [text_content, text_title, text_source], [add_text_result, add_text_stats])
            
            # 搜索标签页
            with gr.TabItem("🔍 搜索文档"):
                search_query = gr.Textbox(label="搜索查询", placeholder="输入搜索关键词...")
                search_top_k = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="返回结果数量")
                search_btn = gr.Button("搜索", variant="primary")
                search_result = gr.Textbox(label="搜索结果", interactive=False, lines=15)
                
                search_btn.click(search_handler, [search_query, search_top_k], [search_result])
            
            # 管理标签页
            with gr.TabItem("⚙️ 知识库管理"):
                clear_btn = gr.Button("🗑️ 清空知识库", variant="stop")
                clear_result = gr.Textbox(label="清空结果", interactive=False)
                clear_stats = gr.Textbox(label="知识库统计", interactive=False)
                
                clear_btn.click(clear_handler, [], [clear_result, clear_stats])
    
    return knowledge_ui 