# RAG系统伪代码文档

## 概述

本文档描述了RAG（检索增强生成）系统的三个核心模块的伪代码实现：

1. **rag_system.py** - RAG系统核心模块
2. **rag_chat.py** - RAG增强聊天模块  
3. **knowledge_manager.py** - 知识管理模块

---

## 1. rag_system.py - RAG系统核心模块

### 类定义：RAGSystem

```pseudocode
CLASS RAGSystem
    // 初始化RAG系统
    CONSTRUCTOR(embedding_model_name, vector_store_path, max_context_length, top_k)
        SET self.embedding_model_name = embedding_model_name
        SET self.vector_store_path = vector_store_path
        SET self.max_context_length = max_context_length
        SET self.top_k = top_k
        
        // 初始化嵌入模型
        IF torch.cuda.is_available() THEN
            SET device = 'cuda'
        ELSE
            SET device = 'cpu'
        END IF
        
        SET self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': device}
        )
        
        // 初始化文本分割器
        SET self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        // 初始化向量存储
        SET self.vector_store = None
        CALL self._load_or_create_vector_store()
        
        LOG "RAG系统初始化完成"
    END CONSTRUCTOR
    
    // 加载或创建向量存储
    METHOD _load_or_create_vector_store()
        IF os.path.exists(self.vector_store_path) THEN
            TRY
                SET self.vector_store = FAISS.load_local(
                    self.vector_store_path, 
                    self.embeddings
                )
                LOG "从路径加载向量存储"
            CATCH Exception as e
                LOG "加载向量存储失败，将创建新的"
                SET self.vector_store = None
            END TRY
        END IF
        
        IF self.vector_store IS None THEN
            // 创建空的向量存储
            SET self.vector_store = FAISS.from_texts(
                ["初始化文档"], 
                self.embeddings
            )
            CALL self._save_vector_store()
        END IF
    END METHOD
    
    // 保存向量存储
    METHOD _save_vector_store()
        CREATE_DIRECTORY(self.vector_store_path)
        CALL self.vector_store.save_local(self.vector_store_path)
        LOG "向量存储已保存"
    END METHOD
    
    // 添加文档到知识库
    METHOD add_documents(documents)
        SET processed_docs = []
        
        FOR EACH doc IN documents
            SET content = doc.get('content', '')
            SET metadata = doc.get('metadata', {})
            
            // 分割长文档
            SET chunks = CALL self.text_splitter.split_text(content)
            
            FOR i, chunk IN chunks
                SET chunk_metadata = metadata.copy()
                SET chunk_metadata['chunk_id'] = i
                SET chunk_metadata['total_chunks'] = len(chunks)
                
                ADD Document(page_content=chunk, metadata=chunk_metadata) TO processed_docs
            END FOR
        END FOR
        
        IF processed_docs IS NOT EMPTY THEN
            CALL self.vector_store.add_documents(processed_docs)
            CALL self._save_vector_store()
            LOG "成功添加文档块到知识库"
        END IF
    END METHOD
    
    // 从文本文件添加文档
    METHOD add_text_file(file_path, metadata)
        READ file_path INTO content
        IF metadata IS None THEN
            SET metadata = {}
        END IF
        
        SET metadata['source'] = file_path
        SET metadata['file_type'] = 'text'
        
        CALL self.add_documents([{
            'content': content,
            'metadata': metadata
        }])
    END METHOD
    
    // 从JSON文件添加文档
    METHOD add_json_file(file_path, content_key)
        READ file_path INTO data
        
        IF data IS LIST THEN
            SET documents = []
            FOR EACH item IN data
                IF content_key IN item THEN
                    ADD {
                        'content': item[content_key],
                        'metadata': {k: v for k, v in item.items() if k != content_key}
                    } TO documents
                END IF
            END FOR
        ELSE
            SET documents = [{
                'content': data.get(content_key, ''),
                'metadata': {k: v for k, v in data.items() if k != content_key}
            }]
        END IF
        
        CALL self.add_documents(documents)
    END METHOD
    
    // 检索相关文档
    METHOD retrieve_relevant_documents(query, top_k)
        IF top_k IS None THEN
            SET top_k = self.top_k
        END IF
        
        TRY
            SET docs = CALL self.vector_store.similarity_search(query, k=top_k)
            LOG "检索到相关文档"
            RETURN docs
        CATCH Exception as e
            LOG "文档检索失败"
            RETURN []
        END TRY
    END METHOD
    
    // 为查询获取上下文信息
    METHOD get_context_for_query(query, max_length)
        IF max_length IS None THEN
            SET max_length = self.max_context_length
        END IF
        
        // 检索相关文档
        SET relevant_docs = CALL self.retrieve_relevant_documents(query)
        
        IF relevant_docs IS EMPTY THEN
            RETURN ""
        END IF
        
        // 构建上下文
        SET context_parts = []
        SET current_length = 0
        
        FOR EACH doc IN relevant_docs
            SET doc_content = doc.page_content
            SET doc_metadata = doc.metadata
            
            // 添加元数据信息
            SET metadata_info = ""
            IF 'source' IN doc_metadata THEN
                SET metadata_info = f"[来源: {doc_metadata['source']}] "
            END IF
            
            SET formatted_content = metadata_info + doc_content
            
            // 检查长度限制
            IF current_length + len(formatted_content) > max_length THEN
                BREAK
            END IF
            
            ADD formatted_content TO context_parts
            SET current_length = current_length + len(formatted_content)
        END FOR
        
        SET context = JOIN(context_parts, "\n\n")
        
        IF context IS NOT EMPTY THEN
            SET context = "基于以下相关信息回答问题:\n\n" + context + "\n\n请基于以上信息回答用户的问题。"
        END IF
        
        RETURN context
    END METHOD
    
    // 获取知识库统计信息
    METHOD get_knowledge_base_stats()
        TRY
            IF self.vector_store IS None THEN
                RETURN {"total_documents": 0, "embedding_dimension": 0}
            END IF
            
            SET index = self.vector_store.index
            SET total_docs = index.ntotal IF hasattr(index, 'ntotal') ELSE 0
            SET embedding_dim = index.d IF hasattr(index, 'd') ELSE 0
            
            RETURN {
                "total_documents": total_docs,
                "embedding_dimension": embedding_dim,
                "embedding_model": self.embedding_model_name,
                "vector_store_path": self.vector_store_path
            }
        CATCH Exception as e
            LOG "获取知识库统计信息失败"
            RETURN {"error": str(e)}
        END TRY
    END METHOD
    
    // 清空知识库
    METHOD clear_knowledge_base()
        TRY
            // 重新创建空的向量存储
            SET self.vector_store = FAISS.from_texts(
                ["初始化文档"], 
                self.embeddings
            )
            CALL self._save_vector_store()
            LOG "知识库已清空"
        CATCH Exception as e
            LOG "清空知识库失败"
            THROW e
        END TRY
    END METHOD
END CLASS
```

---

## 2. rag_chat.py - RAG增强聊天模块

### 类定义：RAGChat

```pseudocode
CLASS RAGChat
    // 初始化RAG聊天模块
    CONSTRUCTOR(rag_system)
        SET self.rag_system = rag_system
        SET self.enable_rag = True  // 默认启用RAG
        SET self.rag_threshold = 0.3  // RAG相似度阈值
    END CONSTRUCTOR
    
    // 启用或禁用RAG模式
    METHOD enable_rag_mode(enable)
        SET self.enable_rag = enable
        LOG "RAG模式已启用/禁用"
    END METHOD
    
    // 设置RAG相似度阈值
    METHOD set_rag_threshold(threshold)
        SET self.rag_threshold = MAX(0.0, MIN(1.0, threshold))
        LOG "RAG相似度阈值已设置"
    END METHOD
    
    // 使用RAG增强查询
    METHOD enhance_query_with_context(query, max_context_length)
        IF NOT self.enable_rag THEN
            RETURN query
        END IF
        
        TRY
            // 获取相关上下文
            SET context = CALL self.rag_system.get_context_for_query(query, max_context_length)
            
            IF context IS NOT EMPTY THEN
                // 构建增强的查询
                SET enhanced_query = context + "\n\n用户问题: " + query
                LOG "查询已使用RAG增强"
                RETURN enhanced_query
            ELSE
                LOG "未找到相关上下文，使用原始查询"
                RETURN query
            END IF
        CATCH Exception as e
            LOG "RAG增强查询失败"
            RETURN query
        END TRY
    END METHOD
    
    // 获取相关文档信息
    METHOD get_relevant_documents(query, top_k)
        TRY
            SET docs = CALL self.rag_system.retrieve_relevant_documents(query, top_k)
            
            SET doc_info = []
            FOR EACH doc IN docs
                SET info = {
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'source': doc.metadata.get('source', '未知'),
                    'relevance_score': doc.metadata.get('score', 0.0)
                }
                ADD info TO doc_info
            END FOR
            
            RETURN doc_info
        CATCH Exception as e
            LOG "获取相关文档失败"
            RETURN []
        END TRY
    END METHOD
    
    // 添加知识文档
    METHOD add_knowledge_documents(documents)
        TRY
            CALL self.rag_system.add_documents(documents)
            LOG "成功添加知识文档"
        CATCH Exception as e
            LOG "添加知识文档失败"
            THROW e
        END TRY
    END METHOD
    
    // 获取知识库信息
    METHOD get_knowledge_base_info()
        RETURN CALL self.rag_system.get_knowledge_base_stats()
    END METHOD
    
    // 清空知识库
    METHOD clear_knowledge_base()
        TRY
            CALL self.rag_system.clear_knowledge_base()
            LOG "知识库已清空"
        CATCH Exception as e
            LOG "清空知识库失败"
            THROW e
        END TRY
    END METHOD
END CLASS
```

### 类定义：RAGEnhancedChat

```pseudocode
CLASS RAGEnhancedChat
    // 初始化RAG增强聊天
    CONSTRUCTOR(original_chat, rag_system)
        SET self.original_chat = original_chat
        SET self.rag_chat = RAGChat(rag_system)
        
        // 保存原始方法
        SET self.original_answer = original_chat.answer
        
        // 替换answer方法
        SET original_chat.answer = self.enhanced_answer
        
        LOG "RAG增强聊天模块初始化完成"
    END CONSTRUCTOR
    
    // RAG增强的答案生成方法
    METHOD enhanced_answer(state, img_list, temperature, max_output_tokens, first_run)
        TRY
            // 获取最后一条用户消息
            IF state.messages IS NOT EMPTY AND len(state.messages) > 0 THEN
                SET last_message = state.messages[-1]
                IF len(last_message) > 1 THEN
                    SET user_query = last_message[1]
                    
                    // 检查是否是文本查询
                    IF user_query IS STRING AND user_query.strip() IS NOT EMPTY THEN
                        // 使用RAG增强查询
                        SET enhanced_query = CALL self.rag_chat.enhance_query_with_context(user_query)
                        
                        // 更新状态中的查询
                        IF enhanced_query != user_query THEN
                            SET state.messages[-1][1] = enhanced_query
                            LOG "查询已使用RAG增强"
                        END IF
                    END IF
                END IF
            END IF
            
            // 调用原始方法
            RETURN CALL self.original_answer(state, img_list, temperature, max_output_tokens, first_run)
        CATCH Exception as e
            LOG "RAG增强答案生成失败"
            // 回退到原始方法
            RETURN CALL self.original_answer(state, img_list, temperature, max_output_tokens, first_run)
        END TRY
    END METHOD
    
    // 启用或禁用RAG功能
    METHOD enable_rag(enable)
        CALL self.rag_chat.enable_rag_mode(enable)
    END METHOD
    
    // 设置RAG相似度阈值
    METHOD set_rag_threshold(threshold)
        CALL self.rag_chat.set_rag_threshold(threshold)
    END METHOD
    
    // 添加知识文档
    METHOD add_knowledge(documents)
        CALL self.rag_chat.add_knowledge_documents(documents)
    END METHOD
    
    // 获取知识库信息
    METHOD get_knowledge_info()
        RETURN CALL self.rag_chat.get_knowledge_base_info()
    END METHOD
    
    // 清空知识库
    METHOD clear_knowledge()
        CALL self.rag_chat.clear_knowledge_base()
    END METHOD
    
    // 获取相关文档
    METHOD get_relevant_docs(query, top_k)
        RETURN CALL self.rag_chat.get_relevant_documents(query, top_k)
    END METHOD
END CLASS
```

---

## 3. knowledge_manager.py - 知识管理模块

### 类定义：KnowledgeManager

```pseudocode
CLASS KnowledgeManager
    // 初始化知识管理模块
    CONSTRUCTOR(rag_system)
        SET self.rag_system = rag_system
        SET self.supported_extensions = {'.txt', '.md', '.json', '.csv'}
    END CONSTRUCTOR
    
    // 上传文档到知识库
    METHOD upload_document(file)
        TRY
            IF file IS None THEN
                RETURN {"success": False, "message": "未选择文件"}
            END IF
            
            SET file_path = file.name
            SET file_extension = Path(file_path).suffix.lower()
            
            IF file_extension NOT IN self.supported_extensions THEN
                RETURN {
                    "success": False, 
                    "message": f"不支持的文件格式: {file_extension}"
                }
            END IF
            
            IF file_extension IN ['.txt', '.md'] THEN
                CALL self.rag_system.add_text_file(file_path)
            ELSE IF file_extension == '.json' THEN
                CALL self.rag_system.add_json_file(file_path)
            END IF
            
            SET stats = CALL self.rag_system.get_knowledge_base_stats()
            RETURN {
                "success": True,
                "message": f"文档 {Path(file_path).name} 上传成功",
                "stats": stats
            }
        CATCH Exception as e
            RETURN {"success": False, "message": f"文档上传失败: {str(e)}"}
        END TRY
    END METHOD
    
    // 添加文本内容到知识库
    METHOD add_text_content(content, title, source)
        TRY
            IF content.strip() IS EMPTY THEN
                RETURN {"success": False, "message": "内容不能为空"}
            END IF
            
            SET metadata = {
                'title': title, 
                'source': source OR '手动添加', 
                'file_type': 'text'
            }
            CALL self.rag_system.add_documents([{
                'content': content, 
                'metadata': metadata
            }])
            
            SET stats = CALL self.rag_system.get_knowledge_base_stats()
            RETURN {"success": True, "message": "文本内容添加成功", "stats": stats}
        CATCH Exception as e
            RETURN {"success": False, "message": f"添加文本内容失败: {str(e)}"}
        END TRY
    END METHOD
    
    // 搜索相关文档
    METHOD search_documents(query, top_k)
        TRY
            SET docs = CALL self.rag_system.retrieve_relevant_documents(query, top_k)
            SET results = []
            
            FOR i, doc IN docs
                SET result = {
                    'rank': i + 1,
                    'content': doc.page_content,
                    'source': doc.metadata.get('source', '未知'),
                    'title': doc.metadata.get('title', ''),
                    'metadata': doc.metadata
                }
                ADD result TO results
            END FOR
            
            RETURN results
        CATCH Exception as e
            RETURN []
        END TRY
    END METHOD
    
    // 获取知识库统计信息
    METHOD get_knowledge_stats()
        RETURN CALL self.rag_system.get_knowledge_base_stats()
    END METHOD
    
    // 清空知识库
    METHOD clear_knowledge_base()
        TRY
            CALL self.rag_system.clear_knowledge_base()
            RETURN {
                "success": True,
                "message": "知识库已清空",
                "stats": CALL self.rag_system.get_knowledge_base_stats()
            }
        CATCH Exception as e
            RETURN {"success": False, "message": f"清空知识库失败: {str(e)}"}
        END TRY
    END METHOD
END CLASS
```

### 函数定义：create_knowledge_ui

```pseudocode
FUNCTION create_knowledge_ui(knowledge_manager)
    // 文件上传处理函数
    FUNCTION upload_file_handler(file)
        SET result = CALL knowledge_manager.upload_document(file)
        SET stats = result.get("stats", {})
        SET stats_text = f"总文档数: {stats.get('total_documents', 0)}"
        RETURN result["message"], stats_text
    END FUNCTION
    
    // 添加文本处理函数
    FUNCTION add_text_handler(content, title, source)
        SET result = CALL knowledge_manager.add_text_content(content, title, source)
        SET stats = result.get("stats", {})
        SET stats_text = f"总文档数: {stats.get('total_documents', 0)}"
        RETURN result["message"], stats_text
    END FUNCTION
    
    // 搜索处理函数
    FUNCTION search_handler(query, top_k)
        SET results = CALL knowledge_manager.search_documents(query, int(top_k))
        IF results IS EMPTY THEN
            RETURN "未找到相关文档"
        END IF
        
        SET output = "搜索结果:\n\n"
        FOR EACH result IN results
            SET output = output + f"排名 {result['rank']}:\n"
            SET output = output + f"来源: {result['source']}\n"
            SET output = output + f"内容: {result['content'][:200]}...\n\n"
        END FOR
        RETURN output
    END FUNCTION
    
    // 清空处理函数
    FUNCTION clear_handler()
        SET result = CALL knowledge_manager.clear_knowledge_base()
        SET stats = result.get("stats", {})
        SET stats_text = f"总文档数: {stats.get('total_documents', 0)}"
        RETURN result["message"], stats_text
    END FUNCTION
    
    // 创建UI组件
    CREATE gr.Blocks(title="知识管理") AS knowledge_ui
    
    ADD gr.Markdown("# 📚 知识库管理")
    
    CREATE gr.Tabs()
    
    // 文件上传标签页
    WITH gr.TabItem("📁 文件上传")
        CREATE gr.File(label="选择文件", file_types=[".txt", ".md", ".json"]) AS file_input
        CREATE gr.Button("上传文档", variant="primary") AS upload_btn
        CREATE gr.Textbox(label="上传结果", interactive=False) AS upload_result
        CREATE gr.Textbox(label="知识库统计", interactive=False) AS upload_stats
        
        SET upload_btn.click(upload_file_handler, [file_input], [upload_result, upload_stats])
    END WITH
    
    // 文本添加标签页
    WITH gr.TabItem("✏️ 添加文本")
        CREATE gr.Textbox(label="文本内容", placeholder="请输入要添加到知识库的文本内容...", lines=10) AS text_content
        CREATE gr.Textbox(label="标题（可选）", placeholder="为内容添加标题...") AS text_title
        CREATE gr.Textbox(label="来源（可选）", placeholder="内容来源...") AS text_source
        CREATE gr.Button("添加文本", variant="primary") AS add_text_btn
        CREATE gr.Textbox(label="添加结果", interactive=False) AS add_text_result
        CREATE gr.Textbox(label="知识库统计", interactive=False) AS add_text_stats
        
        SET add_text_btn.click(add_text_handler, [text_content, text_title, text_source], [add_text_result, add_text_stats])
    END WITH
    
    // 搜索标签页
    WITH gr.TabItem("🔍 搜索文档")
        CREATE gr.Textbox(label="搜索查询", placeholder="输入搜索关键词...") AS search_query
        CREATE gr.Slider(minimum=1, maximum=20, value=5, step=1, label="返回结果数量") AS search_top_k
        CREATE gr.Button("搜索", variant="primary") AS search_btn
        CREATE gr.Textbox(label="搜索结果", interactive=False, lines=15) AS search_result
        
        SET search_btn.click(search_handler, [search_query, search_top_k], [search_result])
    END WITH
    
    // 管理标签页
    WITH gr.TabItem("⚙️ 知识库管理")
        CREATE gr.Button("🗑️ 清空知识库", variant="stop") AS clear_btn
        CREATE gr.Textbox(label="清空结果", interactive=False) AS clear_result
        CREATE gr.Textbox(label="知识库统计", interactive=False) AS clear_stats
        
        SET clear_btn.click(clear_handler, [], [clear_result, clear_stats])
    END WITH
    
    RETURN knowledge_ui
END FUNCTION
```

---

## 系统架构图

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Knowledge     │    │   RAGSystem     │    │   RAGChat       │
│   Manager       │    │                 │    │                 │
│                 │    │                 │    │                 │
│ • 文件上传      │───▶│ • 文档处理      │◀───│ • 查询增强      │
│ • 文本添加      │    │ • 向量存储      │    │ • 上下文检索    │
│ • 文档搜索      │    │ • 相似度检索    │    │ • 知识库管理    │
│ • 知识库管理    │    │ • 上下文构建    │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Gradio UI     │    │   FAISS         │    │   Chat System   │
│                 │    │   Vector Store  │    │                 │
│ • 用户界面      │    │ • 向量索引      │    │ • 对话管理      │
│ • 交互操作      │    │ • 相似度计算    │    │ • 回答生成      │
│ • 结果显示      │    │ • 持久化存储    │    │ • RAG集成       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

---

## 主要功能流程

### 1. 文档添加流程
```
用户上传文档 → KnowledgeManager → RAGSystem → 文本分割 → 向量化 → FAISS存储
```

### 2. 查询增强流程
```
用户查询 → RAGChat → RAGSystem → 相似度检索 → 上下文构建 → 查询增强 → 回答生成
```

### 3. 知识库管理流程
```
用户操作 → KnowledgeManager → RAGSystem → 统计信息/清空操作 → 结果返回
```

---

## 关键特性

1. **模块化设计**: 三个核心模块职责明确，易于维护和扩展
2. **错误处理**: 完善的异常处理机制，确保系统稳定性
3. **配置灵活**: 支持多种参数配置，适应不同使用场景
4. **用户友好**: 提供直观的Gradio界面，便于用户操作
5. **性能优化**: 使用FAISS进行高效的向量检索
6. **持久化存储**: 支持知识库的保存和加载

---

## 使用示例

```python
# 初始化RAG系统
rag_system = RAGSystem(
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
    vector_store_path="vector_store",
    max_context_length=2048,
    top_k=5
)

# 创建知识管理器
knowledge_manager = KnowledgeManager(rag_system)

# 创建RAG增强聊天
rag_enhanced_chat = RAGEnhancedChat(original_chat, rag_system)

# 添加知识文档
knowledge_manager.add_text_content("足球比赛规则...", "足球规则", "官方文档")

# 启用RAG功能
rag_enhanced_chat.enable_rag(True)
``` 