#!/usr/bin/env python
"""
智能问答脚本
提供基于RAG的智能问答功能
"""
import sys
import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich import print as rprint
from rich.prompt import Prompt

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import config
from embeddings import EmbeddingManager
from vector_store import QdrantManager
from qa_generator import QAGenerator

# 配置日志
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

console = Console()

class RAGChatBot:
    """RAG聊天机器人"""
    
    def __init__(self):
        self.console = console
        self.embedding_manager = None
        self.vector_store = None
        self.qa_generator = None
        self.chat_history = []
        
    def initialize(self) -> bool:
        """初始化所有组件"""
        try:
            console.print("初始化系统组件...")
            
            # 初始化向量化管理器
            self.embedding_manager = EmbeddingManager(
                api_key=config.OPENAI_API_KEY,
                api_base=config.OPENAI_API_BASE,
                model=config.EMBEDDING_MODEL,
                use_local=config.USE_LOCAL_EMBEDDING
            )
            
            # 初始化向量数据库
            self.vector_store = QdrantManager(
                url=config.QDRANT_URL,
                api_key=config.QDRANT_API_KEY,
                collection_name=config.COLLECTION_NAME
            )
            
            # 初始化问答生成器
            self.qa_generator = QAGenerator(
                api_key=config.DEEPSEEK_API_KEY,
                api_base=config.DEEPSEEK_API_BASE,
                model=config.CHAT_MODEL
            )
            
            # 检查数据库连接和数据
            if not self.vector_store.check_connection():
                rprint("错误: 数据库连接失败")
                return False
            
            collection_info = self.vector_store.get_collection_info()
            if not collection_info or collection_info.get('points_count', 0) == 0:
                rprint("错误: 向量数据库中没有数据，请先运行上传脚本")
                rprint("运行命令：python upload_script.py")
                return False
            
            console.print("系统初始化完成")
            
            # 显示数据库信息
            self.show_database_info(collection_info)
            
            return True
            
        except Exception as e:
            rprint(f"错误: 初始化失败: {e}")
            return False
    
    def show_database_info(self, info: Dict):
        """显示数据库信息"""
        table = Table(title="数据库状态", show_header=False)
        table.add_column("属性", style="cyan", width=15)
        table.add_column("值", style="green")
        
        table.add_row("集合名称", str(info.get("collection_name", "")))
        table.add_row("文档数量", str(info.get("points_count", 0)))
        table.add_row("向量维度", str(info.get("vector_size", 0)))
        table.add_row("状态", str(info.get("status", "")))
        
        console.print(table)
    
    def search_similar_documents(self, query: str) -> List[Dict]:
        """搜索相似文档"""
        try:
            # 将查询向量化
            query_vector = self.embedding_manager.embed_query(query)
            
            # 使用混合检索
            results = self.vector_store.hybrid_search(
                query=query,
                query_vector=query_vector,
                top_k=config.TOP_K,
                score_threshold=config.SCORE_THRESHOLD,
                keyword_weight=config.KEYWORD_WEIGHT,
                vector_weight=config.VECTOR_WEIGHT
            )
            
            return results
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []
    
    def generate_answer(self, query: str, context_docs: List[Dict]) -> Dict:
        """生成回答"""
        try:
            # 提取文本内容
            context_texts = [doc["text"] for doc in context_docs]
            
            # 生成回答
            result = self.qa_generator.generate_answer(
                query=query,
                context=context_texts,
                max_tokens=config.MAX_TOKENS,
                temperature=config.TEMPERATURE
            )
            
            return result
            
        except Exception as e:
            logger.error(f"生成回答失败: {e}")
            return {
                "answer": "抱歉，生成回答时出现错误。",
                "success": False,
                "error": str(e)
            }
    
    def show_search_results(self, results: List[Dict], show_details: bool = False):
        """显示搜索结果"""
        if not results:
            console.print("未找到相关内容")
            return
        
        console.print(f"\n找到 {len(results)} 个相关文档片段：")
        
        for i, result in enumerate(results, 1):
            hybrid_score = result.get("hybrid_score", 0)
            vector_score = result.get("vector_score", 0)
            keyword_score = result.get("keyword_score", 0)
            text = result.get("text", "")
            
            # 限制显示长度
            display_text = text[:200] + "..." if len(text) > 200 else text
            
            panel_content = f"混合分数: {hybrid_score:.3f}\n"
            panel_content += f"向量相似度: {vector_score:.3f}\n"
            panel_content += f"关键词匹配: {keyword_score:.3f}\n\n"
            panel_content += display_text
            
            if show_details:
                metadata = result.get("metadata", {})
                if metadata:
                    panel_content += f"\n\n元数据: {metadata}"
            
            console.print(Panel(
                panel_content,
                title=f"片段 {i}",
                border_style="blue"
            ))
    
    def process_query(self, query: str, show_context: bool = False) -> bool:
        """处理单个查询"""
        console.print(f"\n问题：{query}")
        
        # 搜索相关文档
        console.print("搜索相关内容...")
        context_docs = self.search_similar_documents(query)
        
        if show_context:
            self.show_search_results(context_docs, show_details=True)
        
        if not context_docs:
            console.print("未找到相关内容，无法生成回答")
            return True
        
        # 生成回答
        console.print("生成回答...")
        result = self.generate_answer(query, context_docs)
        
        if result.get("success", False):
            answer = result["answer"]
            
            # 显示回答
            console.print(Panel(
                Markdown(answer),
                title="AI 回答",
                border_style="green"
            ))
            
            # 记录对话历史
            self.chat_history.append({
                "query": query,
                "answer": answer,
                "context_count": len(context_docs)
            })
            
        else:
            console.print(Panel(
                result.get("answer", "生成回答失败"),
                title="错误",
                border_style="red"
            ))
        
        return True
    
    def show_help(self):
        """显示帮助信息"""
        help_text = """
可用命令：

• 直接输入问题 - 进行问答查询
• /help - 显示此帮助信息  
• /info - 显示数据库信息
• /history - 显示对话历史
• /context - 切换上下文显示模式
• /clear - 清空对话历史
• /quit 或 /exit - 退出程序

示例问题：
• 这个剧本的主要人物有哪些？
• 故事发生在什么地方？
• 主要情节是什么？
        """
        
        console.print(Panel(
            Markdown(help_text),
            title="使用帮助",
            border_style="blue"
        ))
    
    def show_history(self):
        """显示对话历史"""
        if not self.chat_history:
            console.print("暂无对话历史")
            return
        
        console.print(f"\n对话历史 (共 {len(self.chat_history)} 条)：")
        
        for i, item in enumerate(self.chat_history, 1):
            console.print(f"\n{i}. 问题：{item['query']}")
            console.print(f"回答：{item['answer'][:100]}...")
    
    def run(self):
        """运行交互式问答"""
        # 显示欢迎信息
        welcome_text = """
# RAG原型验证系统

欢迎使用基于剧本内容的智能问答系统！

您可以询问关于剧本的任何问题，系统会基于已上传的文档内容进行回答。

输入 /help 查看可用命令，输入 /quit 退出程序。
        """
        
        console.print(Panel(
            Markdown(welcome_text),
            title="欢迎使用",
            border_style="green"
        ))
        
        show_context = False
        
        try:
            while True:
                # 获取用户输入
                query = Prompt.ask("\n请输入您的问题").strip()
                
                if not query:
                    continue
                
                # 处理特殊命令
                if query.lower() in ["/quit", "/exit", "退出", "quit", "exit"]:
                    console.print("再见！")
                    break
                
                elif query.lower() in ["/help", "帮助"]:
                    self.show_help()
                
                elif query.lower() in ["/info", "信息"]:
                    collection_info = self.vector_store.get_collection_info()
                    if collection_info:
                        self.show_database_info(collection_info)
                
                elif query.lower() in ["/history", "历史"]:
                    self.show_history()
                
                elif query.lower() in ["/context", "上下文"]:
                    show_context = not show_context
                    status = "开启" if show_context else "关闭"
                    console.print(f"上下文显示已{status}")
                
                elif query.lower() in ["/clear", "清空"]:
                    self.chat_history.clear()
                    console.print("对话历史已清空")
                
                else:
                    # 处理正常问题
                    self.process_query(query, show_context)
                
        except KeyboardInterrupt:
            console.print("\n程序被用户中断，再见！")
        except Exception as e:
            console.print(f"\n程序异常：{e}")

def validate_config():
    """验证配置文件"""
    if not config.validate():
        rprint("错误: 配置验证失败！")
        rprint("请确保以下配置项已正确填写：")
        rprint("- DEEPSEEK_API_KEY")
        rprint("- QDRANT_URL")
        rprint("- QDRANT_API_KEY")
        rprint("\n请复制 config.example.py 为 config.py 并填入正确的配置信息")
        return False
    return True

def main():
    """主函数"""
    # 验证配置
    if not validate_config():
        return 1
    
    # 创建聊天机器人实例
    chatbot = RAGChatBot()
    
    # 初始化系统
    if not chatbot.initialize():
        return 1
    
    # 开始交互
    chatbot.run()
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 