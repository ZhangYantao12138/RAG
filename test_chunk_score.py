#!/usr/bin/env python
"""
测试文档块相关性分数
用于测试特定文档块与问题的相关性分数
"""
import sys
import os
import logging
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import config
from embeddings import EmbeddingManager
from vector_store import QdrantManager

# 配置日志
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

console = Console()

def test_chunk_score(query: str, target_chunk_id: int):
    """测试特定文档块与问题的相关性分数"""
    try:
        # 初始化组件
        console.print("初始化系统组件...")
        
        # 初始化向量化管理器
        embedding_manager = EmbeddingManager(
            api_key=config.OPENAI_API_KEY,
            api_base=config.OPENAI_API_BASE,
            model=config.EMBEDDING_MODEL,
            use_local=config.USE_LOCAL_EMBEDDING
        )
        
        # 初始化向量数据库
        vector_store = QdrantManager(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
            collection_name=config.COLLECTION_NAME
        )
        
        # 检查数据库连接
        if not vector_store.check_connection():
            console.print("错误: 数据库连接失败")
            return False
        
        # 将查询向量化
        query_vector = embedding_manager.embed_query(query)
        
        # 获取目标文档块
        target_chunk = vector_store.get_document_by_id(target_chunk_id)
        if not target_chunk:
            console.print(f"错误: 未找到ID为 {target_chunk_id} 的文档块")
            return False
        
        # 计算相关性分数
        vector_score = vector_store.calculate_vector_similarity(query_vector, target_chunk["vector"])
        keyword_score = vector_store.calculate_keyword_score(
            vector_store.extract_keywords(query),
            target_chunk["text"]
        )
        
        # 计算混合分数
        hybrid_score = (
            vector_score * config.VECTOR_WEIGHT +
            keyword_score * config.KEYWORD_WEIGHT
        )
        
        # 显示结果
        console.print(f"\n问题：{query}")
        console.print(f"目标文档块ID：{target_chunk_id}")
        
        # 创建结果表格
        table = Table(title="相关性分数分析")
        table.add_column("指标", style="cyan")
        table.add_column("分数", style="green")
        table.add_column("权重", style="yellow")
        
        table.add_row(
            "混合分数",
            f"{hybrid_score:.3f}",
            "1.0"
        )
        table.add_row(
            "向量相似度",
            f"{vector_score:.3f}",
            f"{config.VECTOR_WEIGHT}"
        )
        table.add_row(
            "关键词匹配",
            f"{keyword_score:.3f}",
            f"{config.KEYWORD_WEIGHT}"
        )
        
        console.print(table)
        
        # 显示文档内容
        console.print("\n文档内容：")
        console.print(Panel(
            target_chunk["text"],
            title="文档块内容",
            border_style="blue"
        ))
        
        return True
        
    except Exception as e:
        console.print(f"错误: {e}")
        return False

def main():
    """主函数"""
    if len(sys.argv) != 3:
        console.print("用法: python test_chunk_score.py <问题> <文档块ID>")
        console.print("示例: python test_chunk_score.py '这个故事发生在哪里？' 1")
        return 1
    
    query = sys.argv[1]
    try:
        chunk_id = int(sys.argv[2])
    except ValueError:
        console.print("错误: 文档块ID必须是整数")
        return 1
    
    if test_chunk_score(query, chunk_id):
        return 0
    return 1

if __name__ == "__main__":
    sys.exit(main()) 