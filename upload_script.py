#!/usr/bin/env python3
"""
文档上传脚本
用于处理剧本文档并上传到向量数据库
"""
import sys
import os
import argparse
import logging
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich import print as rprint

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import config
from document_processor import DocumentProcessor
from embeddings import EmbeddingManager
from vector_store import QdrantManager

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

console = Console()

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
    parser = argparse.ArgumentParser(description="上传剧本文档到向量数据库")
    parser.add_argument("--file", type=str, help="指定要上传的文档文件路径")
    parser.add_argument("--clear", action="store_true", help="清空现有集合数据")
    parser.add_argument("--info", action="store_true", help="显示数据库信息")
    
    args = parser.parse_args()
    
    # 验证配置
    if not validate_config():
        return 1
    
    console.print("\nRAG原型验证系统 - 文档上传工具\n")
    
    try:
        # 初始化组件
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            
            init_task = progress.add_task("初始化系统组件...", total=None)
            
            # 初始化文档处理器
            doc_processor = DocumentProcessor()
            
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
            
            progress.update(init_task, completed=True)
            
        # 检查数据库连接
        if not vector_store.check_connection():
            rprint("错误: 数据库连接失败")
            return 1
        
        # 显示数据库信息
        if args.info:
            collection_info = vector_store.get_collection_info()
            if collection_info:
                table = Table(title="数据库信息")
                table.add_column("属性", style="cyan")
                table.add_column("值", style="green")
                
                for key, value in collection_info.items():
                    table.add_row(str(key), str(value))
                
                console.print(table)
            return 0
        
        # 清空集合
        if args.clear:
            console.print("准备清空现有数据...")
            if vector_store.clear_collection():
                rprint("数据清空完成")
            else:
                rprint("数据清空失败")
                return 1
        
        # 确定要处理的文件
        if args.file:
            file_path = args.file
        else:
            file_path = config.get_script_path()
        
        if not os.path.exists(file_path):
            rprint(f"错误: 文件不存在: {file_path}")
            return 1
        
        console.print(f"处理文件: {file_path}")
        
        # 显示文档统计信息
        doc_stats = doc_processor.get_document_stats(file_path)
        if doc_stats:
            table = Table(title="文档统计信息")
            table.add_column("属性", style="cyan")
            table.add_column("值", style="green")
            
            for key, value in doc_stats.items():
                table.add_row(str(key), str(value))
            
            console.print(table)
        
        # 处理文档
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            
            # 1. 处理文档
            process_task = progress.add_task("处理文档...", total=None)
            text_chunks = doc_processor.process_document(file_path)
            progress.update(process_task, completed=True)
            
            if not text_chunks:
                rprint("错误: 文档处理失败，未生成有效文本块")
                return 1
            
            console.print(f"成功生成 {len(text_chunks)} 个文本块")
            
            # 2. 创建向量集合
            collection_task = progress.add_task("创建向量集合...", total=None)
            if not vector_store.create_collection(vector_dim=config.VECTOR_DIM):
                rprint("错误: 创建向量集合失败")
                return 1
            progress.update(collection_task, completed=True)
            
            # 3. 生成向量
            embed_task = progress.add_task("生成文本向量...", total=None)
            vectors = embedding_manager.embed_documents(text_chunks)
            progress.update(embed_task, completed=True)
            
            if len(vectors) != len(text_chunks):
                rprint("错误: 向量生成失败")
                return 1
            
            console.print(f"成功生成 {len(vectors)} 个向量")
            
            # 4. 上传到数据库
            upload_task = progress.add_task("上传到向量数据库...", total=None)
            
            # 准备元数据
            metadata = []
            for i, chunk in enumerate(text_chunks):
                metadata.append({
                    "source_file": os.path.basename(file_path),
                    "chunk_id": i,
                    "upload_time": str(progress.get_time())
                })
            
            if vector_store.upload_documents(text_chunks, vectors, metadata):
                progress.update(upload_task, completed=True)
                rprint("文档上传完成！")
            else:
                rprint("文档上传失败")
                return 1
        
        # 显示最终统计
        final_info = vector_store.get_collection_info()
        if final_info:
            console.print("\n上传完成统计")
            table = Table()
            table.add_column("属性", style="cyan")
            table.add_column("值", style="green")
            
            table.add_row("集合名称", final_info.get("collection_name", ""))
            table.add_row("总向量数", str(final_info.get("points_count", 0)))
            table.add_row("向量维度", str(final_info.get("vector_size", 0)))
            table.add_row("状态", str(final_info.get("status", "")))
            
            console.print(table)
        
        console.print("\n系统已准备就绪，可以开始问答测试！")
        console.print("运行命令：python query_chat.py")
        
        return 0
        
    except KeyboardInterrupt:
        rprint("\n操作被用户中断")
        return 1
    except Exception as e:
        logger.error(f"上传过程中发生错误: {e}")
        rprint(f"错误: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 