#!/usr/bin/env python3
"""
查看Qdrant数据库中的文本切片数据
"""
import os
import sys
from rich.console import Console
from rich.table import Table
from rich import print as rprint
from qdrant_client import QdrantClient
from qdrant_client.http import models

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import config
from vector_store import QdrantManager

console = Console()

def main():
    try:
        # 初始化向量数据库管理器
        vector_store = QdrantManager(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
            collection_name=config.COLLECTION_NAME
        )
        
        # 检查数据库连接
        if not vector_store.check_connection():
            rprint("错误: 数据库连接失败")
            return 1
        
        # 获取集合信息
        collection_info = vector_store.get_collection_info()
        if not collection_info:
            rprint("错误: 无法获取集合信息")
            return 1
        
        # 显示集合信息
        console.print("\n数据库信息:")
        table = Table(title="集合信息")
        table.add_column("属性", style="cyan")
        table.add_column("值", style="green")
        
        for key, value in collection_info.items():
            table.add_row(str(key), str(value))
        
        console.print(table)
        
        # 直接使用QdrantClient获取所有点
        client = QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY
        )
        
        # 获取所有点
        points = client.scroll(
            collection_name=config.COLLECTION_NAME,
            limit=100,  # 每次获取100个点
            with_payload=True,
            with_vectors=True
        )[0]  # scroll返回(records, next_page_offset)
        
        if not points:
            rprint("错误: 无法获取数据点")
            return 1
        
        # 生成输出文件
        output_file = os.path.join(
            'script',
            f"{config.COLLECTION_NAME}_db_chunks.md"
        )
        
        # 写入Markdown文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# 数据库中的文本切片\n\n")
            f.write(f"集合名称: {config.COLLECTION_NAME}\n")
            f.write(f"总切片数: {len(points)}\n\n")
            
            # 写入每个切片
            for i, point in enumerate(points, 1):
                f.write(f"## 切片 {i}\n\n")
                
                # 写入元数据
                if point.payload:
                    f.write("### 元数据\n\n")
                    for key, value in point.payload.items():
                        f.write(f"- {key}: {value}\n")
                    f.write("\n")
                
                # 写入文本内容
                f.write("### 文本内容\n\n")
                f.write("```\n")
                f.write(point.payload.get('text', ''))
                f.write("\n```\n\n")
                
                # 写入向量信息
                f.write("### 向量信息\n\n")
                f.write(f"- 向量维度: {len(point.vector)}\n")
                f.write(f"- 向量ID: {point.id}\n\n")
                
                f.write("---\n\n")
        
        console.print(f"\n切片数据已保存到: {output_file}")
        
        # 显示统计信息
        console.print("\n数据统计:")
        stats_table = Table()
        stats_table.add_column("指标", style="cyan")
        stats_table.add_column("值", style="green")
        
        stats_table.add_row("总切片数", str(len(points)))
        stats_table.add_row("向量维度", str(len(points[0].vector) if points else 0))
        
        # 计算平均文本长度
        avg_length = sum(len(p.payload.get('text', '')) for p in points) / len(points) if points else 0
        stats_table.add_row("平均文本长度", f"{avg_length:.2f}")
        
        console.print(stats_table)
        
        return 0
        
    except Exception as e:
        rprint(f"错误: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 