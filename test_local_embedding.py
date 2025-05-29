#!/usr/bin/env python
"""
测试本地嵌入模型功能
"""
import logging
import torch
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.table import Table
import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

console = Console()

def test_embedding():
    """测试本地嵌入模型"""
    try:
        # 设置设备
        device = "cuda" if torch.cuda.is_available() else "cpu"
        console.print(f"[yellow]使用设备: {device}[/yellow]")
        
        # 初始化模型
        console.print("[yellow]正在加载本地模型...[/yellow]")
        model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        
        # 测试文本
        test_texts = [
            "这是一个测试句子",
            "这是另一个测试句子",
            "这是一个完全不同的句子"
        ]
        
        # 生成向量
        console.print("[yellow]正在生成文本向量...[/yellow]")
        embeddings = model.encode(test_texts, convert_to_numpy=True)
        
        # 显示结果
        table = Table(title="向量化结果")
        table.add_column("文本", style="cyan")
        table.add_column("向量维度", style="green")
        table.add_column("向量前5个值", style="blue")
        
        for text, embedding in zip(test_texts, embeddings):
            table.add_row(
                text,
                str(len(embedding)),
                str(embedding[:5].tolist())
            )
        
        console.print(table)
        
        # 计算相似度
        console.print("\n[yellow]计算文本相似度...[/yellow]")
        similarity_table = Table(title="文本相似度矩阵")
        similarity_table.add_column("文本", style="cyan")
        for text in test_texts:
            similarity_table.add_column(text[:10] + "...", style="green")
        
        # 计算余弦相似度
        for i, text1 in enumerate(test_texts):
            row = [text1]
            for j, text2 in enumerate(test_texts):
                similarity = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                row.append(f"{similarity:.4f}")
            similarity_table.add_row(*row)
        
        console.print(similarity_table)
        
        # 测试长文本
        console.print("\n[yellow]测试长文本处理...[/yellow]")
        long_text = "这是一个较长的测试文本。" * 10
        long_embedding = model.encode(long_text, convert_to_numpy=True)
        console.print(f"长文本向量维度: {len(long_embedding)}")
        
        # 测试批处理
        console.print("\n[yellow]测试批处理功能...[/yellow]")
        batch_texts = ["批处理测试" + str(i) for i in range(5)]
        batch_embeddings = model.encode(batch_texts, convert_to_numpy=True)
        console.print(f"批处理成功，生成了 {len(batch_embeddings)} 个向量")
        
        console.print("\n[green]✅ 所有测试完成！[/green]")
        
    except Exception as e:
        console.print(f"[red]❌ 测试过程中出现错误: {str(e)}[/red]")
        raise

if __name__ == "__main__":
    console.print("[bold blue]🧠 本地嵌入模型测试[/bold blue]\n")
    test_embedding() 