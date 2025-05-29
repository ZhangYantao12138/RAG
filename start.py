#!/usr/bin/env python
"""
RAG原型验证系统启动脚本
自动初始化并启动问答系统
"""
import sys
import os
import subprocess
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich import print as rprint

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

console = Console()

def check_dependencies():
    """检查依赖是否安装"""
    try:
        import langchain
        import openai
        import qdrant_client
        import docx
        import rich
        return True
    except ImportError as e:
        rprint(f"错误: 缺少依赖: {e}")
        rprint("请运行: pip install -r requirements.txt")
        return False

def check_config():
    """检查配置是否有效"""
    try:
        from src.config import config
        if not config.validate():
            rprint("错误: 配置验证失败！请检查config_values.py中的API密钥")
            return False
        return True
    except Exception as e:
        rprint(f"错误: 配置加载失败: {e}")
        return False

def check_document():
    """检查文档是否存在"""
    from src.config import config
    script_path = config.get_script_path()
    if not os.path.exists(script_path):
        rprint(f"错误: 找不到文档文件: {script_path}")
        return False
    return True

def check_vector_data():
    """检查向量数据是否存在"""
    try:
        from src.config import config
        from vector_store import QdrantManager
        
        vector_store = QdrantManager(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
            collection_name=config.COLLECTION_NAME
        )
        
        if not vector_store.check_connection():
            rprint("错误: 数据库连接失败")
            return False
            
        collection_info = vector_store.get_collection_info()
        return collection_info and collection_info.get('points_count', 0) > 0
        
    except Exception as e:
        rprint(f"错误: 检查向量数据失败: {e}")
        return False

def upload_documents():
    """上传文档到向量数据库"""
    rprint("正在上传文档到向量数据库...")
    try:
        import upload_script
        result = upload_script.main()
        if result == 0:
            rprint("文档上传完成")
            return True
        else:
            rprint("文档上传失败")
            return False
    except Exception as e:
        rprint(f"错误: 上传过程出错: {e}")
        return False

def start_chat():
    """启动问答系统"""
    rprint("启动问答系统...")
    try:
        subprocess.call([sys.executable, "query_chat.py"])
    except KeyboardInterrupt:
        rprint("问答系统被中断")
    except Exception as e:
        rprint(f"错误: 启动问答系统失败: {e}")

def show_welcome():
    """显示欢迎信息"""
    welcome_text = """
# RAG原型验证系统

基于Qdrant Cloud + DeepSeek API + LangChain的智能问答系统

## 系统功能
- 智能文档处理和分块
- 文本向量化和存储  
- 语义相似度检索
- 智能问答生成
    """
    
    console.print(Panel(
        Markdown(welcome_text),
        title="欢迎使用",
        border_style="green"
    ))

def main():
    """主函数"""
    show_welcome()
    
    # 1. 检查依赖
    rprint("检查系统依赖...")
    if not check_dependencies():
        return 1
    
    # 2. 检查配置
    rprint("检查配置文件...")
    if not check_config():
        return 1
    
    # 3. 检查文档
    rprint("检查文档文件...")
    if not check_document():
        return 1
    
    # 4. 检查向量数据
    rprint("检查向量数据...")
    if not check_vector_data():
        rprint("向量数据不存在，需要先上传文档")
        if not upload_documents():
            return 1
    else:
        rprint("向量数据已存在")
    
    # 5. 启动问答系统
    rprint("系统准备就绪！")
    start_chat()
    
    return 0

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n程序被用户中断，再见！")
    except Exception as e:
        console.print(f"\n程序异常：{e}") 