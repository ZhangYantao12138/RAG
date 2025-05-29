"""
配置管理模块
"""
import os
import sys
from typing import Optional
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加项目根目录到路径，以便导入config_values
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

class Config:
    """项目配置类"""
    
    def __init__(self):
        # 尝试导入配置文件
        try:
            import config_values as user_config
            self._load_from_file(user_config)
        except ImportError:
            self._load_from_env()
    
    def _load_from_file(self, config_module):
        """从配置文件加载配置"""
        # 嵌入模式配置
        self.USE_LOCAL_EMBEDDING = getattr(config_module, 'USE_LOCAL_EMBEDDING', False)
        
        # OpenAI API配置（用于嵌入）
        self.OPENAI_API_KEY = getattr(config_module, 'OPENAI_API_KEY', '')
        self.OPENAI_API_BASE = getattr(config_module, 'OPENAI_API_BASE', 'https://api.openai.com/v1')
        
        # DeepSeek API配置（用于聊天）
        self.DEEPSEEK_API_KEY = getattr(config_module, 'DEEPSEEK_API_KEY', '')
        self.DEEPSEEK_API_BASE = getattr(config_module, 'DEEPSEEK_API_BASE', 'https://api.deepseek.com/v1')
        
        # Qdrant Cloud配置
        self.QDRANT_URL = getattr(config_module, 'QDRANT_URL', '')
        self.QDRANT_API_KEY = getattr(config_module, 'QDRANT_API_KEY', '')
        self.COLLECTION_NAME = getattr(config_module, 'COLLECTION_NAME', 'script_collection')
        
        # 模型配置
        self.EMBEDDING_MODEL = getattr(config_module, 'EMBEDDING_MODEL', 'text-embedding-ada-002')
        self.CHAT_MODEL = getattr(config_module, 'CHAT_MODEL', 'deepseek-chat')
        self.VECTOR_DIM = getattr(config_module, 'VECTOR_DIM', 1536)
        
        # 处理参数
        self.CHUNK_SIZE = getattr(config_module, 'CHUNK_SIZE', 800)
        self.CHUNK_OVERLAP = getattr(config_module, 'CHUNK_OVERLAP', 100)
        self.TOP_K = getattr(config_module, 'TOP_K', 3)
        self.MAX_TOKENS = getattr(config_module, 'MAX_TOKENS', 1000)
        self.TEMPERATURE = getattr(config_module, 'TEMPERATURE', 0.7)
        
        # 文件路径配置
        self.SCRIPT_DIR = getattr(config_module, 'SCRIPT_DIR', 'script')
        self.DEFAULT_SCRIPT_FILE = getattr(config_module, 'DEFAULT_SCRIPT_FILE', '程聿怀男_本1_文字版.docx')
    
    def _load_from_env(self):
        """从环境变量加载配置"""
        # 嵌入模式配置
        self.USE_LOCAL_EMBEDDING = os.getenv('USE_LOCAL_EMBEDDING', 'False').lower() == 'true'
        
        # OpenAI API配置（用于嵌入）
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
        self.OPENAI_API_BASE = os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1')
        
        # DeepSeek API配置（用于聊天）
        self.DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', '')
        self.DEEPSEEK_API_BASE = os.getenv('DEEPSEEK_API_BASE', 'https://api.deepseek.com/v1')
        
        # Qdrant Cloud配置
        self.QDRANT_URL = os.getenv('QDRANT_URL', '')
        self.QDRANT_API_KEY = os.getenv('QDRANT_API_KEY', '')
        self.COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'script_collection')
        
        # 模型配置
        self.EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-ada-002')
        self.CHAT_MODEL = os.getenv('CHAT_MODEL', 'deepseek-chat')
        self.VECTOR_DIM = int(os.getenv('VECTOR_DIM', '1536'))
        
        # 处理参数
        self.CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '800'))
        self.CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '100'))
        self.TOP_K = int(os.getenv('TOP_K', '3'))
        self.MAX_TOKENS = int(os.getenv('MAX_TOKENS', '1000'))
        self.TEMPERATURE = float(os.getenv('TEMPERATURE', '0.7'))
        
        # 文件路径配置
        self.SCRIPT_DIR = os.getenv('SCRIPT_DIR', 'script')
        self.DEFAULT_SCRIPT_FILE = os.getenv('DEFAULT_SCRIPT_FILE', '程聿怀男_本1_文字版.docx')
    
    def validate(self) -> bool:
        """验证配置是否完整"""
        required_fields = [
            'OPENAI_API_KEY',      # 用于嵌入模型
            'DEEPSEEK_API_KEY',    # 用于聊天模型  
            'QDRANT_URL', 
            'QDRANT_API_KEY'
        ]
        
        for field in required_fields:
            if not getattr(self, field):
                return False
        return True
    
    def get_script_path(self) -> str:
        """获取默认剧本文件的完整路径"""
        return os.path.join(self.SCRIPT_DIR, self.DEFAULT_SCRIPT_FILE)

# 创建全局配置实例
config = Config() 