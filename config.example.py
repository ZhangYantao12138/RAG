"""
RAG原型验证系统配置文件模板
请复制此文件为 config_values.py 并填入实际的配置值
"""

# DeepSeek API配置
DEEPSEEK_API_KEY = "your_deepseek_api_key_here"  # 从DeepSeek平台获取
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"  # API基础URL

# Qdrant Cloud配置
QDRANT_URL = "your_qdrant_url_here"  # 从Qdrant Cloud获取
QDRANT_API_KEY = "your_qdrant_api_key_here"  # 从Qdrant Cloud获取
COLLECTION_NAME = "script_collection"  # 向量集合名称

# 模型配置
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 本地嵌入模型
CHAT_MODEL = "deepseek-chat"  # 对话模型
VECTOR_DIM = 384  # 向量维度

# 文档处理配置
CHUNK_SIZE = 800  # 文本块大小
CHUNK_OVERLAP = 100  # 文本块重叠大小

# 检索配置
TOP_K = 5  # 检索返回的相似文档数量
SCORE_THRESHOLD = 0.1  # 相似度阈值
KEYWORD_WEIGHT = 0.3  # 关键词匹配权重
VECTOR_WEIGHT = 0.7  # 向量相似度权重
MAX_CONTEXT_LENGTH = 2000  # 最大上下文长度

# 生成配置
MAX_TOKENS = 1000  # 最大生成长度
TEMPERATURE = 0.7  # 生成温度

# 本地配置
SCRIPT_PATH = "script.docx"  # 默认文档路径 