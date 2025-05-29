"""
向量化模块
"""
import time
from typing import List
import logging

# 配置日志
logger = logging.getLogger(__name__)

class EmbeddingManager:
    """向量化管理器（仅本地模型）"""
    
    def __init__(self, api_key: str = None, api_base: str = None, model: str = "all-MiniLM-L6-v2", use_local: bool = True):
        self.model = model
        self.use_local = True
        try:
            from sentence_transformers import SentenceTransformer
            self.local_model = SentenceTransformer(model)
            logger.info(f"初始化本地向量化模型: {model}")
        except ImportError:
            logger.error("需要安装 sentence-transformers: pip install sentence-transformers")
            raise

    def embed_query(self, text: str) -> List[float]:
        """对单个查询文本生成向量"""
        try:
            logger.debug(f"开始向量化查询文本，长度: {len(text)}")
            embedding = self.local_model.encode(text).tolist()
            logger.debug(f"查询向量化完成，维度: {len(embedding)}")
            return embedding
        except Exception as e:
            logger.error(f"查询向量化失败: {e}")
            raise

    def embed_documents(self, texts: List[str], batch_size: int = 100, delay: float = 0.1) -> List[List[float]]:
        """对文档列表生成向量，支持批量处理"""
        if not texts:
            return []
        logger.info(f"开始向量化 {len(texts)} 个文档")
        try:
            logger.info("使用本地模型进行向量化")
            embeddings = self.local_model.encode(texts).tolist()
            logger.info(f"文档向量化完成，共生成 {len(embeddings)} 个向量")
            return embeddings
        except Exception as e:
            logger.error(f"文档向量化失败: {e}")
            raise

    def validate_embedding(self, embedding: List[float], expected_dim: int = 384) -> bool:
        """验证向量的有效性"""
        if not isinstance(embedding, list):
            return False
        if len(embedding) != expected_dim:
            logger.warning(f"向量维度不匹配，期望: {expected_dim}，实际: {len(embedding)}")
            return False
        for val in embedding:
            if not isinstance(val, (int, float)) or val != val or abs(val) == float('inf'):
                return False
        return True

    def get_embedding_info(self) -> dict:
        """获取向量化服务信息"""
        return {
            "model": self.model,
            "type": "local",
            "expected_dimension": 384
        } 