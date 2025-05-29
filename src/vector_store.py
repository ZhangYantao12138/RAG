"""
向量数据库模块
"""
import uuid
from typing import List, Dict, Optional, Any
import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# 配置日志
logger = logging.getLogger(__name__)

class QdrantManager:
    """Qdrant向量数据库管理器"""
    
    def __init__(self, url: str, api_key: str, collection_name: str = "script_collection"):
        self.url = url
        self.api_key = api_key
        self.collection_name = collection_name
        
        try:
            self.client = QdrantClient(
                url=url,
                api_key=api_key,
                timeout=60,
                verify=False  # 禁用SSL验证
            )
            logger.info(f"成功连接到Qdrant数据库: {url}")
        except Exception as e:
            logger.error(f"连接Qdrant数据库失败: {e}")
            raise
    
    def create_collection(self, vector_dim: int = 384, distance: Distance = Distance.COSINE) -> bool:
        """创建向量集合"""
        try:
            # 检查集合是否已存在
            collections = self.client.get_collections()
            existing_collections = [col.name for col in collections.collections]
            
            if self.collection_name in existing_collections:
                logger.info(f"集合 '{self.collection_name}' 已存在")
                return True
            
            # 创建新集合
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_dim,
                    distance=distance
                )
            )
            
            logger.info(f"成功创建集合 '{self.collection_name}'，向量维度: {vector_dim}")
            return True
            
        except Exception as e:
            logger.error(f"创建集合失败: {e}")
            return False
    
    def upload_documents(self, texts: List[str], vectors: List[List[float]], 
                        metadata: Optional[List[Dict]] = None, batch_size: int = 100) -> bool:
        """上传文档向量到数据库"""
        try:
            if len(texts) != len(vectors):
                raise ValueError("文本数量与向量数量不匹配")
            
            logger.info(f"开始上传 {len(texts)} 个文档向量")
            
            # 准备数据点
            points = []
            for i, (text, vector) in enumerate(zip(texts, vectors)):
                payload = {
                    "text": text,
                    "text_length": len(text),
                    "chunk_index": i
                }
                
                # 添加额外的元数据
                if metadata and i < len(metadata):
                    payload.update(metadata[i])
                
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload=payload
                )
                points.append(point)
            
            # 分批上传
            total_batches = (len(points) - 1) // batch_size + 1
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                batch_num = i // batch_size + 1
                
                logger.info(f"上传批次 {batch_num}/{total_batches}")
                
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
            
            logger.info(f"成功上传 {len(points)} 个文档向量")
            return True
            
        except Exception as e:
            logger.error(f"上传文档向量失败: {e}")
            return False
    
    def similarity_search(self, query_vector: List[float], top_k: int = 3, 
                         score_threshold: float = 0.0) -> List[Dict]:
        """相似度检索"""
        try:
            logger.debug(f"开始相似度检索，top_k: {top_k}")
            
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True
            )
            
            # 格式化结果
            results = []
            for point in search_result:
                result = {
                    "id": point.id,
                    "score": point.score,
                    "text": point.payload.get("text", ""),
                    "metadata": {k: v for k, v in point.payload.items() if k != "text"}
                }
                results.append(result)
            
            logger.debug(f"检索完成，返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"相似度检索失败: {e}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """获取集合信息"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            return {
                "collection_name": self.collection_name,
                "points_count": collection_info.points_count,
                "vectors_count": collection_info.vectors_count,
                "status": collection_info.status,
                "vector_size": collection_info.config.params.vectors.size,
                "distance": collection_info.config.params.vectors.distance
            }
            
        except Exception as e:
            logger.error(f"获取集合信息失败: {e}")
            return {}
    
    def delete_collection(self) -> bool:
        """删除集合"""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"成功删除集合 '{self.collection_name}'")
            return True
            
        except Exception as e:
            logger.error(f"删除集合失败: {e}")
            return False
    
    def clear_collection(self) -> bool:
        """清空集合中的所有数据"""
        try:
            # 获取所有点的ID
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000  # 一次性获取大量点
            )
            
            if scroll_result[0]:  # 如果有数据
                point_ids = [point.id for point in scroll_result[0]]
                
                # 删除所有点
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.PointIdsList(
                        points=point_ids
                    )
                )
                
                logger.info(f"成功清空集合 '{self.collection_name}'，删除了 {len(point_ids)} 个点")
            else:
                logger.info(f"集合 '{self.collection_name}' 已经为空")
            
            return True
            
        except Exception as e:
            logger.error(f"清空集合失败: {e}")
            return False
    
    def check_connection(self) -> bool:
        """检查数据库连接"""
        try:
            collections = self.client.get_collections()
            logger.info("数据库连接正常")
            return True
        except Exception as e:
            logger.error(f"数据库连接失败: {e}")
            return False 