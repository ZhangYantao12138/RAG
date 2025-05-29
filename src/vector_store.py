"""
向量数据库模块
"""
import uuid
import jieba
import jieba.analyse
from typing import List, Dict, Optional, Any, Tuple
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

    def get_document_by_id(self, chunk_id: int) -> Optional[Dict]:
        """根据chunk_id获取文档"""
        try:
            # 使用scroll方法获取所有点
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,  # 获取足够多的点
                with_payload=True,
                with_vectors=True
            )
            
            # 查找匹配的文档
            for point in scroll_result[0]:
                if point.payload.get("chunk_id") == chunk_id:
                    return {
                        "id": point.id,
                        "text": point.payload.get("text", ""),
                        "vector": point.vector,
                        "metadata": {k: v for k, v in point.payload.items() if k != "text"}
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"获取文档失败: {e}")
            return None

    def calculate_vector_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """计算两个向量的余弦相似度"""
        try:
            import numpy as np
            
            # 转换为numpy数组
            v1 = np.array(vector1)
            v2 = np.array(vector2)
            
            # 计算余弦相似度
            dot_product = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"计算向量相似度失败: {e}")
            return 0.0

    def extract_keywords(self, text: str) -> List[str]:
        """提取文本关键词"""
        try:
            keywords = set()  # 使用集合避免重复
            
            # 1. 使用jieba的TF-IDF提取关键词
            tfidf_keywords = jieba.analyse.extract_tags(text, topK=10, withWeight=False)
            keywords.update(tfidf_keywords)
            
            # 2. 使用jieba的TextRank提取关键词（可能包含不同的重要词）
            textrank_keywords = jieba.analyse.textrank(text, topK=10, withWeight=False)
            keywords.update(textrank_keywords)
            
            # 3. 分词并识别各类实体
            words = jieba.lcut(text)
            for word in words:
                # 人名识别（2字及以上）
                if len(word) >= 2 and any(c in word for c in ['程', '李', '张', '王', '刘', '陈', '杨', '黄', '赵', '周', '吴', '郑', '孙', '马', '朱', '胡', '林', '郭', '何', '高']):
                    keywords.add(word)
                
                # 地点识别（包含特定词）
                if any(c in word for c in ['市', '区', '县', '镇', '村', '路', '街', '巷', '楼', '院', '园', '馆', '店', '厂', '站', '场']):
                    keywords.add(word)
                
                # 时间识别
                if any(c in word for c in ['年', '月', '日', '时', '分', '秒', '天', '周', '星期', '早上', '中午', '下午', '晚上', '凌晨']):
                    keywords.add(word)
                
                # 数字+单位组合
                if any(c in word for c in ['个', '只', '条', '张', '本', '台', '辆', '次', '回', '遍', '趟', '顿', '场', '阵']):
                    keywords.add(word)
                
                # 重要动作/状态词（2字及以上）
                if len(word) >= 2 and any(c in word for c in ['说', '想', '看', '做', '走', '跑', '跳', '唱', '哭', '笑', '吃', '喝', '睡', '醒', '死', '活']):
                    keywords.add(word)
            
            # 4. 添加完整的数字
            import re
            numbers = re.findall(r'\d+', text)
            keywords.update(numbers)
            
            # 5. 添加完整的英文单词
            english_words = re.findall(r'[a-zA-Z]+', text)
            keywords.update(english_words)
            
            # 转换为列表并过滤掉太短的词
            keywords = [k for k in keywords if len(k) >= 2 or k.isdigit()]
            
            logger.debug(f"提取的关键词: {keywords}")
            return list(keywords)
            
        except Exception as e:
            logger.error(f"关键词提取失败: {e}")
            return []

    def calculate_keyword_score(self, query_keywords: List[str], text: str) -> float:
        """计算关键词匹配分数"""
        try:
            if not query_keywords:
                return 0.0
            
            # 提取文本中的所有关键词
            text_keywords = set(self.extract_keywords(text))
            
            # 计算匹配分数
            matches = 0
            for kw in query_keywords:
                # 完全匹配
                if kw in text_keywords:
                    matches += 1
                # 部分匹配（对于较长的词）
                elif len(kw) >= 3:
                    # 检查是否包含在文本关键词中
                    for tk in text_keywords:
                        if kw in tk or tk in kw:
                            matches += 0.5
                            break
                
                # 数字匹配（允许相近数字）
                if kw.isdigit():
                    for tk in text_keywords:
                        if tk.isdigit():
                            diff = abs(int(kw) - int(tk))
                            if diff <= 1:  # 允许误差为1
                                matches += 0.8
                                break
            
            # 归一化分数
            return min(matches / len(query_keywords), 1.0)
            
        except Exception as e:
            logger.error(f"关键词分数计算失败: {e}")
            return 0.0

    def hybrid_search(self, query: str, query_vector: List[float], top_k: int = 5,
                     score_threshold: float = 0.05, keyword_weight: float = 0.3,
                     vector_weight: float = 0.7) -> List[Dict]:
        """混合检索（向量相似度 + 关键词匹配）"""
        try:
            # 提取查询关键词
            query_keywords = self.extract_keywords(query)
            
            # 向量相似度检索
            vector_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=1000,  # 获取足够多的结果用于混合排序
                score_threshold=0.0,  # 先不过滤，等计算完混合分数再过滤
                with_payload=True
            )
            
            # 计算混合分数
            hybrid_results = []
            for point in vector_results:
                text = point.payload.get("text", "")
                vector_score = point.score
                keyword_score = self.calculate_keyword_score(query_keywords, text)
                
                # 计算混合分数
                hybrid_score = (vector_score * vector_weight + 
                              keyword_score * keyword_weight)
                
                result = {
                    "id": point.id,
                    "hybrid_score": hybrid_score,
                    "vector_score": vector_score,
                    "keyword_score": keyword_score,
                    "text": text,
                    "metadata": {k: v for k, v in point.payload.items() if k != "text"}
                }
                hybrid_results.append(result)
            
            # 按混合分数排序
            hybrid_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
            
            # 过滤低分结果
            filtered_results = [r for r in hybrid_results if r["hybrid_score"] >= score_threshold]
            
            # 返回top_k个结果
            final_results = filtered_results[:top_k]
            
            return final_results
            
        except Exception as e:
            logger.error(f"混合检索失败: {e}")
            return [] 