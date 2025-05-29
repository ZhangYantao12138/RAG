"""
文档处理模块
"""
import os
import logging
from typing import List, Dict
import markdown
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.config import config

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """文档处理器"""
    
    def __init__(self):
        self.min_chunk_size = config.CHUNK_SIZE - config.CHUNK_OVERLAP  # 最小块大小
        self.max_chunk_size = config.CHUNK_SIZE  # 最大块大小
        self.overlap_size = config.CHUNK_OVERLAP  # 重叠大小
        
        # 保留 LangChain 的分割器用于备用
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
        )
    
    def split_into_sentences(self, text: str) -> List[str]:
        """将文本分割成句子"""
        # 使用标点符号作为句子分隔符
        sentence_endings = ['。', '！', '？', '…', '；', '\n']
        sentences = []
        current = []
        
        for char in text:
            current.append(char)
            if char in sentence_endings:
                sentences.append(''.join(current))
                current = []
        
        if current:  # 处理最后一个句子
            sentences.append(''.join(current))
            
        return sentences
    
    def validate_document(self, file_path: str) -> bool:
        """验证文档是否存在且格式正确"""
        try:
            if not os.path.exists(file_path):
                logger.error(f"文档文件不存在: {file_path}")
                return False
            
            if not file_path.lower().endswith('.md'):
                logger.error(f"不支持的文档格式，仅支持.md格式: {file_path}")
                return False
            
            # 尝试读取文件验证格式
            with open(file_path, 'r', encoding='utf-8') as f:
                f.read()
            logger.info(f"文档验证成功: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"文档验证失败: {e}")
            return False
    
    def load_document(self, file_path: str) -> str:
        """加载Markdown文档并提取文本内容"""
        try:
            if not self.validate_document(file_path):
                raise ValueError(f"文档验证失败: {file_path}")
            
            logger.info(f"开始加载文档: {file_path}")
            
            # 读取Markdown文件
            with open(file_path, 'r', encoding='utf-8') as f:
                md_content = f.read()
            
            # 将Markdown转换为HTML
            html_content = markdown.markdown(md_content)
            
            # 使用BeautifulSoup提取纯文本
            soup = BeautifulSoup(html_content, 'html.parser')
            text_content = soup.get_text()
            
            if not text_content.strip():
                raise ValueError("文档内容为空")
            
            logger.info(f"文档加载成功，提取了 {len(text_content)} 个字符")
            return text_content
            
        except Exception as e:
            logger.error(f"文档加载失败: {e}")
            raise
    
    def split_text(self, text: str) -> List[str]:
        """智能文本切片，保持上下文重叠"""
        if not text:
            return []
        
        logger.info(f"开始分割文本，原始长度: {len(text)}")
        
        # 将文本分割成句子
        sentences = self.split_into_sentences(text)
        
        # 进行切片
        chunks = []
        current_chunk = []
        current_size = 0
        overlap_buffer = []  # 用于存储重叠部分的句子
        
        for i, sentence in enumerate(sentences):
            current_chunk.append(sentence)
            current_size += len(sentence)
            
            # 检查是否需要切分
            should_split = False
            
            # 1. 超过最大大小必须切分
            if current_size >= self.max_chunk_size:
                should_split = True
            # 2. 达到最小大小且不是最后一句时切分
            elif current_size >= self.min_chunk_size and i < len(sentences)-1:
                should_split = True
                
            if should_split:
                if current_chunk:
                    # 保存当前块
                    chunk_text = ''.join(current_chunk)
                    chunks.append(chunk_text)
                    
                    # 准备重叠部分
                    overlap_buffer = []
                    overlap_size = 0
                    
                    # 从当前块的末尾开始，收集重叠部分
                    for sent in reversed(current_chunk):
                        if overlap_size + len(sent) <= self.overlap_size:
                            overlap_buffer.insert(0, sent)
                            overlap_size += len(sent)
                        else:
                            break
                    
                    # 重置当前块，但保留重叠部分
                    current_chunk = overlap_buffer.copy()
                    current_size = overlap_size
        
        # 添加最后一个块
        if current_chunk:
            chunk_text = ''.join(current_chunk)
            chunks.append(chunk_text)
        
        logger.info(f"文本分割完成，共生成 {len(chunks)} 个文本块")
        return chunks
    
    def process_document(self, file_path: str) -> List[str]:
        """完整的文档处理流程"""
        try:
            logger.info(f"开始处理文档: {file_path}")
            
            # 1. 加载文档
            text = self.load_document(file_path)
            
            # 2. 分割文本
            text_chunks = self.split_text(text)
            
            logger.info(f"文档处理完成，共生成 {len(text_chunks)} 个文本块")
            return text_chunks
            
        except Exception as e:
            logger.error(f"文档处理失败: {e}")
            raise
    
    def get_document_stats(self, file_path: str) -> dict:
        """获取文档统计信息"""
        try:
            text = self.load_document(file_path)
            chunks = self.split_text(text)
            
            return {
                "file_path": file_path,
                "file_size_mb": round(os.path.getsize(file_path) / 1024 / 1024, 2),
                "text_length": len(text),
                "total_chunks": len(chunks),
                "avg_chunk_size": round(sum(len(chunk) for chunk in chunks) / len(chunks), 2) if chunks else 0,
                "min_chunk_size": self.min_chunk_size,
                "max_chunk_size": self.max_chunk_size,
                "overlap_size": self.overlap_size
            }
        except Exception as e:
            logger.error(f"获取文档统计信息失败: {e}")
            return {} 