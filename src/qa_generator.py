"""
问答生成模块
"""
from typing import List, Dict, Optional
import logging
from openai import OpenAI

# 配置日志
logger = logging.getLogger(__name__)

class QAGenerator:
    """问答生成器"""
    
    def __init__(self, api_key: str, api_base: str, model: str = "deepseek-chat"):
        self.api_key = api_key
        self.api_base = api_base
        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base
        )
        logger.info(f"初始化问答生成器，模型: {model}")
    
    def build_prompt(self, query: str, context: List[str], max_context_length: int = 2000) -> str:
        """构建包含上下文的prompt"""
        # 限制上下文长度
        combined_context = ""
        current_length = 0
        
        for text in context:
            if current_length + len(text) > max_context_length:
                break
            combined_context += text + "\n\n"
            current_length += len(text)
        
        prompt = f"""基于以下剧本内容回答用户问题。请确保回答准确、相关，并尽可能引用原文内容。

剧本内容：
{combined_context.strip()}

用户问题：{query}

请根据上述剧本内容回答问题。如果问题无法从给定内容中找到答案，请明确说明。"""

        return prompt
    
    def generate_answer(self, query: str, context: List[str], 
                       max_tokens: int = 1000, temperature: float = 0.7) -> Dict[str, any]:
        """生成回答"""
        try:
            logger.info(f"开始生成回答，问题: {query[:50]}...")
            
            # 构建prompt
            prompt = self.build_prompt(query, context)
            
            # 调用大模型
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "你是一个专业的剧本分析助手，能够根据提供的剧本内容准确回答相关问题。"
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            answer = response.choices[0].message.content
            
            result = {
                "answer": answer,
                "query": query,
                "context_used": len(context),
                "success": True,
                "error": None
            }
            
            logger.info("回答生成成功")
            return result
            
        except Exception as e:
            logger.error(f"生成回答失败: {e}")
            return {
                "answer": "抱歉，生成回答时出现错误。",
                "query": query,
                "context_used": 0,
                "success": False,
                "error": str(e)
            }
    
    def generate_summary(self, texts: List[str], max_tokens: int = 500) -> str:
        """生成文档摘要"""
        try:
            # 合并文本
            combined_text = "\n".join(texts)
            
            # 限制输入长度
            if len(combined_text) > 3000:
                combined_text = combined_text[:3000] + "..."
            
            prompt = f"""请为以下剧本内容生成一个简洁的摘要：

{combined_text}

请生成一个200字以内的摘要，包含主要情节和角色。"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个专业的文档摘要助手。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"生成摘要失败: {e}")
            return "摘要生成失败"
    
    def validate_response(self, response_dict: Dict) -> bool:
        """验证回答的有效性"""
        required_fields = ["answer", "query", "success"]
        
        for field in required_fields:
            if field not in response_dict:
                return False
        
        if not response_dict["success"]:
            return False
            
        if not response_dict["answer"] or len(response_dict["answer"].strip()) < 10:
            return False
        
        return True 