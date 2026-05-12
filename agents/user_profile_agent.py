"""
用户画像Agent
- 实时特征提取：浏览/点击/购买/收藏行为 -> Redis Feature Store
- 用户分群：RFM模型 + 实时标签
- 画像合并：离线标签(T+1) + 在线标签(实时)
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import get_settings
from models.schemas import (
    AgentResult,
    UserProfile,
    UserProfileResult,
    UserSegment,
)

from .base_agent import BaseAgent

SYSTEM_PROMPT = """你是一个电商用户画像分析专家。根据用户的行为数据,分析用户特征并生成画像。

你需要输出以下JSON格式:
{
  "segments": ["new_user"|"active"|"high_value"|"price_sensitive"|"churn_risk"],
  "preferred_categories": ["类目1", "类目2"],
  "price_range": [最低价, 最高价],
  "rfm_score": {"recency": 0-1, "frequency": 0-1, "monetary": 0-1},
  "real_time_tags": {"活跃时段": "...", "偏好风格": "..."}
}

只输出JSON,不要其他内容。"""

class UserProfileAgent(BaseAgent):
    """
    用户画像代理
    """
    def __init__(self):
        """
        初始化用户画像代理实例。

        配置代理的基本属性（名称、超时时间），
        初始化大语言模型（LLM）客户端，
        并预留特征存储接口。
        """
        settings = get_settings()
        super().__init__(
            name="user_profile",
            timeout=settings.agent_timeout_user_profile,
        )
        # 初始化 ChatOpenAI 客户端，配置 API 密钥、基础 URL、模型及生成参数
        self.llm = ChatOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            temperature=0.3,
            max_tokens=1024,
        )
        # 初始化特征存储引用，后续由外部注入或延迟加载
        self.feature_store: Any = None
    async def _execute(self, **kwargs: Any) ->UserProfileResult:
        """
        执行用户画像生成流程。

        通过收集用户行为数据，结合大语言模型进行分析，最终解析并返回结构化的用户画像结果。

        Args:
            **kwargs: 关键字参数，包含以下字段：
                - user_id (str): 用户唯一标识ID。
                - context (dict, optional): 额外的上下文信息，默认为空字典。

        Returns:
            UserProfileResult: 包含成功状态、用户画像数据、原始分析内容及置信度的结果对象。
        """
        user_id: str = kwargs["user_id"]
        context: dict = kwargs.get("context", {})

        # 异步收集用户的行为数据
        #使用下面定义的函数获取行为数据
        behavior_data = await self._cllect_behavior(user_id, context)

        # 构建包含系统提示词和用户具体行为数据的消息列表
        massages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"用户ID:{user_id}\n行为数据:{json.dumps(behavior_data, ensure_ascii=False)}"),
        ]      
        # 调用大语言模型进行异步推理分析
        response = await self.llm.ainvoke(massages)

        # 解析模型返回的内容，提取结构化的用户画像数据
        #使用下面定义的解析函数获取数据
        profile_data = self._parse_profile(user_id, response.content)  

        return UserProfileResult(
            success=True,
            profile=profile_data,
            data={"raw_analysis": response.content, "behavior_data": behavior_data},
            confidence = 0.85
            )   

    async def _cllect_behavior(self, user_id: str, context: dict) -> dict:
        """
        异步收集用户行为数据。
        在_execute 方法中调用。
        根据用户ID和上下文信息，从特征存储中获取用户行为数据。

        Args:
            user_id (str): 用户唯一标识ID。
            context (dict): 额外的上下文信息。

        Returns:
            dict: 用户行为数据。
        """
        # 如果特征存储已注入，则从特征存储中异步获取用户行为数据
        if self.feature_store:
            return await self.feature_store.get_user_behavior(user_id)
        
        # 若特征存储未注入，则基于上下文信息构建并返回默认的用户行为数据
        return {
            "user_id": user_id,
            "recent_views": context.get("recent_views", ["手机", "耳机", "平板"]),
            "recent_purchases": context.get("recent_purchases", ["充电器"]),
            "view_count_7d": context.get("view_count_7d", 25),
            "purchase_count_30d": context.get("purchase_count_30d", 3),
            "avg_order_amount": context.get("avg_order_amount", 299.0),
            "active_hours": context.get("active_hours", [20, 21, 22])
            }
    
    def _parse_profile(self, user_id: str, raw: str) -> UserProfile:
        """解析用户原始数据字符串并构建 UserProfile 对象。
        在_execute 方法中调用。raw为response.content
        Args:
            user_id (str): 用户的唯一标识符。
            raw (str): 包含用户配置信息的原始字符串，可能包含 Markdown 代码块标记。

        Returns:
            UserProfile: 解析后的用户画像对象，包含分段、价格范围、偏好类别等信息。
        """
        try:
            cleaned = raw.strip()
            # 去除可能存在的 Markdown 代码块标记（```），提取中间的 JSON 内容
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n",1)[1].rsplit("```", 1)[0]
            data = json.loads(cleaned)
        except(json.JSONDecodeError, IndexError):
            # 若解析失败或格式错误，则使用空字典作为默认数据
            data = {}

        segment = []
        # 解析用户分段信息，将字符串映射为 UserSegment 枚举，忽略无效值
        for s in data.get("segments", ["active"]):
            try:
                segment.append(UserSegment(s))
            except (ValueError, KeyError):
                try:
                    segment.append(UserSegment[s.upper()])
                except (ValueError, KeyError):
                    continue
        
        price_range_raw = data.get("price_range", [0, 1000])
        # 构建价格范围元组，处理缺失上限的情况并转换为浮点数
        price_range = (
            float(price_range_raw[0]),
            float(price_range_raw[1]) if len(price_range_raw) > 1 else 10000.0
        )

        return UserProfile(
            user_id=user_id,
            segments=segment or [UserSegment.ACTIVE],
            preferred_categories=data.get("preferred_categories", []),
            price_range=price_range,
            rfm_score=data.get("rfm_score", {}),
            real_time_tags=data.get("real_time_tags", {})
        )