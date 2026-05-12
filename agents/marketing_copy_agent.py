"""
营销文案Agent
- Prompt模板引擎：基于用户画像动态选择模板(新客/老客/高价值)
- 个性化生成：调用deepseek生成文案
- 合规校验：敏感词过滤 + 广告法合规检查
"""
from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import get_settings
from models.schemas import (
    MarketingCopyResult,
    Product,
    UserProfile,
    UserSegment,
)

from .base_agent import BaseAgent

PROMPT_TEMPLATES = {
    UserSegment.NEW_USER: """你是电商营销文案专家。为新用户撰写欢迎+推荐文案。
风格要求：热情友好、突出新人专属优惠感、降低决策门槛。
每个商品生成一条文案(30-50字)。""",

    UserSegment.HIGH_VALUE: """你是电商营销文案专家。为高价值VIP用户撰写推荐文案。
风格要求：品质感、尊享感、突出商品高端属性和品牌价值。
每个商品生成一条文案(30-50字)。""",

    UserSegment.PRICE_SENSITIVE: """你是电商营销文案专家。为价格敏感用户撰写推荐文案。
风格要求：突出性价比、促销价格、限时优惠、省钱金额。
每个商品生成一条文案(30-50字)。""",

    UserSegment.ACTIVE: """你是电商营销文案专家。为活跃用户撰写推荐文案。
风格要求：突出商品亮点和使用场景,引发共鸣。
每个商品生成一条文案(30-50字)。""",

    UserSegment.CHURN_RISK: """你是电商营销文案专家。为即将流失的用户撰写召回文案。
风格要求：情感唤回、专属折扣、限时活动、制造紧迫感。
每个商品生成一条文案(30-50字)。""",
}

FORBIDDEN_WORDS = [
    "最好", "第一", "国家级", "全球首", "绝对", "100%",
    "永久", "万能", "祖传", "纯天然",
]

COPY_OUTPUT_INSTRUCTION = """
请以JSON数组格式输出,每个元素格式:
[{"product_id": "xxx", "copy": "文案内容"}]
只输出JSON,不要其他内容。"""


class MarketingCopyAgent(BaseAgent):
    """营销文案生成代理类。

    该类继承自 BaseAgent，专门用于生成营销相关的文案内容。
    初始化时配置代理名称、超时时间以及底层的大语言模型（LLM）实例。
    """

    def __init__(self):
        """初始化营销文案代理实例。

        从配置中加载相关设置，初始化父类代理的基本属性（名称和超时时间），
        并配置用于生成文案的 ChatOpenAI 语言模型实例。

        Args:
            无显式参数，所有配置均通过 get_settings() 获取。

        Returns:
            无返回值。
        """
        settings = get_settings()
        super().__init__(
            name="marketing_copy",
            timeout=settings.agent_timeout_marketing_copy,
        )
        # 初始化大语言模型实例，配置API密钥、基础URL、模型名称及生成参数
        self.llm = ChatOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            temperature=0.9,
            max_tokens=2048,
        )


    async def _execute(self, **kwargs: Any) -> MarketingCopyResult:
        """
        异步执行营销文案生成任务。

        根据用户画像和商品列表，选择合适的提示词模板，调用大语言模型生成营销文案，
        并对生成的文案进行解析和合规性检查。

        Args:
            **kwargs: 关键字参数，包含以下可选键：
                - user_profile (UserProfile | None): 用户画像信息，用于选择提示词模板。
                - products (list[Product]): 商品列表，用于生成针对具体商品的营销文案。默认为空列表。

        Returns:
            MarketingCopyResult: 包含生成结果的对象，具体字段如下：
                - success (bool): 执行是否成功。
                - copies (list[str]): 经过合规性检查后的营销文案列表。
                - confidence (float): 结果置信度。
                - prompt_template_used (str): 实际使用的提示词模板标识。
                - data (dict): 附加数据，包含原始模型响应内容。
        """
        user_profile: UserProfile | None = kwargs.get("user_profile")
        products: list[Product] = kwargs.get("products",[])

        # 若无商品信息，直接返回空结果
        if not products:
            return MarketingCopyResult(success=True,copies=[],confidence=1.0)
        
        # 根据用户画像选择提示词模板并获取系统提示语
        template_key = self._select_template(user_profile)
        system_prompt = PROMPT_TEMPLATES[template_key]

        # 格式化商品信息为字符串，供模型输入使用
        product_info = "\n".join([
            f"-ID:{product.product_id}名称:{product.name} 价格:{product.price} 类目:{product.category}"
            for product in products
        ])

        # 构建包含系统提示语和商品信息的消息列表
        messages = [
            SystemMessage(content=system_prompt + COPY_OUTPUT_INSTRUCTION),
            HumanMessage(content=f"商品信息：\n{product_info}"),
        ]
        # 异步调用大语言模型生成文案
        response = await self.llm.ainvoke(messages)

        # 解析模型响应并执行合规性检查
        copies = self._parse_copies(response.content)
        copies = [self._compliance_check(copy) for copy in copies]

        return MarketingCopyResult(
            success=True,
            copies=copies,
            prompt_template_used=template_key.value,
            data={"raw_response": response.content},
            confidence=0.9,
        )
    
    def _select_template(self, profile: UserProfile | None) -> UserSegment:
        """根据用户画像选择对应的用户细分类型以匹配模板。

        Args:
            profile: 用户画像对象，如果为 None 则视为无特定画像。

        Returns:
            匹配到的最高优先级的用户细分类型；若未匹配任何细分或无画像，则返回 ACTIVE。
        """
        if not profile:
            return UserSegment.ACTIVE

        # 定义用户细分类型的优先级顺序，从高到低进行匹配
        priority = [
            UserSegment.NEW_USER,
            UserSegment.HIGH_VALUE,
            UserSegment.PRICE_SENSITIVE,
            UserSegment.ACTIVE,
            UserSegment.CHURN_RISK,
        ]

        # 按优先级遍历，返回第一个存在于用户画像中的细分类型
        for seg in priority:
            if seg in profile.segments:
                return seg

        # 若未匹配到任何预设的细分类型，则默认返回 ACTIVE
        return UserSegment.ACTIVE
    
    def _parse_copies(self, raw: str) -> list[dict[str, str]]:
        """解析原始字符串中的副本信息。

        尝试从原始字符串中提取并解析 JSON 数据。如果字符串包含 Markdown 代码块标记，
        则先去除标记再解析。若解析失败或格式错误，则返回空列表。

        Args:
            raw: 包含 JSON 数据的原始字符串，可能包裹在 Markdown 代码块中。

        Returns:
            解析后的字典列表，每个字典代表一个副本信息；若解析失败则返回空列表。
        """
        try:
            # 去除首尾空白字符并处理可能的 Markdown 代码块包裹
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n",1)[1].rsplit("```",1)[0]
            return json.loads(cleaned)
        except(json.JSONDecodeError, IndexError):
            return []
        
    def _compliance_check(self, copy_item: dict[str, str]) -> dict[str, str]:
        """
        对文案内容进行合规性检查，替换其中的违禁词。

        Args:
            copy_item (dict[str, str]): 包含文案内容的字典，其中键 'copy' 对应待处理的文本。

        Returns:
            dict[str, str]: 处理后的字典，其中 'copy' 键对应的文本中的违禁词已被替换为 '***'。
        """
        text = copy_item.get("copy","")
        # 遍历违禁词列表，将文本中的违禁词替换为星号
        for word in FORBIDDEN_WORDS:
            text = re.sub(re.escape(word),"***",text)
        copy_item["copy"] = text
        return copy_item
