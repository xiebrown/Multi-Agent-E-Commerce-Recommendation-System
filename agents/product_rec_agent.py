"""
商品推荐Agent
- 召回层：协同过滤 + 向量检索(Milvus) + 热度/新品策略
- 排序层：LLM重排 + 特征交叉(用户画像 x 商品属性)
- 多样性控制：类目打散、卖家去重、新品加权
"""

from __future__ import annotations

import json
import random
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import get_settings
from models.schemas import AgentResult, Product, ProductRecResult, UserProfile

from .base_agent import BaseAgent

RERANK_PROMPT = """你是电商推荐排序专家。根据用户画像和候选商品,重新排序并选出最优的{num_items}个商品。

用户画像:
{user_profile}

候选商品:
{candidates}

排序原则:
1. 用户偏好类目优先
2. 价格在用户可接受范围内
3. 保证类目多样性(相邻商品尽量不同类目)
4. 新品适当加权

请输出商品ID列表(JSON数组),按推荐优先级排序:
["product_id_1", "product_id_2", ...]

只输出JSON数组,不要其他内容。"""

#后续通过调用搜索引擎改进
MOCK_PRODUCTS = [
    Product(product_id="P001", name="iPhone 16 Pro", category="手机", price=7999, brand="Apple", seller_id="S01", stock=500, tags=["旗舰", "新品"]),
    Product(product_id="P002", name="华为 Mate 70", category="手机", price=5999, brand="华为", seller_id="S02", stock=300, tags=["旗舰", "国产"]),
    Product(product_id="P003", name="AirPods Pro 3", category="耳机", price=1899, brand="Apple", seller_id="S01", stock=1000, tags=["降噪", "无线"]),
    Product(product_id="P004", name="Sony WH-1000XM6", category="耳机", price=2499, brand="Sony", seller_id="S03", stock=200, tags=["头戴", "降噪"]),
    Product(product_id="P005", name="iPad Air M3", category="平板", price=4799, brand="Apple", seller_id="S01", stock=400, tags=["学习", "办公"]),
    Product(product_id="P006", name="小米平板7 Pro", category="平板", price=2499, brand="小米", seller_id="S04", stock=600, tags=["性价比", "娱乐"]),
    Product(product_id="P007", name="Anker 140W充电器", category="配件", price=399, brand="Anker", seller_id="S05", stock=2000, tags=["快充", "便携"]),
    Product(product_id="P008", name="机械革命极光X", category="笔记本", price=6999, brand="机械革命", seller_id="S06", stock=150, tags=["游戏", "高性能"]),
    Product(product_id="P009", name="戴尔U2724D显示器", category="显示器", price=3299, brand="Dell", seller_id="S07", stock=80, tags=["4K", "办公"]),
    Product(product_id="P010", name="罗技MX Master 3S", category="配件", price=749, brand="罗技", seller_id="S08", stock=500, tags=["无线", "办公"]),
    Product(product_id="P011", name="三星980 Pro 2TB", category="存储", price=1199, brand="三星", seller_id="S09", stock=300, tags=["SSD", "高速"]),
    Product(product_id="P012", name="绿联氮化镓65W", category="配件", price=129, brand="绿联", seller_id="S10", stock=5000, tags=["快充", "性价比"]),
    Product(product_id="P013", name="Apple Watch Ultra 3", category="穿戴", price=5999, brand="Apple", seller_id="S01", stock=200, tags=["运动", "健康"]),
    Product(product_id="P014", name="大疆Mini 4 Pro", category="无人机", price=4788, brand="大疆", seller_id="S11", stock=100, tags=["航拍", "便携"]),
    Product(product_id="P015", name="Switch 2", category="游戏机", price=2499, brand="Nintendo", seller_id="S12", stock=50, tags=["新品", "游戏"]),
]



class ProductRecAgent(BaseAgent):
    """
    商品推荐代理类，负责执行商品推荐逻辑。

    该类继承自 BaseAgent，并实现了 _execute 方法，用于执行商品推荐逻辑。
    """
    def __init__(self):
        """
        初始化产品推荐代理实例。

        该构造函数负责加载系统配置，初始化父类基础属性，
        并配置用于产品推荐的大语言模型（LLM）客户端及向量存储占位符。

        Args:
            self: 类实例本身，无需手动传递。

        Returns:
            None: 构造函数无返回值。
        """
        # 获取全局配置 settings
        settings = get_settings()
        
        # 调用父类构造函数，设置代理名称和超时时间
        super().__init__(
            name="product_rec",
            timeout=settings.agent_timeout_product_rec,
        )
        
        # 初始化 OpenAI 聊天模型实例，配置 API 密钥、基础 URL、模型名称及生成参数
        self.llm = ChatOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            temperature=0.3,
            max_tokens=512,
        )
        
        # 初始化向量存储引用，暂设为 None，后续可根据需要赋值
        self.vector_store: Any = None
    async def _execute(self, **kwargs: Any) -> ProductRecResult:
        """
        执行商品推荐的核心逻辑，包括召回、重排序及结果组装。

        Args:
            **kwargs: 关键字参数字典，包含以下关键参数：
                - user_profile (UserProfile | None): 用户画像信息，用于个性化推荐。
                - num_items (int): 需要返回的商品数量，默认为 10。
                - candidates (list[Product] | None): 外部传入的候选商品列表（跳过召回阶段）。

        Returns:
            ProductRecResult: 包含推荐结果的对象，具体字段如下：
                - success (bool): 执行是否成功。
                - products (list): 最终推荐的商品列表。
                - recall_strategy (str): 使用的召回策略描述。
                - data (dict): 包含候选集数量和重排序数量的统计信息。
                - confidence (float): 推荐结果的置信度。
        """
        user_profile: UserProfile | None = kwargs.get("user_profile")
        num_items: int = kwargs.get("num_items", 10)
        candidates: list[Product] | None = kwargs.get("candidates")

        # 执行两阶段推荐流程：首先扩大范围进行候选集召回，然后对候选集进行精排
        if candidates is None:
            candidates = await self._recall(user_profile, num_items*3)
        ranked_ids = await self._rerank(user_profile, candidates, num_items)

        # 构建商品ID到商品对象的映射，并根据重排序后的ID提取最终商品列表
        id_to_product = {product.product_id: product for product in candidates}
        final_product = []
        for product_id in ranked_ids:
            if product_id in id_to_product:
                final_product.append(id_to_product[product_id])

        # 若重排序后的商品数量不足，从剩余候选集中补充商品直到满足数量要求
        if len(final_product) < num_items:
            for product in candidates:
                if product.product_id not in ranked_ids:
                    final_product.append(product)
                    if len(final_product) >= num_items:
                        break

        return ProductRecResult(
            success=True,
            products=final_product[:num_items],
            recall_strategy="协同过滤 + 向量检索 + 热度/新品策略",
            data={"candidate_count": len(candidates), "ranked": len(ranked_ids)},
            confidence=0.8,
            )
    
    async def _recall(self, profile: UserProfile | None, limit: int) -> list[Product]:
        """根据用户画像召回候选商品列表。

        优先使用向量存储进行语义检索，若不可用则降级到 MOCK_PRODUCTS。
        Args:
            profile: 用户画像信息，可能为 None。
            limit: 返回商品数量的上限。

        Returns:
            候选商品列表，长度不超过 limit。
        """
        candidates: list[Product] = []

        # 优先通过向量存储召回
        if self.vector_store and self.vector_store.is_connected:
            embedding = await self._get_user_embedding(profile)
            if embedding:
                vector_results = await self.vector_store.search_products(
                    embedding=embedding, top_k=limit
                )
                if vector_results:
                    candidates = await self._resolve_vector_results(vector_results)

        # 降级到 MOCK_PRODUCTS
        if not candidates:
            candidates = list(MOCK_PRODUCTS)
            if profile and profile.preferred_categories:
                preferred = set(profile.preferred_categories)
                candidates.sort(
                    key=lambda p: (p.category in preferred, p.stock > 0, random.random()),
                    reverse=True,
                )

        return candidates[:limit]

    async def _get_user_embedding(self, profile: UserProfile | None) -> list[float] | None:
        """生成用户偏好嵌入向量。

        实际部署中应调用嵌入模型将用户画像转为向量。
        当前返回 None 以触发 MOCK_PRODUCTS 降级（向量检索由外部集成方负责）。

        Returns:
            768 维浮点向量，或 None（表示跳过向量检索）。
        """
        # TODO: 接入嵌入模型，将 profile 转换为向量
        return None

    async def _resolve_vector_results(self, results: list[dict]) -> list[Product]:
        """将向量搜索结果解析为 Product 对象列表。

        优先通过 MySQLDatabase 批量查询商品详情，降级到从 MOCK_PRODUCTS 匹配。

        Args:
            results: 向量搜索结果列表，包含 product_id 等字段。

        Returns:
            商品对象列表。
        """
        product_ids = [r.get("product_id", "") for r in results if r.get("product_id")]

        # 若 MySQL DB 可用，从数据库查询
        if product_ids and hasattr(self, "db") and self.db and self.db.is_connected:
            try:
                db_products = await self.db.get_products_by_ids(product_ids)
                if db_products:
                    return [Product(**p) for p in db_products]
            except Exception:
                pass

        # 降级：从 MOCK_PRODUCTS 中按 product_id 匹配
        id_map = {p.product_id: p for p in MOCK_PRODUCTS}
        matched = []
        for pid in product_ids:
            if pid in id_map:
                matched.append(id_map[pid])
        return matched
        
    async def _rerank(self, profile: UserProfile | None, candidates: list[Product], num_items: int) -> list[str]:
        """根据用户画像对候选商品列表进行重排序。

        优先调用 LLM 进行智能排序；若 LLM 不可用或解析失败，
        则降级使用基于用户画像的规则排序（偏好类目 > 价格适配 > 新品加权）。

        Args:
            profile: 用户画像对象。若为 None，则不使用个性化排序。
            candidates: 待排序的商品对象列表。
            num_items: 需要返回的重排序后的商品数量。

        Returns:
            重排序后的商品 ID 列表。
        """
        if not profile:
            return [product.product_id for product in candidates[:num_items]]

        # 尝试 LLM 智能排序
        try:
            profile_summary = {
                "segments": [s.value for s in profile.segments],
                "preferred_categories": profile.preferred_categories,
                "price_range": list(profile.price_range),
            }
            candidate_summary = [
                {"id": p.product_id, "name": p.name, "category": p.category, "price": p.price, "tags": p.tags}
                for p in candidates
            ]
            prompt = RERANK_PROMPT.format(
                num_items=num_items,
                user_profile=json.dumps(profile_summary, ensure_ascii=False),
                candidates=json.dumps(candidate_summary, ensure_ascii=False),
            )
            message = [
                SystemMessage(content="你是电商推荐排序专家。"),
                HumanMessage(content=prompt),
            ]
            response = await self.llm.ainvoke(message)

            raw = response.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            ranked_ids = json.loads(raw)
            if isinstance(ranked_ids, list) and len(ranked_ids) > 0:
                return ranked_ids[:num_items]
        except Exception:
            pass

        # 降级：基于用户画像的规则排序
        return self._basic_rerank(profile, candidates, num_items)

    def _basic_rerank(self, profile: UserProfile, candidates: list[Product], num_items: int) -> list[str]:
        """基于用户画像进行规则排序，作为 LLM 排序的降级方案。

        排序权重：
        - 偏好类目匹配：+100
        - 价格在用户接受范围内：+10
        - 新品标记：+5
        """
        preferred = set(profile.preferred_categories)
        price_min, price_max = profile.price_range

        def _score(p: Product) -> int:
            s = 0
            if p.category in preferred:
                s += 100
            if price_min <= p.price <= price_max:
                s += 10
            if "新品" in p.tags:
                s += 5
            return s

        sorted_products = sorted(candidates, key=_score, reverse=True)
        return [p.product_id for p in sorted_products[:num_items]]