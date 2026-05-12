"""
Supervisor编排器 — 并行分发 + 聚合模式

                    ┌──────────────┐
                    │  Supervisor   │
                    └──────┬───────┘
           ┌───────┬───────┼───────┬────────┐
           ▼       ▼       ▼       ▼        │
      UserProfile  ProdRec  MktCopy  Inventory │
           │       │       │       │        │
           └───────┴───────┴───────┘        │
                    │                        │
                    ▼                        │
               Aggregator ◄─────────────────┘
                    │
                    ▼
              A/B Test Engine
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

import structlog

from agents import (
    InventoryAgent,
    MarketingCopyAgent,
    ProductRecAgent,
    UserProfileAgent,
)
from models.schemas import (
    Product,
    RecommendationRequest,
    RecommendationResponse,
    UserProfile,
    UserSegment,
)
from services.ab_test import ABTestEngine

logger = structlog.get_logger()


class SupervisorOrchestrator:
    """
    Supervisor 编排器，负责协调多个 Agent 进行并行推荐、库存校验及营销文案生成。

    采用两阶段并行策略：
    1. 第一阶段：并行获取用户画像和初步商品召回列表。
    2. 第二阶段：基于用户画像进行重排序，同时并行检查库存状态。
    3. 第三阶段：为最终选定的商品生成营销文案。
    """

    def __init__(
        self,
        ab_engine: ABTestEngine | None = None,
        vector_store: Any = None,
        mysql_db: Any = None,
    ):
        """初始化编排器及其依赖的 Agent 实例。

        Args:
            ab_engine: A/B 测试引擎实例。如果未提供，则创建默认实例。
            vector_store: 向量存储实例（Milvus），用于向量召回。
            mysql_db: MySQL 数据库实例，用于实时数据查询。
        """
        self.user_profile_agent = UserProfileAgent()
        self.product_rec_agent = ProductRecAgent()
        self.inventory_agent = InventoryAgent()
        self.marketing_copy_agent = MarketingCopyAgent()
        self.ab_engine = ab_engine or ABTestEngine()

        # 注入可选服务依赖
        self.product_rec_agent.vector_store = vector_store
        self.inventory_agent.db = mysql_db

    async def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        """
        执行完整的推荐流程，包括用户画像获取、商品召回、重排序、库存过滤及文案生成。

        Args:
            request: 推荐请求对象，包含用户ID、场景上下文、所需商品数量等信息。

        Returns:
            RecommendationResponse: 包含最终推荐商品列表、营销文案、实验分组信息及性能指标的响应对象。
        """
        request_id = str(uuid.uuid4())
        start = time.perf_counter()

        logger.info(
            "supervisor.start",
            request_id=request_id,
            user_id=request.user_id,
            scene=request.scene,
        )

        experiment = self.ab_engine.assign(request.user_id)

        # 第一阶段并行执行：获取用户画像与初步商品召回（数量为最终需求的2倍以预留重排序空间）
        profile_result, rec_result = await asyncio.gather(
            self.user_profile_agent.run(
                user_id=request.user_id,
                context=request.context,
            ),
            self.product_rec_agent.run(
                user_profile=None,
                num_items=request.num_items * 2,
            ),
        )

        user_profile: UserProfile | None = getattr(profile_result, "profile", None)
        # 画像 agent 失败时，从请求上下文构建基础画像，保证后续仍有一定个性化
        if user_profile is None:
            user_profile = UserProfile(
                user_id=request.user_id,
                preferred_categories=request.context.get("preferred_categories", []),
                segments=[UserSegment.ACTIVE],
            )
        raw_products: list[Product] = getattr(rec_result, "products", [])
        

        # 第二阶段并行执行：基于画像重排序商品，同时检查原始召回商品的库存状态
        rerank_task = self.product_rec_agent.run(
            candidates=raw_products,
            user_profile=user_profile,
            num_items=request.num_items,
        )
        inventory_task = self.inventory_agent.run(
            products=raw_products,
        )
        rerank_result, inventory_result = await asyncio.gather(
            rerank_task,
            inventory_task,
        )
        ranked_products: list[Product] = getattr(rerank_result, "products", raw_products)

        # 根据库存检查结果过滤商品，确保只返回有货商品；若无可用商品则降级使用重排序结果
        available_ids = set(getattr(inventory_result, "available_products", []))
        final_products = [p for p in ranked_products if p.product_id in available_ids]
        if not final_products:
            final_products = ranked_products[:request.num_items]
        final_products = final_products[:request.num_items]


        # 第三阶段串行执行：为最终确定的商品列表生成个性化营销文案
        copy_result = await self.marketing_copy_agent.run(
            user_profile=user_profile,
            products=final_products,
        )
        copies = getattr(copy_result, "copies", [])

        total_latency = (time.perf_counter() - start) * 1000

        logger.info(
            "supervisor.complete",
            request_id=request_id,
            total_latency_ms=round(total_latency, 1),
            product_count = len(final_products),
            copy_count = len(copies),
        )

        return RecommendationResponse(
            request_id=request_id,
            user_id=request.user_id,
            products=final_products,
            marketing_copies=copies,
            experiment_group=experiment.get("group","control"),
            agent_results={
                "user_profile": profile_result,
                "product_rec": rec_result,
                "marketing_copy": copy_result,
                "inventory": inventory_result,
            },
            total_latency_ms=total_latency,
        )