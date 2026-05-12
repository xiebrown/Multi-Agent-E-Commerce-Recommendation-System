"""
库存决策Agent
- 实时库存查询：MCP协议同步WMS
- 库存预警：安全库存阈值 + 补货建议
- 限购策略：基于库存深度 + 促销热度动态调整
"""
from __future__ import annotations

from typing import Any

import structlog

from config import get_settings
from models.schemas import InventoryResult, Product

from .base_agent import BaseAgent

logger = structlog.get_logger()

SAFETY_STOCK_THRESHOLD = 50
LOW_STOCK_THRESHOLD = 100
HOT_ITEM_PURCHASE_LIMIT = 2

class InventoryAgent(BaseAgent):
    """
    库存代理类，用于处理与库存相关的业务逻辑。

    继承自 BaseAgent，初始化时配置代理名称和超时时间，
    并预留数据库连接属性以便后续扩展。
    """

    def __init__(self):
        """
        初始化 InventoryAgent 实例。

        从配置中获取设置，调用父类构造函数设置代理名称为 'inventory'
        并使用配置中的库存代理超时时间。同时初始化数据库连接属性为 None，
        留待后续实现数据库交互功能。

        参数:
            无显式参数，配置通过 get_settings() 内部获取。

        返回值:
            无返回值。
        """
        settings = get_settings()
        super().__init__(
            name="inventory",
            timeout=settings.agent_timeout_inventory,
        )
        # 初始化数据库连接占位符，后续将实现具体的数据库接入逻辑
        self.db = None

    async def _execute(self, **kwargs: Any) -> InventoryResult:
        """执行库存检查与评估逻辑。

        该函数异步处理传入的产品列表，检查每个产品的库存状态，
        识别可用产品、生成低库存警报，并计算购买限制。

        Args:
            **kwargs: 关键字参数，期望包含以下键：
                - products (list[Product]): 待检查的产品对象列表。若未提供，则默认为空列表。

        Returns:
            InventoryResult: 包含以下信息的库存结果对象：
                - success (bool): 执行是否成功。
                - available_products (list[str]): 有库存可用的产品ID列表。
                - low_stock_alerts (list[dict]): 低库存警报详情列表，包含产品ID、名称、当前库存、警报级别和建议操作。
                - purchase_limits (dict[str, int]): 各产品的购买限制映射，键为产品ID，值为限制数量。
                - data (dict): 统计摘要，包括检查总数、可用数量和警报数量。
                - confidence (float): 结果置信度，固定为0.95。
        """
        products: list[Product] = kwargs.get("products",[])

        available = []
        low_stock_alerts = []
        purchase_limits: dict[str, int] = {}

        # 遍历所有产品，检查库存状态并分类处理
        for product in products:
            stock = await self._check_stock(product.product_id, product.stock)
            if stock <= 0:
                continue

            available.append(product.product_id)

            # 根据库存阈值生成相应级别的低库存警报
            if stock <= SAFETY_STOCK_THRESHOLD:
                low_stock_alerts.append({
                    "product_id": product.product_id,
                    "name": product.name,
                    "current_stock": stock,
                    "level":"critical",
                    "action": "urgent_restock"
                    })
            elif stock <= LOW_STOCK_THRESHOLD:
                low_stock_alerts.append({
                    "product_id": product.product_id,
                    "name": product.name,
                    "current_stock": stock,
                    "level":"warning",
                    "action": "plan_restock"
                    })

            # 计算并记录产品的购买限制
            limit = self._calc_purchase_limit(product, product.stock)
            if limit is not None:
                purchase_limits[product.product_id] = limit

        return InventoryResult(
            success=True,
            available_products=available,
            low_stock_alerts=low_stock_alerts,
            purchase_limits=purchase_limits,
            data={
                "total_checked": len(products),
                "available_count": len(available),
                "alert_count": len(low_stock_alerts),
            },
            confidence=0.95,
        )
    
    async def _check_stock(self, product_id: str, fallback_stock: int) -> int:
        """检查指定产品的实时库存数量。

        优先从 MySQL 数据库查询；不可用时返回默认库存值作为降级策略。

        Args:
            product_id: 需要查询库存的产品唯一标识符。
            fallback_stock: 降级时使用的默认库存值。

        Returns:
            实时库存数量；降级时返回 fallback_stock。
        """
        if self.db and self.db.is_connected:
            try:
                db_stock = await self.db.get_stock(product_id)
                if db_stock is not None:
                    return db_stock
            except Exception as e:
                logger.warning(
                    "inventory.stock_query_failed",
                    product_id=product_id,
                    error=str(e),
                )
        return fallback_stock
    
    def _calc_purchase_limit(self, product: Product, stock: int) -> int | None:
        """
        计算商品的购买限制数量。

        根据商品标签（是否为新品或旗舰）以及当前库存水平，
        动态确定单个用户的最大购买数量。

        Args:
            product (Product): 商品对象，用于检查是否包含特定标签。
            stock (int): 当前商品库存数量。

        Returns:
            int | None: 返回限制购买的数量；如果无特殊限制，则返回 None。
        """
        # 判断商品是否为热门商品（新品或旗舰）
        is_hot = "新品" in product.tags or "旗舰" in product.tags

        # 库存低于安全阈值时，严格限制每人仅可购买1件
        if stock <= SAFETY_STOCK_THRESHOLD:
            return 1

        # 热门商品在低库存状态下，应用特定的热门商品购买限额
        if stock <= LOW_STOCK_THRESHOLD and is_hot:
            return HOT_ITEM_PURCHASE_LIMIT

        # 热门商品库存较低但未达到低库存阈值时，限制每人购买3件
        if is_hot and stock <= 300:
            return 3

        # 其他情况无特殊购买限制
        return None