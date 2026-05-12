"""
MySQL 数据库服务
- 基于 SQLAlchemy 异步引擎（async_engine + aiomysql）
- 提供商品、用户、库存等核心数据的持久化操作
- 降级策略：数据库不可用时返回默认值，不影响主流程
"""

from __future__ import annotations

from typing import Any

import structlog
from sqlalchemy import select, text, update
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from models.database import (
    Base,
    InventoryLogORM,
    ProductORM,
    UserBehaviorORM,
    UserProfileORM,
)

logger = structlog.get_logger()


class MySQLDatabase:
    """异步 MySQL 数据库访问服务。

    所有公开方法均包含连接状态检查，在数据库不可用时会优雅降级
    （返回 None / [] / False），不影响主业务流程。

    Attributes:
        database_url: MySQL 连接 URL（格式：mysql+aiomysql://user:pass@host:port/db）。
    """

    def __init__(self, database_url: str | None = None):
        self.database_url = database_url
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker | None = None
        self._connected = False

    async def connect(self) -> bool:
        """创建异步引擎连接池并验证连通性。

        若 database_url 为空或连接失败，将 _connected 置为 False
        并记录警告日志，不影响应用启动。

        Returns:
            连接成功返回 True，否则返回 False。
        """
        if not self.database_url:
            logger.info("mysql_database.no_url_configured")
            self._connected = False
            return False

        try:
            self._engine = create_async_engine(
                self.database_url,
                echo=False,
                pool_size=5,
                max_overflow=10,
            )
            self._session_factory = async_sessionmaker(
                self._engine, expire_on_commit=False
            )

            # 验证连接
            async with self._engine.connect() as conn:
                await conn.execute(text("SELECT 1"))

            self._connected = True
            logger.info("mysql_database.connected", url=self.database_url)
        except Exception as e:
            self._connected = False
            logger.warning(
                "mysql_database.connect_failed",
                error=str(e),
                url=self.database_url,
            )

        return self._connected

    async def close(self) -> None:
        """关闭数据库连接池。"""
        if self._engine:
            try:
                await self._engine.dispose()
            except Exception as e:
                logger.warning("mysql_database.close_error", error=str(e))
            finally:
                self._engine = None
                self._session_factory = None
                self._connected = False
                logger.info("mysql_database.closed")

    async def init_tables(self) -> None:
        """创建所有 ORM 映射的表（如果尚不存在）。

        仅在连接成功时执行，失败时仅记录警告。
        """
        if not self._connected or not self._engine:
            logger.warning("mysql_database.init_tables_skipped_not_connected")
            return

        try:
            async with self._engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("mysql_database.tables_initialized")
        except Exception as e:
            logger.warning(
                "mysql_database.init_tables_failed", error=str(e)
            )

    async def get_stock(self, product_id: str) -> int | None:
        """查询指定商品的当前库存量。

        Args:
            product_id: 商品唯一标识。

        Returns:
            库存数量。未连接或查询失败时返回 None。
        """
        if not self._connected or not self._session_factory:
            return None

        try:
            async with self._session_factory() as session:
                result = await session.execute(
                    select(ProductORM.stock).where(
                        ProductORM.product_id == product_id
                    )
                )
                row = result.scalar_one_or_none()
                return row if row is not None else None
        except Exception as e:
            logger.warning(
                "mysql_database.get_stock_error",
                product_id=product_id,
                error=str(e),
            )
            return None

    async def get_products_by_ids(
        self, product_ids: list[str]
    ) -> list[dict[str, Any]]:
        """批量查询商品信息。

        Args:
            product_ids: 商品 ID 列表。

        Returns:
            商品字典列表。未连接或查询失败时返回空列表。
        """
        if not self._connected or not self._session_factory:
            return []

        try:
            async with self._session_factory() as session:
                result = await session.execute(
                    select(ProductORM).where(
                        ProductORM.product_id.in_(product_ids)
                    )
                )
                rows = result.scalars().all()
                return [row.to_dict() for row in rows]
        except Exception as e:
            logger.warning(
                "mysql_database.get_products_error", error=str(e)
            )
            return []

    async def get_user_profile(
        self, user_id: str
    ) -> dict[str, Any] | None:
        """查询用户画像数据。

        Args:
            user_id: 用户唯一标识。

        Returns:
            用户画像字典，未找到或未连接时返回 None。
        """
        if not self._connected or not self._session_factory:
            return None

        try:
            async with self._session_factory() as session:
                result = await session.execute(
                    select(UserProfileORM).where(
                        UserProfileORM.user_id == user_id
                    )
                )
                row = result.scalar_one_or_none()
                return row.to_dict() if row else None
        except Exception as e:
            logger.warning(
                "mysql_database.get_user_profile_error",
                user_id=user_id,
                error=str(e),
            )
            return None

    async def update_stock(
        self, product_id: str, quantity: int
    ) -> bool:
        """更新商品库存数量。

        Args:
            product_id: 商品唯一标识。
            quantity: 新的库存数量。

        Returns:
            操作成功返回 True，否则返回 False。
        """
        if not self._connected or not self._session_factory:
            return False

        try:
            async with self._session_factory() as session:
                await session.execute(
                    update(ProductORM)
                    .where(ProductORM.product_id == product_id)
                    .values(stock=quantity)
                )
                await session.commit()
            return True
        except Exception as e:
            logger.warning(
                "mysql_database.update_stock_error",
                product_id=product_id,
                error=str(e),
            )
            return False

    async def record_behavior(
        self,
        user_id: str,
        behavior_type: str,
        item_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """记录用户行为到 user_behaviors 表。

        Args:
            user_id: 用户唯一标识。
            behavior_type: 行为类型（view / click / purchase 等）。
            item_id: 行为关联的商品 ID。
            metadata: 附加元数据。

        Returns:
            操作成功返回 True，否则返回 False。
        """
        if not self._connected or not self._session_factory:
            return False

        try:
            import time

            entry = UserBehaviorORM(
                user_id=user_id,
                behavior_type=behavior_type,
                item_id=item_id,
                extra_meta=metadata or {},
                timestamp_ms=int(time.time() * 1000),
            )
            async with self._session_factory() as session:
                session.add(entry)
                await session.commit()
            return True
        except Exception as e:
            logger.warning(
                "mysql_database.record_behavior_error",
                user_id=user_id,
                behavior_type=behavior_type,
                error=str(e),
            )
            return False

    @property
    def is_connected(self) -> bool:
        """是否已成功连接数据库。"""
        return self._connected
