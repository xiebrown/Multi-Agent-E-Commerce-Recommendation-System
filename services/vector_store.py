"""
向量存储服务
- Milvus 向量检索：商品嵌入向量的相似性搜索
- 降级策略：Milvus 不可用时返回空结果，不影响主流程
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import structlog

logger = structlog.get_logger()


class VectorStore:
    """Milvus 向量存储封装，提供商品嵌入向量的相似性搜索能力。

    所有公开方法均包含连接状态检查，在 Milvus 不可用时会优雅降级
    （返回空列表 / False），不影响主业务流程。

    Attributes:
        host: Milvus 服务地址。
        port: Milvus 服务端口。
        collection_name: 集合名称。
        dimension: 嵌入向量维度。
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        collection_name: str = "product_embeddings",
        dimension: int = 768,
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dimension = dimension
        self._client: Any = None
        self._connected = False

    async def connect(self, timeout: float = 5.0) -> bool:
        """初始化 Milvus 连接，若集合不存在则自动创建。

        Args:
            timeout: 连接超时时间（秒），默认 5 秒。

        Returns:
            连接成功返回 True，否则返回 False。
        """
        try:
            from pymilvus import MilvusClient

            def _init() -> MilvusClient:
                client = MilvusClient(
                    host=self.host, port=self.port, timeout=min(timeout, 3)
                )
                collections = client.list_collections()
                if self.collection_name not in collections:
                    client.create_collection(
                        collection_name=self.collection_name,
                        dimension=self.dimension,
                        auto_id=False,
                    )
                    logger.info(
                        "vector_store.collection_created",
                        collection=self.collection_name,
                        dimension=self.dimension,
                    )
                return client

            loop = asyncio.get_event_loop()
            self._client = await asyncio.wait_for(
                loop.run_in_executor(None, _init), timeout=timeout
            )
            self._connected = True
            logger.info(
                "vector_store.connected",
                host=self.host,
                port=self.port,
                collection=self.collection_name,
            )
        except Exception as e:
            self._connected = False
            logger.warning(
                "vector_store.connect_failed",
                error=str(e),
                host=self.host,
                port=self.port,
            )
        return self._connected

    async def close(self) -> None:
        """关闭 Milvus 连接。"""
        if self._client:
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._client.close)
            except Exception as e:
                logger.warning("vector_store.close_error", error=str(e))
            finally:
                self._client = None
                self._connected = False
                logger.info("vector_store.closed")

    async def search_products(
        self, embedding: list[float], top_k: int = 20
    ) -> list[dict[str, Any]]:
        """执行向量相似性搜索，召回最相关的商品。

        Args:
            embedding: 查询向量（768 维浮点数列表）。
            top_k: 返回的最相似结果数量。

        Returns:
            匹配的商品字典列表，包含 product_id、distance 等字段。
            未连接时返回空列表。
        """
        if not self._connected or not self._client:
            return []

        try:
            loop = asyncio.get_event_loop()

            def _search() -> list[dict]:
                results = self._client.search(
                    collection_name=self.collection_name,
                    data=[embedding],
                    limit=top_k,
                    output_fields=["product_id", "category", "price", "brand"],
                )
                # 解析搜索结果
                parsed = []
                for hits in results:
                    for hit in hits:
                        parsed.append(
                            {
                                "product_id": hit.get("id"),
                                "distance": hit.get("distance", 0.0),
                                **hit.get("entity", {}),
                            }
                        )
                return parsed

            return await loop.run_in_executor(None, _search)
        except Exception as e:
            logger.warning("vector_store.search_error", error=str(e))
            return []

    async def insert_embedding(
        self,
        product_id: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """插入或更新商品嵌入向量。

        Args:
            product_id: 商品唯一标识。
            embedding: 768 维嵌入向量。
            metadata: 附加元数据（category, price, brand, tags 等）。

        Returns:
            操作成功返回 True，未连接或失败返回 False。
        """
        if not self._connected or not self._client:
            return False

        try:
            data = {
                "product_id": product_id,
                "embedding": embedding,
                **(metadata or {}),
            }
            loop = asyncio.get_event_loop()

            def _insert():
                self._client.insert(
                    collection_name=self.collection_name, data=[data]
                )

            await loop.run_in_executor(None, _insert)
            return True
        except Exception as e:
            logger.warning(
                "vector_store.insert_error",
                product_id=product_id,
                error=str(e),
            )
            return False

    async def delete_embedding(self, product_id: str) -> bool:
        """按 product_id 删除嵌入向量。

        Args:
            product_id: 要删除的商品 ID。

        Returns:
            操作成功返回 True，未连接或失败返回 False。
        """
        if not self._connected or not self._client:
            return False

        try:
            loop = asyncio.get_event_loop()

            def _delete():
                self._client.delete(
                    collection_name=self.collection_name,
                    filter=f'product_id == "{product_id}"',
                )

            await loop.run_in_executor(None, _delete)
            return True
        except Exception as e:
            logger.warning(
                "vector_store.delete_error",
                product_id=product_id,
                error=str(e),
            )
            return False

    @property
    def is_connected(self) -> bool:
        """是否已成功连接 Milvus。"""
        return self._connected
