"""
实时特征存储服务
- Redis Sorted Set 存储用户行为序列 (score=timestamp)
- 滑动窗口计算实时特征 (1h/24h/7d)
- 离线+在线特征合并
- RFM模型计算
"""

from __future__ import annotations

import json
import time
from typing import Any

import structlog

logger = structlog.get_logger()


class FeatureStore:
    def __init__(self, redis_client: Any = None, ttl: int = 86400):
        """
        初始化特征存储实例。

        Args:
            redis_client (Any, optional): Redis 客户端实例。如果为 None，则相关操作将不执行。
            ttl (int, optional): 键的过期时间（秒），默认为 86400 秒（24小时）。
        """
        self.redis = redis_client
        self.ttl = ttl

    async def record_behavior(
            self, user_id: str, behavior_type: str, item_id: str, metadata: dict | None = None
    ):
        """
        异步记录用户行为数据到 Redis 有序集合中。

        该方法将用户的行为信息序列化为 JSON 字符串，并作为成员添加到以用户ID和行为类型构成的键对应的有序集合中，
        分数为当前时间戳。同时设置键的过期时间。

        Args:
            user_id (str): 用户唯一标识符。
            behavior_type (str): 行为类型（例如：click, view, purchase 等）。
            item_id (str): 物品唯一标识符。
            metadata (dict | None, optional): 额外的元数据字典，默认为 None。如果提供，其内容将被合并到 payload 中。

        Returns:
            None: 如果 redis 客户端未初始化，则直接返回；否则无显式返回值。
        """
        if not self.redis:
            return

        # 构建 Redis 键，格式为 behavior:{user_id}:{behavior_type}
        key = f"behavior:{user_id}:{behavior_type}"

        # 序列化行为数据，包含 item_id、时间戳以及可选的元数据
        payload = json.dumps({"item_id": item_id, "ts": time.time(), **(metadata or {})})

        # 将 payload 作为成员，当前时间戳作为分数，添加到有序集合中
        await self.redis.zadd(key, {payload: time.time()})

        # 设置键的过期时间
        await self.redis.expire(key, self.ttl)

    async def get_recent_behaviors(
            self, user_id: str, behavior_type: str, window_second: int = 3600
    ) -> list[dict]:
        """
        获取指定用户在最近时间窗口内的行为记录。

        Args:
            user_id (str): 用户唯一标识 ID。
            behavior_type (str): 行为类型，用于区分不同的行为数据集合。
            window_second (int): 时间窗口大小（秒），默认为 3600 秒（1小时）。

        Returns:
            list[dict]: 包含最近行为记录的字典列表。如果 Redis 不可用或无数据，则返回空列表。
        """
        if not self.redis:
            return []
        key = f"behavior:{user_id}:{behavior_type}"
        cutoff = time.time() - window_second
        # 从 Redis 有序集合中查询指定时间范围内的行为数据
        raw_items = await self.redis.zrangebyscore(key, cutoff, "+inf")
        # 将序列化的 JSON 字符串解析为字典对象并返回
        return [json.loads(item) for item in raw_items]
    
    async def get_user_features(self, user_id: str) -> dict[str, Any]:
        """
        获取指定用户的特征数据。

        通过查询用户在不同时间窗口内的行为记录（浏览、点击、购买），
        提取最近浏览和购买的物品ID列表作为用户特征。

        Args:
            user_id (str): 用户的唯一标识符。

        Returns:
            dict[str, Any]: 包含用户行为特征的字典，主要包含最近浏览和购买的物品ID列表。
        """
        # 获取不同时间窗口和行为类型的用户行为记录
        views_1h = await self.get_recent_behaviors(user_id, "view", 3600)
        views_24h = await self.get_recent_behaviors(user_id, "view", 86400)
        clicks_1h = await self.get_recent_behaviors(user_id, "click", 3600)
        purchases_7d = await self.get_recent_behaviors(user_id, "purchase", 604800)

        # 提取最近24小时内浏览的最后20个物品ID
        recent_view_items = [v.get("item_id","") for v in views_24h[-20:]]
        # 提取最近7天内购买的最后10个物品ID
        recent_purchase_items = [p.get("item_id","") for p in purchases_7d[-10:]]

        # 计算用户的RFM特征
        rfm = await self._compute_rfm(user_id,purchases_7d)

        # 从Redis中获取用户的离线标签信息
        profile_key = f"profile:{user_id}"
        offline_tags = {}
        if self.redis:
            raw = await self.redis.get(profile_key)
            if raw:
                offline_tags = json.loads(raw)
        
        return{
            "user_id": user_id,
            "view_count_1h": len(views_1h),
            "view_count_24h": len(views_24h),
            "click_count_1h": len(clicks_1h),
            "purchase_count_7d": len(purchases_7d),
            "recent_views": recent_view_items,
            "recent_purchases": recent_purchase_items,
            "rfm": rfm,
            "offline_tags": offline_tags,
        }
    
    async def _compute_rfm(self, user_id: str, purchases: list[dict]) -> dict[str, float]:
        """
        计算用户的 RFM（最近一次消费、消费频率、消费金额）评分，结果归一化到 0-1 范围。
        在缺乏完整历史数据的情况下，使用启发式规则进行估算。

        Args:
            user_id (str): 用户唯一标识符。
            purchases (list[dict]): 用户的购买记录列表，每个字典应包含 'ts'（时间戳）和 'amount'（金额）字段。

        Returns:
            dict[str, float]: 包含 'recency'、'frequency' 和 'monetary' 三个键的字典，对应的值为 0 到 1 之间的浮点数评分。
        """
        if not purchases:
            return {"recency": 0.0, "frequency": 0.0, "monetary": 0.0}

        now = time.time()
        latest_ts = max(p.get("ts",0) for p in purchases)
        days_since = (now - latest_ts) / 86400

        # 计算最近一次消费评分：距离当前时间越近，分数越高（以30天为基准线性衰减）
        recency = max(0.0, 1.0 - days_since / 30.0)
        
        # 计算消费频率评分：基于购买次数，以10次为满分基准进行归一化
        frequency = min(1.0, len(purchases) / 10.0)
        
        avg_amount = sum(p.get("amount", 100) for p in purchases) / len(purchases)
        
        # 计算消费金额评分：基于平均订单金额，以1000元为满分基准进行归一化
        monetary = min(1.0, avg_amount / 1000.0)

        return {
            "recency": round(recency, 3),
            "frequency": round(frequency, 3),
            "monetary": round(monetary, 3),
        }
    
    async def merge_offline_tags(self, user_id: str, tags: dict[str, Any]):
        """
        将离线标签数据合并并存储到 Redis 中。

        Args:
            user_id (str): 用户的唯一标识符，用于构建 Redis 键。
            tags (dict[str, Any]): 需要存储的标签数据字典。

        Returns:
            None: 如果 Redis 未初始化则直接返回，否则无返回值。
        """
        # 如果 Redis 实例未初始化，则直接退出
        if not self.redis: 
            return
        # 构建用户个人资料的 Redis 键
        key = f"profile:{user_id}"
        # 将标签数据序列化为 JSON 并存入 Redis，设置过期时间
        await self.redis.set(key, json.dumps(tags), ex=self.ttl)