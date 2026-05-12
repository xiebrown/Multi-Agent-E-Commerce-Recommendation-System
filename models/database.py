"""
SQLAlchemy ORM 模型定义
- products: 商品主表
- user_profiles: 用户画像表
- user_behaviors: 用户行为日志表
- inventory_log: 库存变更日志表
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    JSON,
    String,
    Text,
    text,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class ProductORM(Base):
    __tablename__ = "products"

    product_id = Column(String(64), primary_key=True)
    name = Column(String(255), nullable=False)
    category = Column(String(64), default="")
    price = Column(Float, default=0.0)
    description = Column(Text, default="")
    brand = Column(String(128), default="")
    seller_id = Column(String(64), default="")
    stock = Column(Integer, default=0)
    tags = Column(JSON, default=list)
    score = Column(Float, default=0.0)
    image_url = Column(String(512), default="")
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(
        DateTime, default=datetime.now, onupdate=datetime.now
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "product_id": self.product_id,
            "name": self.name,
            "category": self.category,
            "price": self.price,
            "description": self.description,
            "brand": self.brand,
            "seller_id": self.seller_id,
            "stock": self.stock,
            "tags": self.tags if isinstance(self.tags, list) else json.loads(self.tags or "[]"),
            "score": self.score,
            "image_url": self.image_url,
        }


class UserProfileORM(Base):
    __tablename__ = "user_profiles"

    user_id = Column(String(64), primary_key=True)
    age = Column(Integer, nullable=True)
    gender = Column(String(16), default="")
    city = Column(String(64), default="")
    segments = Column(JSON, default=list)
    preferred_categories = Column(JSON, default=list)
    price_range_min = Column(Float, default=0.0)
    price_range_max = Column(Float, default=10000.0)
    rfm_score = Column(JSON, default=dict)
    real_time_tags = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(
        DateTime, default=datetime.now, onupdate=datetime.now
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "age": self.age,
            "gender": self.gender,
            "city": self.city,
            "segments": self.segments if isinstance(self.segments, list) else json.loads(self.segments or "[]"),
            "preferred_categories": self.preferred_categories if isinstance(self.preferred_categories, list) else json.loads(self.preferred_categories or "[]"),
            "price_range": (self.price_range_min, self.price_range_max),
            "rfm_score": self.rfm_score if isinstance(self.rfm_score, dict) else json.loads(self.rfm_score or "{}"),
            "real_time_tags": self.real_time_tags if isinstance(self.real_time_tags, dict) else json.loads(self.real_time_tags or "{}"),
        }


class UserBehaviorORM(Base):
    __tablename__ = "user_behaviors"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(String(64), nullable=False)
    behavior_type = Column(String(32), nullable=False)
    item_id = Column(String(64), nullable=False)
    extra_meta = Column(JSON, default=dict)
    timestamp_ms = Column(BigInteger, nullable=False)
    created_at = Column(DateTime, default=datetime.now)

    __table_args__ = (
        Index("idx_user_behavior", "user_id", "behavior_type", "timestamp_ms"),
    )


class InventoryLogORM(Base):
    __tablename__ = "inventory_log"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    product_id = Column(String(64), nullable=False)
    change_amount = Column(Integer, nullable=False)
    reason = Column(String(255), default="")
    created_at = Column(DateTime, default=datetime.now)

    __table_args__ = (
        Index("idx_inv_log_product", "product_id"),
    )
