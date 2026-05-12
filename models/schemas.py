from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class UserSegment(str, Enum):
    """
    用户细分枚举类。

    定义不同的用户群体类别，用于对用户进行分类和标记。
    该枚举继承自 str 和 Enum，确保枚举值既是字符串又是唯一的枚举成员。

    Attributes:
        NEW_USER (str): 新用户，指最近注册或首次使用的用户。
        ACTIVE (str): 活跃用户，指经常使用产品或服务的用户。
        HIGH_VALUE (str): 高价值用户，指消费金额高或对业务贡献大的用户。
        PRICE_SENSITIVE (str): 价格敏感型用户，指对价格变动较为敏感的用户。
        CHURN_RISK (str): 流失风险用户，指有较高可能性停止使用产品或服务的用户。
    """
    NEW_USER = "new_user"
    ACTIVE = "active"
    HIGH_VALUE = "high_value"
    PRICE_SENSITIVE = "price_sensitive"
    CHURN_RISK = "churn_risk"

class UserProfile(BaseModel):
    """
    用户画像数据模型，用于存储和管理用户的静态属性、行为偏好及实时标签。

    Attributes:
        user_id (str): 用户唯一标识符。
        age (int | None): 用户年龄，可选字段。
        gender (str | None): 用户性别，可选字段。
        city (str | None): 用户所在城市，可选字段。
        segments (list[UserSegment]): 用户所属的分群列表，默认为空列表。
        preferred_categories (list[str]): 用户偏好的商品类别列表，默认为空列表。
        price_range (tuple[float, float]): 用户消费价格区间，格式为(最小值, 最大值)。
        recent_views (list[str]): 用户最近浏览的商品或内容ID列表，默认为空列表。
        recent_purchases (list[str]): 用户最近购买的商品ID列表，默认为空列表。
        rfm_score (dict[str, float]): 用户的RFM模型评分字典，包含Recency, Frequency, Monetary等指标。
        real_time_tags (dict[str, Any]): 用户的实时动态标签字典，键为标签名，值为标签内容。
    """
    user_id: str
    age: int | None = None
    gender: str | None = None
    city: str | None = None
    segments: list[UserSegment] = Field(default_factory=list)
    preferred_categories: list[str] = Field(default_factory=list)
    price_range: tuple[float, float] = Field(default=(0.0, 10000.0))
    recent_views: list[str] = Field(default_factory=list)
    recent_purchases: list[str] = Field(default_factory=list)
    rfm_score: dict[str, float] = Field(default_factory=dict)
    real_time_tags: dict[str, Any] = Field(default_factory=dict)


class Product(BaseModel):
    """
    商品数据模型，用于定义商品的结构和属性。

    Attributes:
        product_id (str): 商品的唯一标识符。
        name (str): 商品名称。
        category (str): 商品所属类别。
        price (float): 商品价格。
        description (str): 商品描述，默认为空字符串。
        brand (str): 商品品牌，默认为空字符串。
        seller_id (str): 卖家ID，默认为空字符串。
        stock (int): 商品库存数量，默认为0。
        tags (list[str]): 商品标签列表，默认为空列表。
        score (float): 商品评分，默认为0.0。
        image_url (str): 商品图片URL，默认为空字符串。
    """
    product_id: str
    name: str
    category: str
    price: float
    description: str = ""
    brand: str = ""
    seller_id: str = ""
    stock: int = 0
    tags: list[str] = Field(default_factory=list)
    score: float = 0.0
    image_url: str = ""

class RecommendationRequest(BaseModel):
    """
    推荐请求数据模型。

    用于接收前端或外部服务发送的商品推荐请求。

    Attributes:
        user_id (str): 用户唯一标识（必填）。
        scene (str): 推荐场景，如 "homepage"（首页）、"detail"（详情页）、"cart"（购物车），默认 "homepage"。
        num_items (int): 需要返回的商品数量，默认 10。
        context (dict): 额外上下文信息，如 preferred_categories、recent_views、avg_order_amount 等。
    """
    user_id: str
    scene: str = "homepage"
    num_items: int = 10
    context: dict = Field(default_factory=dict)

    model_config = {"extra": "ignore"}

class AgentResult(BaseModel):
    """
    智能体执行结果的数据模型。

    该模型用于标准化智能体（Agent）执行任务后的返回结构，包含执行状态、性能指标及结果数据。

    Attributes:
        agent_name (str): 执行任务的智能体名称。
        success (bool): 执行是否成功，默认为 True。
        latency_ms (float): 执行耗时（毫秒），默认为 0.0。
        error (str | None): 错误信息描述，若执行成功则为 None，默认为 None。
        data (dict[str, Any]): 智能体返回的具体业务数据字典，默认为空字典。
        confidence (float): 结果的可信度或置信分数，范围通常在 0.0 到 1.0 之间，默认为 1.0。
    """
    agent_name: str
    success: bool = True
    latency_ms: float = 0.0
    error: str | None = None
    data:dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0

class UserProfileResult(AgentResult):
    """
    用户画像代理的执行结果类。

    该类继承自 AgentResult，用于封装用户画像代理（user_profile）执行后的返回数据。

    Attributes:
        agent_name (str): 代理名称，固定为 "user_profile"，标识此结果来源于用户画像代理。
        profile (UserProfile | None): 获取到的用户画像数据。如果代理执行失败或未获取到数据，则为 None。
    """
    agent_name: str = "user_profile"
    profile: UserProfile | None = None

class ProductRecResult(AgentResult):
    """
    商品推荐结果数据模型。

    该类用于封装商品推荐代理（Agent）的执行结果，包含代理名称、召回的商品列表以及所使用的召回策略。

    Attributes:
        agent_name (str): 代理名称，默认为 "product_rec"，标识此结果来源于商品推荐代理。
        products (list[Product]): 召回的商品列表，默认值为空列表。每个元素为 Product 对象，包含商品的详细信息。
        recall_strategy (str): 召回策略标识符，用于说明本次推荐所采用的具体算法或策略名称，默认为空字符串。
    """
    agent_name: str = "product_rec"
    products: list[Product] = Field(default_factory=list)
    recall_strategy: str = ""

class MarketingCopyResult(AgentResult):
    """
    营销文案生成结果的数据模型。

    该类用于封装营销文案代理（Agent）执行后的返回结果，
    包含生成的文案列表以及所使用的提示词模板信息。

    Attributes:
        agent_name (str): 代理名称，固定为 "marketing_copy"，标识此结果来源于营销文案生成代理。
        copies (list[dict[str, str]]): 生成的营销文案列表。每个元素是一个字典，
                                       通常包含文案的不同版本或相关元数据（如标题、正文等）。
                                       默认为空列表。
        prompt_template_used (str): 实际使用的提示词模板字符串。用于追溯生成结果所依据的 prompt 结构。
                                   默认为空字符串。
    """
    agent_name: str = "marketing_copy"
    copies: list[dict[str, str]] = Field(default_factory=list)
    prompt_template_used: str = ""

class InventoryResult(AgentResult):
    """
    库存代理的执行结果数据模型。

    该类用于封装库存查询或管理代理（Inventory Agent）执行后的返回结果，
    包含可用产品列表、低库存警报信息以及购买限制详情。

    Attributes:
        agent_name (str): 代理名称，固定为 "inventory"，用于标识结果来源。
        available_products (list[str]): 当前可用的产品 ID 列表，默认为空列表。
        low_stock_alerts (list[dict[str, Any]]): 低库存警报信息列表，每个元素为包含警报详情的字典，默认为空列表。
        purchase_limits (dict[str, int]): 购买限制字典，键为产品标识，值为最大允许购买数量，默认为空字典。
    """
    agent_name: str = "inventory"
    available_products: list[str] = Field(default_factory=list)
    low_stock_alerts: list[dict[str, Any]] = Field(default_factory=list)
    purchase_limits: dict[str, int] = Field(default_factory=dict)

class RecommendationResponse(BaseModel):
    """
    推荐系统响应数据模型。

    该模型定义了推荐接口返回的标准数据结构，包含请求追踪信息、
    推荐结果列表、营销文案、实验分组信息以及性能监控指标。

    Attributes:
        request_id (str): 唯一请求标识符，用于链路追踪和问题排查。
        user_id (str): 目标用户标识符。
        products (list[Product]): 推荐的商品列表，默认为空列表。
        marketing_copies (list[dict[str, str]]): 与推荐商品对应的营销文案列表，
            每个元素为包含文案键值对的字典，默认为空列表。
        experiment_group (str): 用户所属的实验分组名称，默认为 "control"（对照组）。
        agent_results (dict[str, Any]): 智能代理或中间处理步骤产生的附加结果数据，
            以字典形式存储，默认为空字典。
        total_latency_ms (float): 请求处理的总延迟时间（毫秒），默认为 0.0。
        timestamp (datetime): 响应生成的时间戳，默认使用当前时间。
    """
    request_id: str
    user_id: str
    products: list[Product] = Field(default_factory=list)
    marketing_copies: list[dict[str, str]] = Field(default_factory=list)
    experiment_group: str = "control"
    agent_results: dict[str, Any] = Field(default_factory=dict)
    total_latency_ms: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)