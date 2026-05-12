"""
用于多智能体推荐流水线的 LangGraph 状态图。

可视化智能体执行的有向无环图 (DAG)：

  [开始] -> 扇出(fan_out) -> {用户画像(user_profile), 商品召回(product_recall)}  (并行)
          -> 合并阶段1(merge_phase1) -> {重排序(rerank), 库存检查(inventory)}   (并行)
          -> 合并阶段2(merge_phase2) -> 营销文案生成(marketing_copy)
          -> 聚合(aggregate) -> [结束]
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from agents import (
    InventoryAgent,
    MarketingCopyAgent,
    ProductRecAgent,
    UserProfileAgent,
)
from models.schemas import Product, UserProfile
from services.ab_test import ABTestEngine


class PipelineState(TypedDict, total=False):
    """
    定义推荐 pipeline 中各阶段的状态数据结构。

    该 TypedDict 用于在 pipeline 的不同处理步骤之间传递上下文和中间结果。
    由于 total=False，所有字段均为可选，允许在不同阶段逐步填充数据。

    Attributes:
        request_id (str): 唯一请求标识符，用于追踪和日志记录。
        user_id (str): 当前请求关联的用户标识符。
        scene (str): 业务场景标识，用于区分不同的推荐场景（如首页、详情页等）。
        num_items (int): 期望返回的推荐物品数量。
        context (dict[str, Any]): 额外的上下文信息，可能包含设备信息、地理位置等。
        experiment_group (str): A/B 测试实验组标识，用于分流和效果评估。

        user_profile (UserProfile | None): 用户画像数据，包含用户兴趣、历史行为等结构化信息。
        raw_products (list[Product]): 从召回阶段获取的原始候选物品列表。
        ranked_products (list[Product]): 经过排序模型打分后的物品列表。
        available_ids (set[str]): 当前可用的物品 ID 集合，用于过滤无效或下架物品。
        final_products (list[Product]): 最终推荐给用户的物品列表，经过重排、去重等后处理。
        marketing_copies (list[dict[str, str]]): 为最终推荐物品生成的营销文案列表。

        agent_results (dict[str, Any]): 各 Agent 执行结果的汇总，便于调试和监控。
        total_latency_ms (float): 整个 pipeline 执行的总耗时（毫秒）。
        _start_time (float): pipeline 开始执行的时间戳，用于计算耗时。
    """
    request_id: str
    user_id : str
    scene: str
    num_items: int
    context: dict[str, Any]
    experiment_group: str

    user_profile: UserProfile | None
    raw_products: list[Product]
    ranked_products: list[Product]
    available_ids: set[str]
    final_products: list[Product]
    marketing_copies: list[dict[str, str]]

    agent_results: dict[str, Any]
    total_latency_ms: float
    _start_time: float


# 初始化推荐 pipeline 所需的核心组件实例
user_profile_agent = UserProfileAgent()
product_rec_agent = ProductRecAgent()
inventory_agent = InventoryAgent()
marketing_copy_agent = MarketingCopyAgent()
ab_engine = ABTestEngine()

async def init_node(state: PipelineState) -> PipelineState:
    """初始化管道节点状态。

    该函数负责为当前的处理流程生成唯一的请求标识，记录起始时间以用于后续的性能监控，
    并初始化代理结果的存储容器。此外，它还会根据用户ID分配实验组别，以支持A/B测试逻辑。

    Args:
        state (PipelineState): 当前的管道状态字典，包含用户ID等上下文信息。

    Returns:
        PipelineState: 更新后的管道状态字典，包含了新生成的请求ID、起始时间、
                       空的代理结果字典以及分配的实验组别。
    """
    state["request_id"] = str(uuid.uuid4())
    state["_start_time"] = time.perf_counter()
    state["agent_results"] = {}

    # 根据用户ID分配A/B测试实验组
    exp = ab_engine.assign(state["user_id"])
    state["experiment_group"] = exp.get("group","control")
    return state

async def user_profile_node(state: PipelineState) -> PipelineState:
    """
    异步执行用户画像代理节点，获取并更新用户画像信息。

    Args:
        state (PipelineState): 当前管道状态字典，包含用户ID和上下文信息。

    Returns:
        PipelineState: 更新后的管道状态字典，包含用户画像数据和代理执行结果。
    """
    # 调用用户画像代理获取用户详细信息
    result = await user_profile_agent.run(
        user_id=state["user_id"],
        context=state.get("context",{}),
    )
    # 从代理结果中提取画像数据并存储到状态中，同时保存完整的代理执行结果
    state["user_profile"] = getattr(result, "profile", None)
    state["agent_results"]["user_profile"] = result
    return state

async def product_recall_node(state: PipelineState) -> PipelineState:
    """
    执行产品召回节点，从代理获取相关产品列表并更新状态。

    Args:
        state (PipelineState): 当前管道状态对象，包含查询参数如 num_items。

    Returns:
        PipelineState: 更新后的状态对象，包含原始产品列表和代理执行结果。
    """
    # 调用产品召回代理，获取两倍于指定数量的候选产品以提供更大的选择空间
    result = await product_rec_agent.run(
        user_profile = None,
        num_items = state.get("num_items",10)*2,
    )
    # 提取代理返回的产品列表，若不存在则默认为空列表
    state["raw_products"] = getattr(result,"products",[])
    # 将完整的代理执行结果存储到状态中，供后续节点使用
    state["agent_results"]["product_recall"] = result
    return state

async def parallel_phase1(state: PipelineState):
    """
    并行执行用户画像构建和商品召回两个阶段。

    该函数通过异步并发的方式，同时调用用户画像节点和商品召回节点，
    以优化流水线执行效率。执行完成后，将两个节点返回的状态更新合并到主状态中。

    Args:
        state (PipelineState): 当前流水线的状态字典，包含执行所需的上下文信息。

    Returns:
        PipelineState: 更新后的流水线状态，包含用户画像和商品召回的结果。
    """
    # 并发执行用户画像构建和商品召回任务
    profile_state, recall_state = await asyncio.gather(
        user_profile_node(dict(state)),
        product_recall_node(dict(state))
    )
    # 将并行任务的结果合并回主状态
    state.update(profile_state)
    state.update(recall_state)
    return state


async def rerank_node(state: PipelineState) -> PipelineState:
    """
    执行商品重排序节点，调用代理获取基于用户画像的推荐结果并更新状态。

    Args:
        state (PipelineState): 当前管道状态对象，包含用户画像、所需商品数量等上下文信息。

    Returns:
        PipelineState: 更新后的管道状态对象，其中包含了重排序后的商品列表及代理执行结果。
    """
    # 调用产品推荐代理，根据用户画像和指定数量获取重排序结果
    result = await product_rec_agent.run(
        candidates = state.get("raw_products"),
        user_profile = state.get("user_profile"),
        num_items=state.get("num_items",10),
    )

    # 从代理结果中提取商品列表，若提取失败则保留原有值或默认为空列表
    state["ranked_products"] = getattr(result,"products",state.get("ranked_products",[]))
    
    # 记录代理的完整执行结果到状态中
    state["agent_results"]["rerank"] = result
    
    return state

async def inventory_node(state: PipelineState) -> PipelineState:
    """
    执行库存代理节点，处理原始产品列表并更新管道状态。

    该异步函数调用库存代理来获取可用产品信息，并将结果整合到当前的管道状态中。
    主要操作包括提取可用产品ID集合以及保存代理的完整执行结果。

    Args:
        state (PipelineState): 当前的管道状态字典，包含待处理的原始产品数据('raw_products')
                               和用于存储代理结果的字典('agent_results')。

    Returns:
        PipelineState: 更新后的管道状态，新增了 'available_ids' 字段并在 'agent_results'
                       中记录了库存代理的执行结果。
    """
    result = await inventory_agent.run(
        products = state.get("raw_products",[]),
    )

    # 从代理结果中提取可用产品ID并转换为集合，若结果中不存在该属性则默认为空列表
    state["available_ids"] = set(getattr(result,"available_products",[]))

    # 将库存代理的完整执行结果保存到状态的代理结果记录中
    state["agent_results"]["inventory"] = result

    return state

async def parallel_phase2(state: PipelineState) -> PipelineState:
    """
    并行执行重排序和库存检查阶段。

    该函数通过异步并发方式同时调用重排序节点和库存节点，
    以优化处理延迟。两个节点均基于当前管道状态的副本独立运行，
    最终将两者的结果合并更新到原始状态中。

    Args:
        state (PipelineState): 当前的管道状态对象，包含处理所需的所有上下文信息。

    Returns:
        PipelineState: 更新后的管道状态对象，融合了重排序和库存检查的结果。
    """
    # 并发执行重排序和库存检查任务，并等待两者完成
    rerank_state, inv_state = await asyncio.gather(
        rerank_node(dict(state)),
        inventory_node(dict(state)),
    )
    # 将并行任务的结果合并回主状态
    state.update(rerank_state)
    state.update(inv_state)
    return state

async def filter_node(state: PipelineState) -> PipelineState:
    """
    过滤并截取最终的产品列表。

    该函数从状态中获取已排序的产品列表和可用ID集合，
    筛选出既在排序列表中又在可用ID集合中的产品。
    如果筛选结果为空，则回退使用原始排序列表。
    最后根据指定的数量截取产品列表并更新到状态中。

    Args:
        state (PipelineState): 管道状态对象，包含以下键：
            - ranked_products (list): 已排序的产品对象列表。
            - available_ids (set): 可用产品ID的集合。
            - num_items (int): 需要返回的最终产品数量，默认为10。

    Returns:
        PipelineState: 更新后的状态对象，新增了 'final_products' 键，
                       其值为筛选并截取后的产品列表。
    """
    ranked = state.get("ranked_products", [])
    avail = state.get("available_ids", set())
    num = state.get("num_items", 10)

    # 筛选出同时存在于排序列表和可用ID集合中的产品
    final = [p for p in ranked if p.product_id in avail]

    # 如果筛选后没有剩余产品，则回退使用原始排序列表
    if not final:
        final = ranked

    # 截取指定数量的产品并更新到状态中
    state["final_products"] = final[:num]
    return state


async def marketing_copy_node(state: PipelineState) -> PipelineState:
    """
    生成营销文案节点。

    该异步函数负责调用营销文案代理，根据用户画像和最终产品列表生成相应的营销文案，
    并将结果更新到状态对象中。

    Args:
        state (PipelineState): 当前管道状态对象，包含用户画像、产品列表等上下文信息。

    Returns:
        PipelineState: 更新后的管道状态对象，其中包含了生成的营销文案及代理执行结果。
    """
    # 调用营销文案代理生成文案
    result = await marketing_copy_agent.run(
        user_profile = state.get("user_profile"),
        products = state.get("final_products",[]),
    )
    # 提取生成的文案并存储到状态中，若不存在则默认为空列表
    state["marketing_copies"] = getattr(result,"copies",[])
    # 记录完整的代理执行结果
    state["agent_results"]["marketing_copy"] = result
    return state

async def aggregate_node(state: PipelineState) -> PipelineState:
    """
    聚合节点状态，计算并记录总延迟时间。

    该函数从状态中获取起始时间戳，计算从开始到当前的耗时（毫秒），
    并将结果存入状态的 'total_latency_ms' 字段中。

    Args:
        state (PipelineState): 管道当前的状态字典，应包含 '_start_time' 键（可选，默认为0）。

    Returns:
        PipelineState: 更新后的状态字典，新增了 'total_latency_ms' 字段表示总延迟毫秒数。
    """
    state["total_latency_ms"] = (time.perf_counter() - state.get("_start_time",0))*1000
    return state


def build_recommendation_graph(
    vector_store: Any = None,
    mysql_db: Any = None,
) -> StateGraph:
    """构建并编译推荐系统的状态图工作流。

    在构建图之前注入可选的服务依赖（向量存储、数据库），
    使图中的 agent 节点在运行时能够使用这些服务。

    Args:
        vector_store: 向量存储实例（Milvus）。
        mysql_db: MySQL 数据库实例。

    Returns:
        StateGraph: 编译完成的状态图实例，可用于执行推荐流程。
    """
    # 注入可选服务依赖到模块级 agent 实例
    product_rec_agent.vector_store = vector_store
    inventory_agent.db = mysql_db

    graph = StateGraph(PipelineState)

    # 注册工作流中的所有处理节点
    graph.add_node("init", init_node)
    graph.add_node("parallel_phase1", parallel_phase1)
    graph.add_node("parallel_phase2", parallel_phase2)
    graph.add_node("filter", filter_node)
    graph.add_node("marketing_copy", marketing_copy_node)
    graph.add_node("aggregate", aggregate_node)

    # 定义工作流的入口点及节点间的线性执行路径
    graph.set_entry_point("init")
    graph.add_edge("init", "parallel_phase1")
    graph.add_edge("parallel_phase1", "parallel_phase2")
    graph.add_edge("parallel_phase2", "filter")
    graph.add_edge("filter", "marketing_copy")
    graph.add_edge("marketing_copy", "aggregate")
    graph.add_edge("aggregate", END)

    return graph.compile()

