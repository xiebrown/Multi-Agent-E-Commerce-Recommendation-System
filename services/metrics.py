"""
监控指标收集
- Agent调用成功率 / 延迟
- 推荐CTR / CVR / GMV
- A/B测试实验指标
"""


from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentMetrics:
    """
    用于记录和统计代理（Agent）运行指标的数据类。

    Attributes:
        call_count (int): 代理被调用的总次数，默认为 0。
        success_count (int): 代理成功执行的次数，默认为 0。
        total_latency_ms (float): 代理执行的总延迟时间（毫秒），默认为 0.0。
        errors (list[str]): 记录执行过程中发生的错误信息列表，默认为空列表。
    """
    call_count: int = 0
    success_count: int = 0
    total_latency_ms: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """
        计算并返回代理的成功率。

        成功率定义为成功次数除以总调用次数。如果总调用次数为 0，则返回 0.0 以避免除零错误。

        Returns:
            float: 代理的成功率，范围在 0.0 到 1.0 之间。
        """
        return self.success_count / self.call_count if self.call_count else 0.0
    
    @property
    def avg_latency_ms(self) -> float:
        """
        计算并返回代理的平均延迟时间。

        平均延迟定义为总延迟时间除以总调用次数。如果总调用次数为 0，则返回 0.0 以避免除零错误。

        Returns:
            float: 代理的平均延迟时间（毫秒）。
        """
        return self.total_latency_ms / self.call_count if self.call_count else 0.0
    

class MetricsCollector:
    """
    指标收集器类，用于聚合和管理代理指标及业务事件数据。

    Attributes:
        _agent_metrics (dict[str, AgentMetrics]): 存储每个代理的指标数据，键为代理标识，值为对应的AgentMetrics对象。
        _business_evnets (list[dict[str, Any]]): 存储捕获的业务事件列表，每个事件以字典形式保存。
    """
    def __init__(self):
        self._agent_metrics: dict[str, AgentMetrics] = defaultdict(AgentMetrics)
        self._business_evnets: list[dict[str, Any]] = []

    def record_agent_call(self, agent_name: str, success: bool, latency_ms: float, error: str =""):
        """记录智能体调用的 metrics 信息。

        Args:
            agent_name (str): 智能体的名称，用于标识对应的指标记录对象。
            success (bool): 调用是否成功。
            latency_ms (float): 调用延迟，单位为毫秒。
            error (str, optional): 如果调用失败，记录具体的错误信息。默认为空字符串。

        Returns:
            None
        """
        m = self._agent_metrics[agent_name]
        # 更新调用计数和累计延迟
        m.call_count += 1
        if success:
            m.success_count += 1
        m.total_latency_ms += latency_ms

        # 如果存在错误信息，则将其添加到错误列表中
        if error:
            m.errors.append(error)

    def record_business_event(self, event_type: str, **kwargs: Any):
        """
        记录业务事件。

        将事件类型、当前时间戳以及额外的关键字参数打包成字典，
        并添加到内部业务事件列表中。

        Args:
            event_type (str): 业务事件的类型标识。
            **kwargs (Any): 与事件相关的额外键值对数据。

        Returns:
            None
        """
        self._business_evnets.append({"type": event_type, "timestamp":time.time(),**kwargs})
    
    def get_agent_stats(self) -> dict[str, dict[str, Any]]:
        """获取所有代理的统计信息。

        Returns:
            dict[str, dict[str, Any]]: 一个字典，键为代理名称，值为包含以下指标的字典：
                - call_count (int): 调用次数
                - success_rate (float): 成功率，保留4位小数
                - avg_latency_ms (float): 平均延迟（毫秒），保留1位小数
                - recent_errors (list): 最近5条错误记录
        """
        result={}
        # 遍历每个代理的指标数据，构建统计结果
        for agent_name, m in self._agent_metrics.items():
            result[agent_name] = {
                "call_count": m.call_count,
                "success_rate": round(m.success_rate, 4),
                "avg_latency_ms": round(m.avg_latency_ms, 1),
                "recent_errors": m.errors[-5:],
            }
        return result
    

    def get_business_stats(self) -> dict[str, Any]:
        """
        获取业务事件的统计信息。

        该方法将内部存储的业务事件按类型进行分组，并统计每种类型的事件数量。

        Returns:
            dict[str, Any]: 一个字典，键为事件类型（str），值为包含统计信息的字典。
                            目前统计信息仅包含该类型事件的数量（"count": int）。
                            如果没有业务事件，则返回空字典。
        """
        if not self._business_evnets: 
            return {}
        # 按事件类型对业务事件进行分组
        by_type: dict[str, list[dict]] = defaultdict(list)
        for event in self._business_evnets:
            by_type[event["type"]].append(event)
        
        # 计算每种事件类型的统计数据
        stats = {}
        for event_type, events in by_type.items():
            stats[event_type] = {
                "count": len(events),
            }
        return stats