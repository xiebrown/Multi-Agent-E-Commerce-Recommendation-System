"""
A/B测试引擎
- 流量分桶：用户ID哈希取模分桶
- 实验层：Agent级别 / 模型级别 / Prompt级别实验
- MAB算法：Thompson Sampling动态分配流量
- 指标收集：CTR / CVR / GMV / 停留时长
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Experiment:
    """
    实验配置数据类，用于定义一个A/B测试或功能实验的基本属性。

    Attributes:
        id (str): 实验的唯一标识符。
        name (str): 实验的名称，用于展示和识别。
        groups (list[ExperimentGroup]): 实验包含的分组列表，每个分组定义了不同的实验变体及其流量分配。
        enabled (bool): 实验是否启用，默认为 True。
        start_time (float): 实验开始的时间戳（Unix时间），默认为 0.0。
        end_time (float): 实验结束的时间戳（Unix时间），默认为 0.0。
    """
    id: str
    name: str
    groups: list[ExperimentGroup]
    enabled: bool = True
    start_time: float = 0.0
    end_time: float = 0.0


@dataclass
class ExperimentGroup:
    """
    表示一个实验组的配置和状态数据类。

    该类用于存储实验组的基本信息，包括名称、权重、自定义配置以及成功和失败的计数。
    通常用于A/B测试或多臂老虎机算法中，以跟踪不同实验组的表现。

    Attributes:
        name (str): 实验组的唯一标识名称。
        weight (int): 实验组的权重，用于决定流量分配比例。默认为50。
        config (dict[str, Any]): 与该实验组相关的额外配置参数。默认为空字典。
        successes (int): 该实验组成功的次数。默认为1（用于平滑处理或先验假设）。
        failures (int): 该实验组失败的次数。默认为1（用于平滑处理或先验假设）。
    """
    name: str
    weight: int = 50
    config: dict[str, Any] = field(default_factory=dict)
    successes: int = 1
    failures: int = 1


class ABTestEngine:
    """
    A/B 测试引擎类，用于管理实验分组、指标收集及默认实验初始化。

    Attributes:
        bucket_count (int): 分桶总数，默认为 100。
        experiments (dict[str, Experiment]): 存储所有已注册的实验对象。
        _metrics (list[dict[str, Any]]): 内部存储收集的指标数据列表。
    """

    def __init__(self, bucket_count: int = 100):
        """
        初始化 ABTestEngine 实例。

        Args:
            bucket_count (int): 分桶的总数量，用于决定用户或流量被分配到的桶数。
                                默认值为 100。

        Returns:
            None
        """
        self.bucket_count = bucket_count
        self.experiments: dict[str, Experiment] = {}
        self._metrics: list[dict[str, Any]] = []
        
        # 初始化默认的实验配置
        self._init_default_experiments()
    def _init_default_experiments(self):
        """
        初始化默认的A/B测试实验配置。

        该函数负责注册系统启动时所需的默认实验，包括推荐策略对比实验和文案风格对比实验。
        每个实验均包含对照组和处理组，并默认分配相等的流量权重。

        Returns:
            None
        """
        # 注册推荐策略实验：对比基于规则的重排（对照组）与基于LLM的重排（处理组），各占50%流量
        self.register_experiment(
            Experiment(
                id="rec_strategy",
                name="推荐策略实验",
                groups=[
                    ExperimentGroup(name="control", weight=50, config={"rerank": "rule_based"}),
                    ExperimentGroup(name="treatment_llm", weight=50, config={"rerank": "llm"}),
                ],
            )
        )
        # 注册文案风格实验：对比正式风格（对照组）与休闲风格（处理组），各占50%流量
        self.register_experiment(
            Experiment(
                id="copy_style",
                name="文案风格实验",
                groups=[
                    ExperimentGroup(name="formal", weight=50, config={"style": "formal"}),
                    ExperimentGroup(name="casual", weight=50, config={"style": "casual"}),
                ],
            )
        )
    
    def register_experiment(self, exp: Experiment):
        """注册一个实验对象。

        将传入的实验对象存储到内部字典中，以实验ID作为键。

        Args:
            exp (Experiment): 要注册的实验对象，必须包含有效的id属性。

        Returns:
            None
        """
        self.experiments[exp.id] = exp

    def assign(self, user_id: str, experiment_id: str = "rec_strategy") -> dict[str, Any]:
        """分配用户到指定的实验组。

        Args:
            user_id: 用户的唯一标识符。
            experiment_id: 实验的唯一标识符，默认为 "rec_strategy"。

        Returns:
            包含用户所属组名和对应配置的字典。如果实验不存在或未启用，则返回控制组信息。
        """
        exp = self.experiments.get(experiment_id)
        # 如果实验不存在或未启用，直接返回控制组默认配置
        if not exp or not exp.enabled: return {"group": "control","config": {}}

        bucket = self._hash_bucket(user_id,experiment_id)
        group = self._bucket_to_group(bucket, exp.groups)
        return {"group": group.name, "config": group.config}
    
    def assign_thompson(self, user_id: str, experiment_id: str = "rec_strategy") -> dict[str, Any]:
        """
        使用汤普森采样算法为用户分配实验组。

        Args:
            user_id (str): 用户唯一标识符。
            experiment_id (str): 实验ID，默认为 "rec_strategy"。

        Returns:
            dict[str, Any]: 包含分配结果的字典，结构为 {"group": 组名, "config": 配置信息}。
                            如果实验不存在或未启用，则返回默认控制组。
        """
        exp = self.experiments.get(experiment_id)
        if not exp or not exp.enabled: return {"group": "control","config": {}}

        # 为每个实验组计算汤普森采样样本值并找出最优组
        samples = []
        for group in exp.groups: 
            sample = np.random.beta(group.successes, group.failures)
            samples.append((sample,group))

        best = max(samples,key=lambda x:x[0])[1]
        return {"group": best.name, "config": best.config}
    
    def record_outcome(self, experiment_id: str, group_name: str, success: bool):
        """记录实验结果。

        根据实验ID和组名，更新对应组的成功或失败计数。

        Args:
            experiment_id (str): 实验的唯一标识符。
            group_name (str): 实验组的名称。
            success (bool): 实验结果是否成功。

        Returns:
            None
        """
        exp = self.experiments.get(experiment_id)
        if not exp: return

        # 查找匹配的实验组并更新对应的成功或失败计数
        for group in exp.groups: 
            if group.name == group_name: 
                if success: group.successes += 1
                else: group.failures += 1
                break
    def record_metric(
            self,
            experiment_id: str,
            group_name: str,
            metric_name: str,
            value: float,
            user_id: str = "",
    ):
        """记录实验指标数据。

        将指定的指标信息封装为字典并追加到内部指标列表中，同时自动添加当前时间戳。

        Args:
            experiment_id (str): 实验的唯一标识ID。
            group_name (str): 指标所属的组名。
            metric_name (str): 指标的名称。
            value (float): 指标的数值。
            user_id (str, optional): 用户ID，默认为空字符串。

        Returns:
            None
        """
        # 构建包含实验信息、指标数据及当前时间戳的记录字典，并添加到指标列表中
        self._metrics.append(
            {
                "experiment_id": experiment_id,
                "group": group_name,
                "metric": metric_name,
                "value": value,
                "user_id": user_id,
                "timestamp": time.time(),
            }
        )

    def get_stats(self, experiment_id: str) -> dict[str, Any]:
        """获取指定实验的统计信息。

        Args:
            experiment_id: 实验的唯一标识符。

        Returns:
            一个字典，键为组名，值为该组下各指标的统计结果字典。
            统计结果包含：count（数量）、mean（均值）、std（标准差）、min（最小值）、max（最大值）。
            如果实验不存在，则返回空字典。
        """
        exp = self.experiments.get(experiment_id)
        if not exp: return {}

        # 筛选出属于当前实验的所有指标记录
        relevant = [m for m in self._metrics if m["experiment_id"] == experiment_id]
        
        # 将指标数据按组和指标名称进行分组聚合
        stats: dict[str, dict[str, list[float]]] = {}
        for m in relevant:
            grp = m["group"]
            metric = m["metric"]
            if grp not in stats: 
                stats[grp] = {}
            if metric not in stats[grp]: 
                stats[grp][metric] = []
            stats[grp][metric].append(m["value"])

        # 计算每个组下每个指标的统计量
        result: dict[str, Any] = {}
        for grp, metrics in stats.items(): 
            result[grp] = {}
            for metric_name, values in metrics.items(): 
                arr = np.array(values)
                result[grp][metric_name] = {
                    "count": len(values),
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                }
        return result
    
    def _hash_bucket(self, user_id: str, experiment_id: str) -> int:
        """
        根据用户ID和实验ID计算哈希桶索引。

        通过拼接用户ID和实验ID生成字符串，使用MD5算法进行哈希处理，
        取哈希值的前8位十六进制字符转换为整数后对桶数量取模，
        从而确定该用户在该实验中所属的桶编号。

        Args:
            user_id (str): 用户的唯一标识符。
            experiment_id (str): 实验的唯一标识符。

        Returns:
            int: 计算得到的桶索引，范围在 [0, self.bucket_count) 之间。
        """
        raw = f"{user_id}:{experiment_id}"
        h = hashlib.md5(raw.encode()).hexdigest()
        return int(h[:8], 16) % self.bucket_count
    
    def _bucket_to_group(self, bucket: int, groups: list[ExperimentGroup]) -> ExperimentGroup:
        """
        将桶索引映射到对应的实验组。

        根据各实验组的权重比例，将离散的桶索引转换为连续的权重空间，
        并确定该桶所属的实验组。

        Args:
            bucket (int): 桶的索引值，范围通常为 [0, self.bucket_count)。
            groups (list[ExperimentGroup]): 实验组列表，每个组包含权重信息。

        Returns:
            ExperimentGroup: 匹配到的实验组对象。如果未找到匹配项（理论上不应发生），
                             则返回最后一个实验组作为兜底。
        """
        # 计算所有实验组的总权重
        total_weight = sum(g.weight for g in groups)
        cumulative = 0
        # 将桶索引归一化到总权重空间，得到当前桶在权重分布中的位置
        normalized_bucket = bucket * total_weight / self.bucket_count
        for group in groups: 
            # 累加当前组的权重，检查是否覆盖归一化后的桶位置
            cumulative += group.weight
            if cumulative >= normalized_bucket: return group

        # 兜底返回最后一个实验组，防止因浮点精度或边界情况导致无匹配
        return groups[-1]