"""A/B测试引擎单元测试"""

import sys
import os



sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.ab_test import ABTestEngine, Experiment, ExperimentGroup


def test_consistent_assignment():
    """
    测试同一用户多次分配实验组的一致性。

    验证 ABTestEngine 对相同用户ID进行多次分配时，
    返回的实验组结果保持一致。

    Returns:
        None: 该函数为测试用例，无返回值，通过断言验证结果。
    """
    engine = ABTestEngine()
    group1 = engine.assign("user_001")
    group2 = engine.assign("user_001")
    assert group1["group"]  == group2["group"]

def test_distribution():
    """
    测试 AB 测试引擎的用户分组分布是否均匀。

    该函数通过模拟 1000 次用户分配请求，统计各分组的用户数量，
    并验证每个分组的用户数是否处于合理区间（300 到 700 之间），
    以确保分组算法没有严重的偏差。

    Returns:
        None: 如果断言失败则抛出 AssertionError，否则正常返回。
    """
    engine = ABTestEngine()
    counts: dict[str, int] = {}

    # 模拟 1000 次用户分配并统计各分组的人数
    for i in range(1000):
        result = engine.assign(f"user_{i}")
        grp = result["group"]
        counts[grp] = counts.get(grp, 0) + 1

    # 验证每个分组的用户数量是否在预期范围内，防止分组严重倾斜
    for grp, count in counts.items():
        assert 300 < count < 700, f"Group {grp} has {count} users — too skewed"

def test_thompson_sampling():
    """
    测试 Thompson Sampling 算法在 A/B 测试引擎中的基本行为。

    该测试通过模拟“treatment_llm”组全部成功、“control”组全部失败的极端场景，
    验证实验引擎是否正确记录了各组的结果，并确保处理组的成功次数高于对照组。

    Returns:
        None: 此函数不返回任何值，仅通过断言验证内部状态。
    """
    engine = ABTestEngine()

    # 模拟 treatment_llm 组的 100 次成功结果
    for _ in range(100):
        engine.record_outcome("rec_strategy", "treatment_llm", True)

    # 模拟 control 组的 100 次失败结果
    for _ in range(100):
        engine.record_outcome("rec_strategy", "control", False)

    # 获取实验对象并提取对应的处理组和对照组
    exp = engine.experiments["rec_strategy"]
    treatment = next(g for g in exp.groups if g.name == "treatment_llm")
    control = next(g for g in exp.groups if g.name == "control")

    # 验证处理组的成功次数是否严格大于对照组
    assert treatment.successes > control.successes

def test_custom_experiment():
    """
    测试自定义实验的注册与用户分组分配功能。

    该函数初始化AB测试引擎，注册一个包含两个不同权重分组的提示模板实验，
    并验证指定用户是否被正确分配到其中一个实验组中。

    Returns:
        None: 此函数不返回任何值，通过断言验证实验逻辑的正确性。
    """
    engine = ABTestEngine()

    # 注册一个名为“Prompt模板实验”的A/B测试，包含两个权重分别为30%和70%的分组
    engine.register_experiment(
        Experiment(
            id="prompt_test",
            name="Prompt模板实验",
            groups=[
                ExperimentGroup(name="template_a", weight=30),
                ExperimentGroup(name="template_b", weight=70),
            ],
        )
    )

    # 为特定用户分配实验组并验证结果是否在预期的分组范围内
    result = engine.assign("user_999", "prompt_test")
    assert result["group"] in ["template_a", "template_b"]


def test_metrics_recording():
    """
    测试 ABTestEngine 的指标记录与统计功能。

    该测试用例验证以下逻辑：
    1. 能够正确记录不同实验组（control, treatment_llm）下的用户指标数据。
    2. 能够通过 get_stats 方法获取指定实验的统计数据。
    3. 验证统计结果中是否包含预期的实验组，且指标计数准确。
    """
    engine = ABTestEngine()
    engine.record_metric("rec_strategy", "control", "ctr", 0.05, "user_001")
    engine.record_metric("rec_strategy", "control", "ctr", 0.08, "user_002")
    engine.record_metric("rec_strategy", "treatment_llm", "ctr", 0.12, "user_003")

    # 获取 "rec_strategy" 实验的统计数据并验证控制组的存在性及 CTR 记录数量
    stats = engine.get_stats("rec_strategy")
    assert "control" in stats
    assert stats["control"]["ctr"]["count"] == 2


if __name__ == "__main__":
    # 执行A/B测试引擎的核心功能测试，包括一致性分配、流量分布、汤普森采样算法、自定义实验配置及指标记录
    test_consistent_assignment()
    test_distribution()
    test_thompson_sampling()
    test_custom_experiment()
    test_metrics_recording()
    print("All A/B test engine tests passed!")