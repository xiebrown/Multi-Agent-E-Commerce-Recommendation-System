from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from models.schemas import AgentResult


# 初始化结构化日志记录器实例，用于后续的业务逻辑日志输出
logger = structlog.get_logger()

class BaseAgent(ABC):
    """
    BaseAgent 是所有代理的基类，定义了代理的通用接口（重试，超时，降级）。
    """

    def __init__(self, name: str, timeout: float = 10.0, max_retries: int = 2):
        self.name = name
        self.timeout = timeout
        self.max_retries = max_retries
        self._call_count = 0
        self._error_count = 0

    @abstractmethod
    async def _execute(self,**kwargs: Any) -> AgentResult:
        """
        执行代理逻辑的抽象方法，由子类实现。
        """
        pass

    async def run(self, **kwargs: Any) -> AgentResult:
        """
        公共入口：对 _execute 进行包装，加入计时、重试、降级（兜底）功能。
        """
        # 记录函数调用（成功）开始时间并增加调用计数
        start = time.perf_counter()
        self._call_count += 1

        try:
            # 执行带重试机制的操作并记录耗时
            result = await self._retry_execute(**kwargs)
            result.latency_ms = (time.perf_counter() - start) * 1000
            logger.info("Agent completed successfully",
                        agent=self.name,
                        latency_ms=round(result.latency_ms,1),
            )
            return result
        except Exception as e:
            # 捕获异常，更新错误（失败）计数并记录日志，最后执行降级策略
            self._error_count += 1
            latency_ms = (time.perf_counter() - start) * 1000
            logger.error("Agent failed",
                         agent=self.name,
                         error=str(e)
            )
            return self._fallback(e, latency_ms)
        
    async def _retry_execute(self, **kwargs: Any) -> AgentResult:
        """
        执行代理逻辑，并使用 tenacity 库进行重试。
        """
        @retry(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential(multiplier=0.5, min=0.5, max=4)
            )
        async def _inner():
            return await self._execute(**kwargs)
            
        return await _inner()
    
    def _fallback(self, error: Exception, latency_ms: float) -> AgentResult:
        """
        执行降级策略，返回一个默认的 AgentResult 对象。
        """
        return AgentResult(
            agent_name=self.name,
            success=False,
            error=str(error),
            latency_ms=latency_ms,
            confidence=0.0,
        )
    
    @property
    def error_rate(self) -> float:
        """
        获取代理的错误率。
        """
        if self._call_count == 0:
            return 0.0
        return self._error_count / self._call_count
