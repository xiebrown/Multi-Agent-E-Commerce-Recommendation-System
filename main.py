"""
Multi-Agent E-Commerce Recommendation System — FastAPI Entry Point

Endpoints:
  POST /api/v1/recommend          - 获取个性化推荐
  POST /api/v1/recommend/graph    - 通过LangGraph pipeline推荐
  GET  /api/v1/experiments        - 查看A/B实验状态
  GET  /api/v1/metrics            - 查看系统监控指标
  GET  /health                    - 健康检查
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from contextlib import asynccontextmanager
from json import JSONDecodeError
from typing import Any

import structlog
import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse

from config import get_settings
from models.schemas import RecommendationRequest, RecommendationResponse
from orchestrator.supervisor import SupervisorOrchestrator
from orchestrator.graph import build_recommendation_graph
from services import ABTestEngine, MetricsCollector, MySQLDatabase, VectorStore

logger = structlog.get_logger()
settings = get_settings()


ab_engine = ABTestEngine()
metrics_collector = MetricsCollector()
supervisor: SupervisorOrchestrator | None = None
rec_graph = None
vector_store: VectorStore | None = None
mysql_db: MySQLDatabase | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rec_graph, supervisor, vector_store, mysql_db

    # 初始化 VectorStore（Milvus）
    vector_store = VectorStore(
        host=settings.milvus_host,
        port=settings.milvus_port,
        collection_name=settings.milvus_collection,
        dimension=settings.milvus_dimension,
    )
    await vector_store.connect()

    # 初始化 MySQLDatabase
    if settings.mysql_database_url:
        mysql_db = MySQLDatabase(database_url=settings.mysql_database_url)
        await mysql_db.connect()
        await mysql_db.init_tables()
    else:
        mysql_db = MySQLDatabase(database_url=None)

    # 使用服务依赖创建编排器
    supervisor = SupervisorOrchestrator(
        ab_engine=ab_engine,
        vector_store=vector_store,
        mysql_db=mysql_db,
    )
    rec_graph = build_recommendation_graph(
        vector_store=vector_store,
        mysql_db=mysql_db,
    )

    logger.info("app.startup", model=settings.llm_model)
    yield
    # 关闭服务连接
    if vector_store:
        await vector_store.close()
    if mysql_db:
        await mysql_db.close()
    logger.info("app.shutdown")


app = FastAPI(
    title="Multi-Agent E-Commerce Recommendation System",
    description="用户画像Agent + 商品推荐Agent + 营销文案Agent + 库存决策Agent，并行+聚合模式",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def json_validation_handler(_request: Request, exc: RequestValidationError):
    """捕获 JSON 解析错误并返回友好提示。"""
    for err in exc.errors():
        if err.get("type") == "json_invalid":
            ctx = err.get("ctx", {})
            return JSONResponse(
                status_code=400,
                content={
                    "detail": f"请求体 JSON 格式错误: {ctx.get('error', '未知错误')}",
                    "hint": "请检查 JSON 是否有末尾逗号、引号不匹配等语法问题。推荐用 Python requests 或 Swagger UI (/docs) 测试。",
                },
            )
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.get("/health")
async def health():
    return {"status": "healthy", "model": settings.llm_model}


@app.post("/api/v1/recommend", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest):
    """使用Supervisor编排器进行推荐 (生产推荐用法)"""
    response = await supervisor.recommend(request)
    _collect_metrics(response)
    return response


@app.post("/api/v1/recommend/graph")
async def recommend_via_graph(request: RecommendationRequest):
    """使用LangGraph状态图进行推荐 (展示LangGraph能力)"""
    if not rec_graph:
        return {"error": "Graph not initialized"}
    state = {
        "user_id": request.user_id,
        "scene": request.scene,
        "num_items": request.num_items,
        "context": request.context,
    }
    result = await rec_graph.ainvoke(state)
    return {
        "request_id": result.get("request_id"),
        "user_id": result.get("user_id"),
        "products": [p.model_dump() for p in result.get("final_products", [])],
        "marketing_copies": result.get("marketing_copies", []),
        "experiment_group": result.get("experiment_group", "control"),
        "total_latency_ms": round(result.get("total_latency_ms", 0), 1),
    }


@app.get("/api/v1/experiments")
async def get_experiments():
    """查看所有A/B实验状态"""
    experiments = {}
    for exp_id, exp in ab_engine.experiments.items():
        experiments[exp_id] = {
            "name": exp.name,
            "enabled": exp.enabled,
            "groups": [
                {
                    "name": g.name,
                    "weight": g.weight,
                    "config": g.config,
                    "successes": g.successes,
                    "failures": g.failures,
                }
                for g in exp.groups
            ],
            "stats": ab_engine.get_stats(exp_id),
        }
    return experiments


@app.get("/api/v1/metrics")
async def get_metrics():
    """查看系统监控指标"""
    return {
        "agents": metrics_collector.get_agent_stats(),
        "business": metrics_collector.get_business_stats(),
    }


@app.post("/api/v1/experiments/{experiment_id}/outcome")
async def record_outcome(experiment_id: str, group: str, success: bool):
    """记录A/B测试结果,更新Thompson Sampling"""
    ab_engine.record_outcome(experiment_id, group, success)
    return {"status": "recorded"}


def _custom_openapi():
    """为 /api/v1/recommend 的 Request body 添加 Media Type 级别示例，
    确保 Swagger UI 的 "Try it out" 能正确预填 context 的列表值。"""
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        summary=app.summary,
        description=app.description,
        routes=app.routes,
    )
    try:
        content = openapi_schema["paths"]["/api/v1/recommend"]["post"]["requestBody"]["content"]["application/json"]
        content["example"] = {
            "user_id": "user_001",
            "scene": "homepage",
            "num_items": 10,
            "context": {"preferred_categories": ["手机", "平板"]},
        }
    except KeyError:
        pass
    app.openapi_schema = openapi_schema
    return openapi_schema


app.openapi = _custom_openapi


def _collect_metrics(response: RecommendationResponse):
    for name, result in response.agent_results.items():
        metrics_collector.record_agent_call(
            agent_name=name,
            success=result.success,
            latency_ms=result.latency_ms,
        )


if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
