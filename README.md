# Multi-Agent E-Commerce Recommendation System

基于多智能体协作的电商推荐系统，集成 LLM 推理、向量检索、A/B 实验与监控能力。

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FastAPI Application                         │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                    请求入口 (main.py)                          │  │
│  │  POST /api/v1/recommend       POST /api/v1/recommend/graph    │  │
│  │  GET  /api/v1/experiments     GET  /api/v1/metrics            │  │
│  │  GET  /health                                                  │  │
│  └──────────────┬────────────────────────────────────────────────┘  │
│                 │                                                   │
│         ┌───────┴────────┐                                          │
│         │  Supervisor    │  ←── A/B Test Engine (分流 & Thompson)   │
│         │  Orchestrator  │                                          │
│         └───────┬────────┘                                          │
│                 │                                                   │
│    ┌────────────┼────────────┐────────────┐                        │
│    ▼            ▼            ▼            ▼                         │
│ ┌────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐                  │
│ │ User   │ │ Product  │ │Inventory │ │Marketing │                  │
│ │Profile │ │  Rec     │ │  Agent   │ │  Copy    │                  │
│ │ Agent  │ │  Agent   │ │          │ │  Agent   │                  │
│ └───┬────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘                  │
│     │           │            │            │                         │
│     └───────────┴────────────┴────────────┘                         │
│                 ▼                                                    │
│          ┌──────────────┐                                           │
│          │  Aggregator  │  (结果聚合、库存过滤、文案关联)            │
│          └──────────────┘                                           │
└─────────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐   ┌──────────────────┐   ┌──────────────────┐
│    Redis      │   │     Milvus       │   │      MySQL       │
│  Feature Store│   │  Vector Store    │   │  Business Data   │
│  行为缓存 &   │   │ 商品 Embedding   │   │ 商品/用户/库存    │
│  实时特征     │   │ 相似性搜索       │   │  持久化          │
└───────────────┘   └──────────────────┘   └──────────────────┘
```

---

## 数据流详解

### Supervisor 编排模式 (`POST /api/v1/recommend`)

**三阶段流水线：**

```
第一阶段 [并行]
  ┌─ UserProfileAgent → 用户画像 (LLM + Redis 行为特征)
  └─ ProductRecAgent  → 商品召回 (Milvus 向量召回 / Mock 数据)

第二阶段 [并行]
  ┌─ ProductRecAgent (rerank) → LLM 重排 / 规则降级排序
  └─ InventoryAgent           → 库存检查 & 限购策略

第三阶段 [串行]
  └─ MarketingCopyAgent → 基于用户分群的文案生成 + 合规校验
```

### LangGraph DAG 模式 (`POST /api/v1/recommend/graph`)

```
[init] → [parallel_phase1] → [parallel_phase2] → [filter] → [marketing_copy] → [aggregate] → [END]
           ├─ user_profile     ├─ rerank
           └─ product_recall   └─ inventory
```

两种模式共享相同的 Agent 组件，LangGraph 模式显式展示了有向无环图的执行拓扑。

---

## 核心组件

### Agents（智能体层）

| Agent | 职责 | 技术实现 |
|-------|------|---------|
| **UserProfileAgent** | 用户分群（RFM）、偏好提取、实时标签 | `ChatOpenAI` + Redis Feature Store |
| **ProductRecAgent** | 两阶段召回+排序：召回 (向量+规则) → 重排 (LLM/规则) | Milvus 语义检索 + LLM 重排 + 类目打散/新品加权 |
| **InventoryAgent** | 实时库存查询、安全库存预警、动态限购 | MySQL 实时查询 (降级: 默认值) |
| **MarketingCopyAgent** | 基于用户分群选择文案模板、LLM 生成、违禁词过滤 | 5 套 Prompt 模板（新客/高价值/价格敏感/活跃/流失）+ 合规词表 |

**Agent 基类 (`BaseAgent`)：**
- 统一接口：`run()` → `_execute()`（模板方法模式）
- 自动重试：`tenacity` 指数退避（`wait_exponential`）
- 优雅降级：异常时返回 `AgentResult(success=False, confidence=0.0)` 而非抛出

### 基础设施层

| 服务 | 用途 | 降级策略 |
|------|------|---------|
| **Redis** | 用户行为序列（Sorted Set）、实时 RFM 特征计算 | Redis 不可用 → 使用请求上下文中的默认数据 |
| **Milvus** | 商品 Embedding 向量存储、相似性搜索 | Milvus 不可用 → 切换到 Mock 商品池，按偏好类目排序 |
| **MySQL** | 商品/用户画像/库存/行为日志持久化 | DB 不可用 → 返回默认值，不影响主流程 |

### A/B 实验引擎

- **流量分桶**: 用户 ID + 实验 ID → MD5 哈希取模（确定性分桶）
- **默认实验**: 推荐策略实验（rule_based vs llm）、文案风格实验（formal vs casual）
- **Thompson Sampling**: 基于 Beta 分布的自适应流量分配（`assign_thompson`）
- **实验层级**: Agent 级别、模型参数级别

### 监控指标收集

| 维度 | 指标 | 实现 |
|------|------|------|
| Agent 性能 | 调用次数、成功率、平均延迟 | `MetricsCollector.record_agent_call()` |
| 业务事件 | CTR / CVR / GMV | `record_business_event()` |
| A/B 实验 | 各分组指标均值、标准差 | `ABTestEngine.get_stats()` |

---

## 环境要求

- Python 3.12+
- Docker & Docker Compose（Redis / Milvus / MySQL）

---

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/your-username/Multi-Agent-E-Commerce-Recommendation-System.git
cd Multi-Agent-E-Commerce-Recommendation-System
```

### 2. 创建虚拟环境

```bash
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate
```

### 3. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`：

```ini
ECOM_LLM_API_KEY=sk-your-api-key
ECOM_LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
ECOM_LLM_MODEL=qwen-plus
ECOM_REDIS_URL=redis://localhost:6379/0
ECOM_MILVUS_HOST=localhost
ECOM_MILVUS_PORT=19530
ECOM_DATABASE_URL=sqlite:///./ecommerce.db
ECOM_AB_TEST_ENABLED=true
```

### 4. 安装依赖

```bash
pip install -r requirements.txt
```

### 5. 启动基础设施

```bash
docker compose up -d redis mysql milvus
```

### 6. 启动 API 服务

```bash
python main.py
```

服务默认监听 `http://localhost:8000`，Swagger UI 访问 `http://localhost:8000/docs`。

---

## API 参考

### 健康检查

```bash
curl http://localhost:8000/health
```

### 获取个性化推荐（Supervisor 模式）

```bash
curl -X POST http://localhost:8000/api/v1/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_001",
    "scene": "homepage",
    "num_items": 5,
    "context": {
      "preferred_categories": ["手机", "平板"],
      "device": "mobile"
    }
  }'
```

### 通过 LangGraph 管道推荐

```bash
curl -X POST http://localhost:8000/api/v1/recommend/graph \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_001",
    "scene": "homepage",
    "num_items": 3
  }'
```

### 查看 A/B 实验状态

```bash
curl http://localhost:8000/api/v1/experiments
```

### 查看系统监控指标

```bash
curl http://localhost:8000/api/v1/metrics
```

---

## 项目结构

```
├── main.py                     # FastAPI 应用入口，路由定义，依赖初始化
├── config/
│   └── settings.py             # Pydantic Settings（环境变量 + .env 文件）
│
├── models/
│   ├── schemas.py              # 数据模型（UserProfile / Product / 各类请求响应）
│   └── database.py             # SQLAlchemy ORM 定义（products / user_profiles / user_behaviors / inventory_log）
│
├── agents/                     # 智能体层（继承 BaseAgent）
│   ├── base_agent.py           # 抽象基类（重试 / 超时 / 降级 / 错误率统计）
│   ├── user_profile_agent.py   # 用户画像 Agent（LLM分群 + RFM + 实时标签）
│   ├── product_rec_agent.py    # 商品推荐 Agent（召回 + LLM重排 + 规则降级）
│   ├── inventory_agent.py      # 库存决策 Agent（库存检查 / 预警 / 限购策略）
│   └── marketing_copy_agent.py # 营销文案 Agent（Prompt模板 + LLM生成 + 合规审核）
│
├── orchestrator/               # 编排层
│   ├── supervisor.py           # Supervisor 编排器（并行分发 + 聚合）
│   └── graph.py                # LangGraph 状态图（DAG 流水线）
│
├── services/                   # 基础设施服务层
│   ├── feature_store.py        # Redis 特征存储（行为序列 + 滑动窗口 RFM）
│   ├── vector_store.py         # Milvus 向量存储（商品 Embedding 检索）
│   ├── mysql_database.py       # MySQL 异步数据库（SQLAlchemy async_engine）
│   ├── ab_test.py              # A/B 测试引擎（分桶 / Thompson Sampling / 指标统计）
│   └── metrics.py              # 监控指标收集器（Agent 延迟、成功率、业务事件）
│
├── tests/
│   └── test_ab_test.py         # A/B 测试引擎单元测试
│
├── docker-compose.yml          # Redis + Milvus + MySQL 容器编排
├── Dockerfile                  # Python 应用容器镜像
├── requirements.txt            # Python 依赖清单
└── .env.example                # 环境变量模板
```
