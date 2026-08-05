# 通用文档分析 Agent · 企业版 v5.0

> **定位：通用文档分析底座 + 领域适配层。** 底座（RAG/记忆/HITL/多租户/SSE/Skill 筛选）完全领域无关；电力只是第一个落地的领域，已内置第二领域（通用文档）演示——**换领域只需切配置 + 注册 Skill，底座零改动**。

## 领域切换（核心特性）

```bash
# 切到电力领域（默认）
DOMAIN=power python main.py --port 8090
# 切到通用文档分析领域
DOMAIN=generic python main.py --port 8090
```

| 领域                | 意图                    | Skills                                        | 演示文档                    |
| ------------------- | ----------------------- | --------------------------------------------- | --------------------------- |
| **power**（电力）   | 规程检索/造价/故障/对比 | power_rag / quota_match / comparison_analysis | DL/T 572 等 4 篇            |
| **generic**（通用） | 文档问答/对比/总结      | doc_qa / doc_compare / doc_summary            | 差旅报销/请假/机房出入 3 篇 |

**领域化设计**：意图关键词、提示词、Skill、演示文档全部在 `config/domains/<domain>.py` 配置。新领域 = 新建一个 DomainConfig 子类，底座层（LangGraph 图、记忆、HITL、多租户、SSE）零改动。

## 技术栈（本机落地的企业级能力）

| 层     | 技术                                                                | 本机形态                                      |
| ------ | ------------------------------------------------------------------- | --------------------------------------------- |
| 编排   | LangGraph（分级双路径 + HITL interrupt + 并行 fan-out）             | 纯 Python                                     |
| 向量库 | **Milvus**（多租户分区 + expr 过滤）                                | **Milvus Lite**（进程内，`./data/milvus.db`） |
| 推理   | **DeepSeek v4 API**（reasoner，核心推理）+ 本地 qwen2.5（轻量任务） | 无 key 自动降级本地 deepseek-r1               |
| 记忆   | 分层记忆：chat_message 三 ID 表（user/thread/reply）+ 按需加载      | SQLite（生产切 MySQL/Postgres）               |
| Skill  | 注册中心 + 三层筛选（权限→租户→语义）+ 对比分析 Skill               | 纯 Python                                     |
| 安全   | Prompt 注入检测 + 三级置信度 FactCheck                              | 内置                                          |
| SSE    | sse-starlette（提前推 token_stat + 心跳）+ 前端 SseClient           | 后端已实现                                    |
| 前端   | Vue3 + **lke-component-vue3**（MsgContent/MsgThought）              | Vite dev                                      |

## 快速启动

### 0. 前置（本机已具备）

- Redis（`redis-server`）
- Ollama（`qwen2.5:7b` / `deepseek-r1:7b` 已拉）
- Python 3.12 + uv

### 1. 后端

```bash
cd /Users/yuzhenhua/Desktop/agent-企业级电力项目
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r <(sed -n '/dependencies/,/\]/p' pyproject.toml | grep -E '"[a-z]"' | tr -d '",')
# 可选：配置 DeepSeek key（核心推理走云端，无 key 自动降级本地）
echo "DEEPSEEK_API_KEY=sk-xxx" >> .env
# 灌入演示文档（可选）
KMP_DUPLICATE_LIB_OK=TRUE .venv/bin/python scripts/seed_docs.py
# 启动
KMP_DUPLICATE_LIB_OK=TRUE .venv/bin/python main.py --port 8090
```

### 2. 前端

```bash
cd /Users/yuzhenhua/Desktop/agent-企业电力项目-vue
npm install
npm run dev   # http://localhost:5173（代理到 8090）
```

## 核心 API

| 端点                                       | 说明                                                 |
| ------------------------------------------ | ---------------------------------------------------- |
| `POST /chat`                               | 普通对话                                             |
| `POST /chat/stream`                        | **SSE 流式**（token_stat → thinking → reply → done） |
| `POST /chat/abort`                         | 取消生成                                             |
| `POST /chat/human-confirm`                 | HITL 人工确认恢复                                    |
| `POST /docs/upload`                        | 文档上传（异步入库）                                 |
| `GET /docs/list`                           | 文档列表                                             |
| `GET /metrics` / `GET /metrics/prometheus` | 指标                                                 |
| `GET /audit/chats`                         | 审计日志                                             |
| `GET /memory/threads/{id}`                 | 记忆回溯                                             |

## 目录结构

```
agent-企业级电力项目/
├── main.py                 # 启动入口（环境预检 + uvicorn）
├── config/
│   ├── settings.py         # 全局配置（含 DOMAIN 领域开关）
│   ├── domain.py           # ★ 领域配置基类 + 工厂（领域只在配置层）
│   └── domains/            # ★ 领域目录：power.py / generic.py
│       └── power.py        #   电力领域（意图词/提示词/Skills/演示文档）
├── agent/                  # LangGraph 主图 + Harness + Skills + 记忆
│   ├── graph.py            # 8 节点图（领域无关，从 DomainConfig 读意图）
│   ├── harness/            # 风险分级 + 高危拦截 + 人工确认审计
│   └── skills/             # 注册中心 + 动态筛选 + 领域 Skill
├── rag/                    # Milvus 适配层（Lite/生产双模式）+ 切片
├── llm/                    # UnifiedLLM + ModelRouter（DeepSeek/本地降级）
├── memory/                 # chat_message 三 ID 表 + 按需加载
├── gateway/                # 鉴权/限流/日志中间件
├── mq/                     # Redis transport 模拟 MQ（重试/死信）
├── observability/          # 追踪 + 指标 + 审计
└── api/main.py             # FastAPI + sse-starlette
```

## 面试亮点（本机可演示）

1. **领域无关底座**（压轴）：这套不是电力专用系统，是**通用文档分析底座**，电力只是第一个领域验证。换领域只需换 DomainConfig + 注册 Skill，底座（记忆/HITL/多租户/SSE）全部复用——已内置 generic 领域演示证明
2. **多租户隔离**：Milvus 分区 + expr 双保险，A 租户搜不到 B 租户
3. **SSE 提前推**：token_stat 在图启动前就推，TTFT 感知 ≈ 0
4. **模型分级**：意图/闲聊/校验走本地小模型，核心推理走 DeepSeek v4 API
5. **分层记忆**：按 user/thread/reply 三 ID 落库，推理前按需加载不溢出
6. **动态 Skill 筛选**：只注入 Top-K Skill 到 Prompt，上下文精简
7. **三级置信度**：FactCheck 区分 high/medium/low，敢说不确定
8. **HITL**：高危操作 interrupt 挂起，人工确认后恢复
