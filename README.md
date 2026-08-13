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
| **power**（电力）   | 规程检索/造价/故障/对比 | power_rag / quota_match / comparison_analysis / fact_check | DL/T 572 等 4 篇            |
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
cd /Users/yuzhenhua/Desktop/PowerDocumentReasoningAgent/agent-企业级电力项目
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r <(sed -n '/dependencies/,/\]/p' pyproject.toml | grep -E '"[a-z]"' | tr -d '",')
# 可选：配置 DeepSeek key（核心推理走云端，无 key 自动降级本地）
echo "DEEPSEEK_API_KEY=sk-xxx" >> .env
# 灌入演示文档（可选）
uv run python main.py --seed
# 启动（main.py 已内置 KMP_DUPLICATE_LIB_OK，无需手动加前缀）
uv run python main.py --port 8090
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
| `GET /health`                             | 健康检查（Redis / 向量库 / DeepSeek 配置）          |
| `POST /chat`                              | 普通对话                                             |
| `POST /chat/stream`                       | **SSE 流式**（token_stat → thinking → reply → done；HITL 挂起时推 human_confirm） |
| `POST /chat/abort`                        | 取消生成                                             |
| `POST /chat/human-confirm`                | HITL 人工确认恢复（含 resume_token / 幂等键）        |
| `POST /chat/recover`                      | 崩溃恢复：从 checkpoint 回放未完成线程               |
| `POST /docs/upload`                       | 文档上传（切分入库，返回 chunk 数）                  |
| `GET /docs/list`                          | 文档列表                                             |
| `POST /eval/run`                          | 固定 case 回归评测（cases.yml）                      |
| `GET /metrics` / `GET /metrics/prometheus`| 指标（后者为 Prometheus 文本格式）                   |
| `GET /audit/chats`                        | 审计日志                                             |
| `GET /memory/threads/{id}`                | 记忆回溯                                             |

## 目录结构

```
agent-企业级电力项目/
├── main.py                 # 启动入口（环境预检 Redis/Ollama/Milvus/DeepSeek + uvicorn）
├── pyproject.toml          # 依赖声明（uv）
├── config/
│   ├── settings.py         # 全局配置（含 DOMAIN 领域开关）
│   ├── domain.py           # ★ 领域配置基类 + 工厂（领域只在配置层）
│   ├── cache.py            # Redis 缓存（失败降级内存）
│   ├── logging_config.py   # loguru 日志
│   └── domains/            # ★ 领域目录：power.py / generic.py
├── api/main.py             # FastAPI + sse-starlette（SSE/HITL/文档上传/评测）
├── agent/                  # LangGraph 主图 + Harness + Skills + 记忆
│   ├── graph.py            # 8 节点图（领域无关，从 DomainConfig 读意图）
│   ├── state.py            # AgentState 共享状态
│   ├── checkpointer.py     # AsyncSqliteSaver（checkpoint / 崩溃恢复）
│   ├── context_builder.py  # 多来源上下文构建（信任排序/冲突解决/压缩）
│   ├── context_manager.py  # recent_rounds 短上下文
│   ├── clarification.py    # 工具澄清（缺参数追问）
│   ├── degradation.py      # 错误降级策略
│   ├── fact_checker.py     # 四层幻觉抑制 + 三级置信度
│   ├── hooks.py            # Skill 执行前后钩子（治理/脱敏）
│   ├── rag_cache.py        # RAG 缓存（索引版本指纹）
│   ├── data_tools.py       # SQLite 数据工具（带租户）
│   ├── execution_log.py    # 执行日志 → 审计
│   ├── harness/            # 风险分级 + 高危拦截 + 人工确认审计
│   ├── prompts/            # Supervisor 提示词
│   └── skills/             # 注册中心 + 三层筛选 + 领域 Skill
├── rag/                    # 混合检索（向量+关键词+RRF）+ 切片 + 重排
│   ├── retriever.py        # RAGService.search 统一入口
│   ├── doc_splitter.py     # 标题层级感知切片
│   ├── embedder.py         # BGE-M3 embedding（MPS/懒加载/预热）
│   ├── reranker.py         # BGE-reranker 重排（无则本地兜底）
│   ├── query_expander.py   # 查询改写（L13）
│   ├── metadata_filter.py  # metadata 过滤（intent/电压等级）
│   ├── kg/entity_index.py  # 实体索引（一跳关系 → 证据）
│   └── vector_store/       # base / milvus（多租户分区+expr）/ chroma
├── llm/                    # UnifiedLLM + ModelRouter（DeepSeek/本地降级）
│   └── backends/           # deepseek / local_reasoning / local_small
├── memory/                 # chat_message 三 ID 表 + 按需加载
├── gateway/                # 鉴权（APIKey/JWT）/ 限流 / 日志中间件
├── safety/prompt_guard.py  # Prompt 注入检测（分源扫描/脱敏）
├── mq/                     # Redis transport 模拟 MQ（重试/死信）
├── mcp_catalog/            # 从 Skill 生成 MCP 工具定义
├── mcp_servers/            # MCP 服务
├── observability/          # 追踪（span）/ 指标（Prometheus）/ 审计（SQLite）
├── eval/                   # 固定 case 回归评测（cases.yml + runner）
├── scripts/seed_docs.py    # 演示文档灌入
├── tests/smoke_test.py     # 冒烟测试
└── db/ data/ logs/         # SQLite / 向量库 / 日志数据目录
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
9. **崩溃恢复 + 幂等**：checkpoint 断点回放续跑未完成线程（`/chat/recover`），HITL 恢复带幂等键防重复提交
