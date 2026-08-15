"""FastAPI 接口 — 企业版 v5.0：网关中间件 + SSE 流式（sse-starlette）+ HITL + 文档异步"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional, AsyncIterator

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config.settings import settings
from config.cache import cache_service
from config.logging_config import setup_logging, logger

setup_logging()
logger = logging.getLogger(__name__)

# ── 生命周期 ──────────────────────────────────────
# 应用生命周期:启动时初始化 OTel/缓存/图,关闭时清理资源
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("⚡ 电力智能运维 Agent v5.0（企业版）启动中…")
    # OTel 初始化（OTLP → Jaeger/Tempo；未配后端时 BatchSpanProcessor 静默丢弃，不影响服务）
    from observability.otel_setup import setup_otel, shutdown_otel
    setup_otel(service_name="power-agent")
    await cache_service.connect()
    # 初始化官方 AsyncSqliteSaver + 编译图（必须在 event loop 内）
    from agent.graph import init_graph
    await init_graph()
    # KG 种子:空库时灌入人工维护的三元组基线,避免 _build_kg_evidence 空转
    from rag.kg.entity_index import entity_index
    entity_index.seed()
    # 预热 RAG 向量库 + 注册 Skills
    try:
        from rag.retriever import rag_service
        logger.info(f"✅ 向量库就绪: {settings.vector_store_type} "
                    f"(doc_count={rag_service.count()})")
    except Exception as e:
        logger.warning(f"向量库未就绪: {e}")
    # 预热 Embedding 模型（BGE-M3，首次加载 ~6s；to_thread 避免阻塞事件循环）
    try:
        from rag.embedder import embedding_provider
        await asyncio.to_thread(embedding_provider._get)
        logger.info(f"✅ Embedding 模型预热完成: {settings.embedding_model} "
                    f"(device={settings.embedding_device})")
    except Exception as e:
        logger.warning(f"Embedding 预热失败（懒加载兜底）: {e}")
    from agent.skills.bootstrap import skill_registry
    logger.info(f"✅ Skills 已注册: {skill_registry.all_names()}")
    logger.info(f"✅ API 就绪: http://{settings.api_host}:{settings.api_port}")
    yield
    from observability.otel_setup import shutdown_otel
    shutdown_otel()
    from agent.checkpointer import close_checkpointer
    await close_checkpointer()
    logger.info("🛑 API 关闭")


app = FastAPI(title="电力智能运维 Agent（企业版）", version="5.0.0", lifespan=lifespan)

# CORS（放在网关中间件最外层，先处理 OPTIONS）
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

# 网关中间件（顺序：日志 → 鉴权 → 限流）
from gateway.logging import RequestLoggingMiddleware
from gateway.auth import AuthMiddleware
from gateway.rate_limit import RateLimitMiddleware
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(AuthMiddleware)
app.add_middleware(RateLimitMiddleware)


# ── 数据模型 ──────────────────────────────────────
# 聊天请求体:消息内容与租户/用户等上下文信息
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    thread_id: str = ""
    account: str = "anonymous"
    employee_id: str = ""
    tenant_id: str = "default"


# HITL 人工确认请求体:审批动作与恢复凭证
class HumanConfirmRequest(BaseModel):
    thread_id: str
    action: str = "approve"       # approve / reject / modify
    reason: str = ""
    modified_params: dict = None
    resume_token: str = ""        # 恢复凭证(防冒用)
    idempotency_key: str = ""     # 幂等键(防重复提交)


# ── 健康检查 ──────────────────────────────────────
# 健康检查接口:返回 Redis/向量库/模型配置状态
@app.get("/health")
async def health():
    redis_ok = await cache_service.health_check()
    return {"status": "ok", "redis": redis_ok,
            "vector_store": settings.vector_store_type,
            "deepseek_api": bool(settings.deepseek_api_key)}


# ── 普通对话 ──────────────────────────────────────
# 普通对话接口:调用 Agent 图并返回结果
@app.post("/chat")
async def chat(req: ChatRequest, request: Request = None):
    from agent.graph import run_agent
    tenant = getattr(request.state, "tenant_id", req.tenant_id)
    user_id = getattr(request.state, "user_id", req.account)
    result = await run_agent(
        user_input=req.message, thread_id=req.thread_id or None,
        account=req.account, employee_id=req.employee_id,
        tenant_id=tenant, user_id=user_id,
    )
    return {"success": result.get("success", False), "data": result}


# ── SSE 流式对话（sse-starlette，提前推 token_stat）──
# SSE 流式对话接口:边跑图边推送 token/回复等事件
@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request = None):
    from sse_starlette.sse import EventSourceResponse, ServerSentEvent
    from agent.graph import agent_graph, STREAM_QUEUE, _ABORT_EVENT, create_initial_state
    from agent.state import create_initial_state

    thread_id = req.thread_id or f"th-{uuid.uuid4().hex[:8]}"
    tenant = getattr(request.state, "tenant_id", req.tenant_id)
    user_id = getattr(request.state, "user_id", req.account)
    abort_event = asyncio.Event()
    _ABORT_EVENT[thread_id] = abort_event

    # SSE 事件生成器:推送 token_stat/图完成等事件
    async def event_generator():
        token_queue: asyncio.Queue = asyncio.Queue()
        token_ctx = STREAM_QUEUE.set(token_queue)

        # ⭐ 提前 SSE：图还没跑就先推状态，TTFT 感知 ≈ 0
        yield ServerSentEvent(event="token_stat",
                              data=json.dumps({"status_summary_title": "正在分析意图…"},
                                              ensure_ascii=False))
        try:
            async for evt in _run_graph_stream(thread_id, req, tenant, user_id, token_queue,
                                               abort_event):
                yield evt
        except asyncio.CancelledError:
            abort_event.set()
            raise
        finally:
            STREAM_QUEUE.reset(token_ctx)

    return EventSourceResponse(event_generator(), ping=15)


# 流式执行 Agent 图并消费 token 队列,产出 SSE 事件
async def _run_graph_stream(thread_id, req, tenant, user_id, token_queue, abort_event):
    from agent.state import create_initial_state
    from agent.graph import _ensure_graph, STREAM_QUEUE, _ABORT_EVENT
    from langgraph.errors import NodeCancelledError
    from sse_starlette.sse import ServerSentEvent

    agent_graph = await _ensure_graph()

    state = create_initial_state(
        thread_id=thread_id, user_id=user_id, account=req.account,
        employee_id=req.employee_id, tenant_id=tenant, user_input=req.message,
    )
    state["start_time"] = time.time()
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

    final_state = {}
    graph_error = None
    queue_done = False

    # 后台 task：跑图，节点完成写 token_queue
    async def _run():
        nonlocal final_state, graph_error
        try:
            async for update in agent_graph.astream(state, config, stream_mode="updates"):
                if abort_event.is_set():
                    logger.info(f"⏹ 线程 {thread_id} 被中止（前端取消）")
                    break
                if "__interrupt__" in update:
                    # HITL：图挂起等待人工确认，通知主循环后退出
                    logger.info(f"⏸ [HITL] 线程 {thread_id} 在节点执行处挂起，等待人工确认")
                    await token_queue.put({"type": "hitl_interrupt", "data": {}})
                    break
                for node_name, node_data in update.items():
                    final_state.update(node_data)
                    logger.info(f"  ✅ 节点 [{node_name}] 完成")
                    await token_queue.put({"type": "token_stat",
                                           "data": {"status_summary_title":
                                                    f"节点 {node_name} 完成"}})
        except (asyncio.CancelledError, NodeCancelledError):
            logger.info(f"⏹ 线程 {thread_id} 被取消")
        except Exception as e:
            graph_error = e
            logger.error(f"❌ 图执行异常终止: {str(e)[:200]}")
        finally:
            await token_queue.put({"type": "graph_done", "data": {}})

    graph_task = asyncio.create_task(_run())

    # ⭐ 主循环：边跑图边消费 token_queue → SSE 事件
    try:
        while True:
            try:
                evt = await asyncio.wait_for(token_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                if graph_task.done() and queue_done:
                    break
                if abort_event.is_set():
                    break
                continue
            typ = evt["type"]
            data = evt["data"]
            if typ == "graph_done":
                queue_done = True
                if graph_task.done():
                    break
            elif typ == "hitl_interrupt":
                # HITL：图已挂起，等待队列收尾后统一由下方 get_state 检测发 human_confirm
                pass
            elif typ == "token_stat":
                yield ServerSentEvent(event="token_stat",
                                      data=json.dumps(data, ensure_ascii=False))
            elif typ == "thinking":
                yield ServerSentEvent(event="thinking",
                                      data=json.dumps(data, ensure_ascii=False))
            elif typ == "reply":
                yield ServerSentEvent(event="reply",
                                      data=json.dumps(data, ensure_ascii=False))
    finally:
        if not graph_task.done():
            graph_task.cancel()

    # ⭐ HITL：图因 interrupt 挂起时，推送 human_confirm 事件，不发 done
    try:
        snapshot = await agent_graph.aget_state(config)
        interrupts = getattr(snapshot, "interrupts", ()) or ()
        if interrupts:
            review = interrupts[0]
            value = getattr(review, "value", review) or {}
            logger.info(f"[HITL] 高危操作挂起等待人工确认: thread={thread_id}")
            yield ServerSentEvent(event="human_confirm",
                                  data=json.dumps(value, ensure_ascii=False))
            return
    except Exception as e:
        logger.warning(f"[HITL] 检测 interrupt 失败: {e}")

    # 最终 done 事件
    yield ServerSentEvent(event="done", data=json.dumps({
        "reply": final_state.get("agent_output", ""),
        "intent": final_state.get("intent", ""),
        "confidence": final_state.get("confidence", 0),
        "fact_check_passed": final_state.get("fact_check_passed", True),
        "duration_ms": int((time.time() - state.get("start_time", time.time())) * 1000),
        "citations": final_state.get("citations", []),
        "cost_summary": final_state.get("cost_summary"),
    }, ensure_ascii=False))


# ── 取消 ──────────────────────────────────────────
# 取消对话接口:置位中止事件通知图执行停止
@app.post("/chat/abort")
async def abort_chat(req: ChatRequest):
    from agent.graph import _ABORT_EVENT
    evt = _ABORT_EVENT.get(req.thread_id)
    if evt:
        evt.set()
    return {"success": True}


# ── HITL 人工确认恢复 ─────────────────────────────
# HITL 人工确认接口:校验凭证并恢复高危操作
@app.post("/chat/human-confirm")
async def human_confirm(req: HumanConfirmRequest):
    from agent.graph import resume_agent
    # ⭐ resume_token 校验:凭 thread_id 不能恢复高危操作,必须有恢复凭证
    if req.resume_token and len(req.resume_token) < 8:
        return {"success": False, "error": "resume_token 不合法"}
    result = await resume_agent(
        req.thread_id,
        {"action": req.action, "reason": req.reason,
         "modified_params": req.modified_params,
         "resume_token": req.resume_token,
         "idempotency_key": req.idempotency_key},
    )
    return {"success": result.get("success", False), "data": result}


# ── 崩溃恢复：进程 crash 后从 checkpoint 重新拉起未完成线程 ──
# 崩溃恢复接口:从 checkpoint 重新拉起未完成线程
@app.post("/chat/recover")
async def chat_recover(req: HumanConfirmRequest):
    from agent.graph import recover_thread
    result = await recover_thread(
        req.thread_id,
        {"action": req.action or "reject", "resume_token": req.resume_token,
         "idempotency_key": req.idempotency_key},
    )
    return {"success": result.get("success", False), "data": result}


# ── 文档上传（异步，MQ 消费）──────────────────────
# 文档上传接口:切分入库并返回处理结果
@app.post("/docs/upload")
async def upload_doc(file: UploadFile = File(...), tenant_id: str = Header("default")):
    """上传文档：保存文件 → 发 MQ 消息 → 立即返回处理中"""
    content = await file.read()
    text = content.decode("utf-8", errors="ignore")
    file_name = file.filename or "doc.txt"

    from rag.doc_splitter import split_document
    docs = split_document(text, source=file_name, title=file_name.split(".")[0])
    from rag.retriever import rag_service
    n = rag_service.add_documents(docs, tenant_id=tenant_id)
    logger.info(f"文档 {file_name} 入库 {n} 条 (tenant={tenant_id})")
    return {"success": True, "file_id": f"f{uuid.uuid4().hex[:8]}",
            "chunks": n, "status": "completed", "msg": "文档已入库"}


# 文档列表接口:按租户返回已入库文档
@app.get("/docs/list")
async def list_docs(tenant_id: str = Header("default")):
    from rag.retriever import rag_service
    return {"success": True, "count": rag_service.count(tenant_id),
            "docs": rag_service.query(tenant_id=tenant_id, limit=20)}


# ── 指标 / 审计 / 记忆 ────────────────────────────
# 指标快照接口:返回当前运行指标
@app.get("/metrics")
async def metrics_endpoint():
    from observability.metrics import metrics
    return metrics.snapshot()


# Prometheus 指标接口:输出文本格式指标
@app.get("/metrics/prometheus")
async def metrics_prometheus():
    from observability.metrics import metrics
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(metrics.to_prometheus_text())


# 审计记录查询接口:返回最近聊天审计日志
@app.get("/audit/chats")
async def audit_chats(limit: int = 20):
    from observability.audit import audit_logger
    return {"success": True, "data": audit_logger.query("audit_chat", limit=limit)}


# 获取线程记忆接口:返回指定线程的消息记录
@app.get("/memory/threads/{thread_id}")
async def get_memory(thread_id: str):
    from memory.message_store import message_store
    msgs = message_store.get_thread(tenant_id="default", user_id="", thread_id=thread_id)
    return {"success": True, "count": len(msgs), "messages": msgs}


# ── 回归评测(证据治理层) ──────────────────────────
# 回归评测接口:运行固定 case 并返回评测报告
@app.post("/eval/run")
async def eval_run(body: dict = None):
    """固定 case 回归评测 — 每次改 Prompt/检索/模型后必跑"""
    from eval.runner import run_eval
    case_ids = (body or {}).get("case_ids")
    report = await run_eval(case_ids)
    return {"code": 0, "msg": "success", "data": report}
