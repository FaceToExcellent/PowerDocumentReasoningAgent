"""电力 Agent 的 A2A 服务端 + 客户端 — 跨系统/跨 agent 委托(P4)

服务端:FastAPI 挂 JSON-RPC 端点(initialize / sendMessage / tasks/get / tasks/cancel),
  Agent Card 发布于 /.well-known/agent.json —— 外部系统/agent 可把电力问答任务委托进来。
客户端:delegate_to_agent() 用官方 a2a-sdk Client 把子任务委托给远程 A2A agent。

定位:项目内多 agent(LangGraph 子图)解决"一个进程内的分工",
  A2A 解决"跨进程/跨系统/跨团队的协作",与 MCP(agent↔工具)正交叠加。
"""
import asyncio
import logging
import uuid
from typing import Any, Dict

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# 进程内任务存储(task_id -> {status, result})
_TASKS: Dict[str, Dict[str, Any]] = {}


# ── Agent Card ─────────────────────────────────────
def build_agent_card() -> Dict[str, Any]:
    """电力 Agent 的 A2A Agent Card(声明能力,外部系统据此发现/委托)。"""
    return {
        "name": "电力智能运维 Agent",
        "description": "电力规程检索、造价核算、故障处置、设备对比分析、通用问答",
        "url": "/",
        "version": "1.0.0",
        "capabilities": {"streaming": False, "pushNotifications": False},
        "skills": [
            {"id": "power_qa", "name": "电力问答",
             "description": "处理电力领域问答,返回回答与引用", "tags": ["power", "qa"]}
        ],
    }


async def _run_power_task(message: str, tenant_id: str) -> str:
    """运行电力 agent,返回回复。"""
    from agent.graph import run_agent
    result = await run_agent(message, tenant_id=tenant_id, account="a2a")
    if result.get("success"):
        return result.get("reply", "")
    return f"处理失败: {result.get('error', '')}"


def _extract_text(message: Any) -> str:
    """从 A2A Message 或纯文本中提取文本。"""
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        parts = message.get("parts") or []
        if parts:
            return str(parts[0].get("text", "") if isinstance(parts[0], dict) else parts[0])
        return str(message.get("text", ""))
    return str(message)


# ── JSON-RPC 端点 ──────────────────────────────────
@router.post("/a2a")
async def a2a_endpoint(request: Request):
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"jsonrpc": "2.0", "id": None,
                             "error": {"code": -32700, "message": "解析错误"}})
    method = payload.get("method", "")
    params = payload.get("params", {}) or {}
    rid = payload.get("id")

    if method == "initialize":
        return {"jsonrpc": "2.0", "id": rid, "result": {
            "protocolVersion": "0.2.0", "capabilities": {"streaming": False}}}

    if method == "sendMessage":
        try:
            text = _extract_text(params.get("message", ""))
            if not text:
                return {"jsonrpc": "2.0", "id": rid,
                        "error": {"code": -32602, "message": "message 为空"}}
            task_id = f"task-{uuid.uuid4().hex[:8]}"
            _TASKS[task_id] = {"status": "working", "result": ""}
            reply = await _run_power_task(text, params.get("tenantId", "default"))
            _TASKS[task_id] = {"status": "completed", "result": reply}
            return {"jsonrpc": "2.0", "id": rid, "result": {
                "id": task_id, "status": {"state": "completed"},
                "artifacts": [{"name": "reply", "parts": [{"text": reply}]}]}}
        except Exception as e:
            logger.error(f"A2A sendMessage 失败: {e}")
            return {"jsonrpc": "2.0", "id": rid,
                    "error": {"code": -32000, "message": str(e)[:200]}}

    if method == "tasks/get":
        task = _TASKS.get(params.get("id", ""))
        if not task:
            return {"jsonrpc": "2.0", "id": rid,
                    "error": {"code": -32602, "message": "task not found"}}
        return {"jsonrpc": "2.0", "id": rid, "result": {
            "id": params.get("id", ""), "status": {"state": task["status"]},
            "artifacts": [{"name": "reply", "parts": [{"text": task["result"]}]}]
            if task["result"] else []}}

    if method == "tasks/cancel":
        task_id = params.get("id", "")
        _TASKS.pop(task_id, None)
        return {"jsonrpc": "2.0", "id": rid, "result": {
            "id": task_id, "status": {"state": "canceled"}}}

    return {"jsonrpc": "2.0", "id": rid,
            "error": {"code": -32601, "method": method, "message": "方法不存在"}}


@router.get("/.well-known/agent.json")
async def agent_card():
    return build_agent_card()


# ── A2A 客户端:委托子任务给远程 agent ───────────────
async def delegate_to_agent(card_url: str, message: str, timeout: float = 60.0) -> str:
    """把子任务委托给远程 A2A agent(官方 a2a-sdk Client),返回其回答。"""
    from a2a.client import create_client
    from a2a.types import Message, Part, Role
    client = await create_client(agent=card_url)
    task = await client.send_message(
        message=Message(role=Role.USER, parts=[Part(text=message)]))
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        state = getattr(getattr(task, "status", None), "state", None)
        if state in ("completed", "failed", "canceled"):
            break
        await asyncio.sleep(0.2)
        task = await client.get_task(task_id=task.id)
    for artifact in getattr(task, "artifacts", []) or []:
        for part in getattr(artifact, "parts", []) or []:
            if getattr(part, "text", None):
                return part.text
    return ""
