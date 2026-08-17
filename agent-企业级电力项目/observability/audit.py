"""结构化审计日志 — 双写：loguru audit.log（结构化 JSON）+ SQLite audit.db
本机用 SQLite 保证可跑，生产可换 MySQL。
"""
import json
import sqlite3
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from config.settings import settings
from config.logging_config import logger
from observability.tracing import get_trace_id

_audit_logger = logger.bind(audit=True)


# 审计日志双写器：结构化 JSON 落盘 + SQLite 入库
class AuditLogger:
    # 初始化：加载数据库路径、加锁并建表
    def __init__(self):
        self._db_path = settings.sqlite_audit_db
        self._lock = threading.Lock()
        self._init_db()

    # 初始化 SQLite 库，创建审计相关三张表
    def _init_db(self):
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._lock, sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_chat (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT, tenant_id TEXT, user_id TEXT, thread_id TEXT,
                    intent TEXT, user_input TEXT, agent_output TEXT,
                    confidence REAL, fact_check_passed INTEGER, duration_ms INTEGER,
                    cache_hit INTEGER, success INTEGER, error TEXT,
                    created_at TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_tool (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT, tenant_id TEXT, tool_name TEXT,
                    params TEXT, result TEXT, success INTEGER,
                    duration_ms INTEGER, created_at TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_human (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT, thread_id TEXT, user_id TEXT, skill_name TEXT,
                    risk_level TEXT, action TEXT, params TEXT, reason TEXT,
                    created_at TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_agent (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT, tenant_id TEXT, thread_id TEXT,
                    agent TEXT, user_input TEXT, agent_output TEXT,
                    fact_check_passed INTEGER, confidence_level TEXT,
                    tool_calls INTEGER, citations_count INTEGER, iteration INTEGER,
                    created_at TEXT
                )
            """)

    # 通用插入：向指定审计表写入一条记录
    def _insert(self, table: str, data: dict):
        try:
            with self._lock, sqlite3.connect(self._db_path) as conn:
                cols = ",".join(data.keys())
                ph = ",".join("?" for _ in data)
                conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({ph})", tuple(data.values()))
        except Exception as e:
            logger.error(f"审计写入失败: {e}")

    # ── 对话审计 ──
    def log_chat(self, *, tenant_id="", user_id="", thread_id="", intent="",
                 user_input="", agent_output="", confidence=0.0, fact_check_passed=True,
                 duration_ms=0, cache_hit=False, success=True, error=""):
        rec = {
            "trace_id": get_trace_id(), "tenant_id": tenant_id, "user_id": user_id,
            "thread_id": thread_id, "intent": intent, "user_input": user_input[:2000],
            "agent_output": (agent_output or "")[:4000], "confidence": confidence,
            "fact_check_passed": int(fact_check_passed), "duration_ms": duration_ms,
            "cache_hit": int(cache_hit), "success": int(success), "error": error[:500],
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        _audit_logger.info(json.dumps({"event": "chat", **rec}, ensure_ascii=False))
        self._insert("audit_chat", rec)

    # ── 工具调用审计 ──
    def log_tool(self, *, tenant_id="", tool_name="", params=None, result=None,
                 success=True, duration_ms=0):
        rec = {
            "trace_id": get_trace_id(), "tenant_id": tenant_id, "tool_name": tool_name,
            "params": json.dumps(params or {}, ensure_ascii=False)[:1000],
            "result": json.dumps(result or {}, ensure_ascii=False)[:2000],
            "success": int(success), "duration_ms": duration_ms,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        _audit_logger.info(json.dumps({"event": "tool", **rec}, ensure_ascii=False))
        self._insert("audit_tool", rec)

    # ── 人工干预审计 ──
    def log_human(self, *, thread_id="", user_id="", skill_name="", risk_level="",
                  action="", params=None, reason=""):
        rec = {
            "trace_id": get_trace_id(), "thread_id": thread_id, "user_id": user_id,
            "skill_name": skill_name, "risk_level": risk_level, "action": action,
            "params": json.dumps(params or {}, ensure_ascii=False)[:1000],
            "reason": reason[:500], "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        _audit_logger.warning(json.dumps({"event": "human", **rec}, ensure_ascii=False))
        self._insert("audit_human", rec)

    # ── 安全事件审计 ──
    def log_security(self, *, event_type="", detail=""):
        rec = {
            "trace_id": get_trace_id(), "event_type": event_type, "detail": detail[:500],
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        _audit_logger.error(json.dumps({"event": "security", **rec}, ensure_ascii=False))

    # ── Agent 级审计(项目内多 agent:记录每个子图 agent 的一次执行)──
    def log_agent(self, *, tenant_id="", thread_id="", agent="", user_input="",
                  agent_output="", fact_check_passed=True, confidence_level="high",
                  tool_calls=0, citations_count=0, iteration=0):
        rec = {
            "trace_id": get_trace_id(), "tenant_id": tenant_id, "thread_id": thread_id,
            "agent": agent, "user_input": user_input[:2000],
            "agent_output": (agent_output or "")[:4000],
            "fact_check_passed": int(fact_check_passed), "confidence_level": confidence_level,
            "tool_calls": int(tool_calls), "citations_count": int(citations_count),
            "iteration": int(iteration),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        _audit_logger.info(json.dumps({"event": "agent", **rec}, ensure_ascii=False))
        self._insert("audit_agent", rec)

    # 查询审计表，返回最近 limit 条记录
    def query(self, table: str, limit: int = 20) -> list:
        try:
            with self._lock, sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    f"SELECT * FROM {table} ORDER BY id DESC LIMIT ?", (limit,)
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"审计查询失败: {e}")
            return []


audit_logger = AuditLogger()
