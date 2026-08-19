"""消息流水存储 — chat_message 三 ID 表（user_id / thread_id / reply_id）
本机用 SQLite 保证可跑；表结构与计划书 MySQL 版一致，生产可切。
核心设计：记忆独立于大模型存储，推理前按需读取，不放模型上下文。
"""
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from config.settings import settings
from config.logging_config import logger


# 从查询提取关键词(jieba 中文分词 + ASCII token),用于记忆语义召回
def _query_keywords(query: str, limit: int = 8) -> List[str]:
    import re
    import jieba
    kws = []
    for seg in jieba.cut(str(query or "")):
        seg = seg.strip()
        if len(seg) >= 2 and not seg.isspace():
            kws.append(seg)
    kws += re.findall(r"[a-zA-Z0-9]{2,}", str(query or ""))
    return list(dict.fromkeys(kws))[:limit]


# 消息流水存储：chat_message 三 ID 表读写
class MessageStore:
    # 初始化数据库路径、加锁并建表
    def __init__(self, db_path: str = None):
        self._db_path = db_path or settings.sqlite_audit_db
        self._lock = threading.Lock()
        self._init_db()

    # 初始化 SQLite 库，创建 chat_message 表及索引
    def _init_db(self):
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._lock, sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_message (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_id TEXT DEFAULT '',
                    user_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    reply_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_type TEXT DEFAULT 'text',
                    intent TEXT DEFAULT '',
                    tokens INTEGER DEFAULT 0,
                    created_at REAL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_thread "
                         "ON chat_message(tenant_id, user_id, thread_id, created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_reply "
                         "ON chat_message(tenant_id, thread_id, reply_id)")

    # ── 写 ──
    def add(self, *, tenant_id="", user_id="", thread_id="", reply_id="",
            role="", content="", content_type="text", intent="", tokens=0) -> int:
        with self._lock, sqlite3.connect(self._db_path) as conn:
            cur = conn.execute(
                "INSERT INTO chat_message (tenant_id,user_id,thread_id,reply_id,role,"
                "content,content_type,intent,tokens,created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (tenant_id, user_id, thread_id, reply_id, role, content,
                 content_type, intent, tokens, time.time()),
            )
            return cur.lastrowid

    # ── 读 ──
    def get_recent(self, *, tenant_id="", user_id="", thread_id="", limit=20) -> List[Dict]:
        """最近 N 轮消息（按时间倒序取，再正序返回）"""
        with self._lock, sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM chat_message WHERE tenant_id=? AND user_id=? AND thread_id=? "
                "ORDER BY created_at DESC LIMIT ?",
                (tenant_id, user_id, thread_id, limit),
            ).fetchall()
        msgs = [dict(r) for r in reversed(rows)]
        return msgs

    # 查询整条会话流水
    def get_thread(self, *, tenant_id="", user_id="", thread_id="") -> List[Dict]:
        """整条会话流水"""
        with self._lock, sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM chat_message WHERE tenant_id=? AND user_id=? AND thread_id=? "
                "ORDER BY created_at",
                (tenant_id, user_id, thread_id),
            ).fetchall()
        return [dict(r) for r in rows]

    # 按 reply_id 精确回溯多段消息
    def get_by_reply(self, *, tenant_id="", thread_id="", reply_id="") -> List[Dict]:
        """按 reply_id 精确回溯（thinking + 正文 + 工具调用多段）"""
        with self._lock, sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM chat_message WHERE tenant_id=? AND thread_id=? AND reply_id=? "
                "ORDER BY id",
                (tenant_id, thread_id, reply_id),
            ).fetchall()
        return [dict(r) for r in rows]

    # 轻量语义召回：按关键词匹配最近消息
    def semantic_search(self, *, tenant_id="", user_id="", query="", top_k=5) -> List[Dict]:
        """轻量语义召回：关键词匹配最近消息（本机无向量版；生产可接 Milvus 记忆向量）"""
        kws = _query_keywords(query)
        if not kws:
            return []
        cond = " OR ".join(["content LIKE ?"] * len(kws))
        params = [f"%{k}%" for k in kws]
        with self._lock, sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"SELECT * FROM chat_message WHERE tenant_id=? AND user_id=? AND ({cond}) "
                f"AND role='assistant' ORDER BY created_at DESC LIMIT ?",
                (tenant_id, user_id, *params, top_k),
            ).fetchall()
        return [dict(r) for r in rows]


message_store = MessageStore()
