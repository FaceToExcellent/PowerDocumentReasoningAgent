"""LangGraph Checkpointer — 官方 langgraph-checkpoint-sqlite AsyncSqliteSaver

用官方 AsyncSqliteSaver 替换自研 SqliteSaver，获得：
  - WAL 模式 + 单锁，多 worker 并发写更稳
  - 按 channel 的版本管理（get_next_version / new_versions）
  - 完整的 alist / delete_thread / adelete_thread 语义
  - JsonPlusSerializer 类型安全序列化（不再 default=str 静默转字符串）

⚠️ 约束（官方 AsyncSqliteSaver 设计）：
  1. __init__ 必须在运行中的 event loop 内调用（内部取 asyncio.get_running_loop()）
  2. 同步 get_tuple/list/put 在主 loop 线程会抛 InvalidStateError；
     必须用 await checkpointer.aget_tuple(...) / aput(...)
  3. 不能直接用 from_conn_string().__aenter__()（async generator 被 GC 会关闭连接），
     这里改为手动创建 aiosqlite 连接并持有，生命周期由 init/close 控制
"""
import asyncio
import logging
from pathlib import Path

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from config.settings import settings

logger = logging.getLogger(__name__)

checkpointer: AsyncSqliteSaver = None  # type: ignore[assignment]


# 在运行中的 event loop 内创建并初始化官方 AsyncSqliteSaver
async def init_checkpointer() -> AsyncSqliteSaver:
    """在运行中的 event loop 内创建官方 AsyncSqliteSaver"""
    global checkpointer
    if checkpointer is None:
        Path(settings.checkpoint_db).parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(settings.checkpoint_db)
        checkpointer = AsyncSqliteSaver(conn)
        # 立即建 schema：避免懒初始化遇到旧库/空库时中途抛 no such column 类错误
        await checkpointer.setup()
        logger.info(f"✅ 官方 AsyncSqliteSaver 就绪: {settings.checkpoint_db}")
    return checkpointer


# 关闭 checkpointer 连接并置空
async def close_checkpointer() -> None:
    global checkpointer
    if checkpointer is not None:
        try:
            await checkpointer.conn.close()
        finally:
            checkpointer = None
