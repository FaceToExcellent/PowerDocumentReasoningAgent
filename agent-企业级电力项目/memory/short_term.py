"""短期会话记忆 — Redis 最近 N 轮（轻量；生产可接 Mem0 摘要）"""
import json
import logging
from typing import List, Dict

from config.cache import cache_service

logger = logging.getLogger(__name__)

_MAX_ROUNDS = 10


# 短期会话记忆：Redis 最近 N 轮，天然 TTL
class ShortTermMemory:
    """基于 Redis 的短期记忆：当前 thread 最近 N 轮，天然 TTL"""

    # 构造 Redis 缓存键
    def _key(self, tenant_id: str, thread_id: str) -> str:
        return f"stm:{tenant_id}:{thread_id}"

    # 写入一轮消息，只保留最近 N 轮
    async def push(self, *, tenant_id="", thread_id="", role="", content="", ttl=3600):
        key = self._key(tenant_id, thread_id)
        rounds = await cache_service.get(key) or []
        rounds.append({"role": role, "content": content})
        # 只保留最近 N 轮
        rounds = rounds[-_MAX_ROUNDS:]
        await cache_service.set(key, rounds, ttl=ttl)

    # 读取当前线程的短期记忆列表
    async def get(self, *, tenant_id="", thread_id="") -> List[Dict]:
        return await cache_service.get(self._key(tenant_id, thread_id)) or []

    # 清空当前线程的短期记忆
    async def clear(self, *, tenant_id="", thread_id=""):
        await cache_service.delete(self._key(tenant_id, thread_id))


short_term_memory = ShortTermMemory()
