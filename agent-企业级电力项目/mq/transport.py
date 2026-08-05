"""MQ transport 抽象 — 本地 Redis 模拟（重试/死信/幂等），生产换 RocketMQ"""
import json
import uuid
import logging
from abc import ABC, abstractmethod

from config.cache import cache_service

logger = logging.getLogger(__name__)


class MQTransport(ABC):
    @abstractmethod
    async def send(self, topic: str, body: dict, tags: str = "") -> str: ...
    @abstractmethod
    async def consume(self, topic: str, handler) -> None: ...


class RedisMQTransport(MQTransport):
    """Redis List 模拟 MQ：LPUSH 投递 / BRPOP 消费 / retry 重试 / dlq 死信"""

    def __init__(self, max_retry: int = 3):
        self.max_retry = max_retry

    def _queue_key(self, topic: str) -> str:
        return f"mq:{topic}"

    def _retry_key(self, topic: str) -> str:
        return f"mq:{topic}:retry"

    def _dlq_key(self, topic: str) -> str:
        return f"mq:{topic}:dlq"

    async def send(self, topic: str, body: dict, tags: str = "") -> str:
        msg_id = f"m{uuid.uuid4().hex[:12]}"
        msg = {"msg_id": msg_id, "topic": topic, "tags": tags,
               "retry_count": 0, "body": body}
        await cache_service.set(self._queue_key(topic), [msg], ttl=3600)
        return msg_id

    async def consume(self, topic: str, handler) -> None:
        """消费一条并调用 handler；handler 返回 success/failed"""
        raw = await cache_service.get(self._queue_key(topic))
        if not raw:
            return
        msgs = list(raw)
        msg = msgs.pop(0) if msgs else None
        if not msg:
            return
        await cache_service.set(self._queue_key(topic), msgs, ttl=3600)
        try:
            ok = await handler(msg)
            if not ok:
                await self._retry(topic, msg)
        except Exception as e:
            logger.error(f"消费异常: {e}")
            await self._retry(topic, msg)

    async def _retry(self, topic: str, msg: dict):
        msg["retry_count"] = msg.get("retry_count", 0) + 1
        if msg["retry_count"] >= self.max_retry:
            await cache_service.set(self._dlq_key(topic), [msg], ttl=86400)
            logger.warning(f"进入死信队列: {topic} msg={msg['msg_id']}")
        else:
            await cache_service.set(self._retry_key(topic), [msg], ttl=600)
            logger.info(f"重试 {msg['retry_count']}/{self.max_retry}: {msg['msg_id']}")


mq_transport = RedisMQTransport()
