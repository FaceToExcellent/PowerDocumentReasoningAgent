"""Redis 缓存封装（企业版）"""
import json
import logging
from typing import Optional, Any

import redis.asyncio as aioredis

from config.settings import settings

logger = logging.getLogger(__name__)


# 基于 Redis 的缓存服务，TTL 自动过期，Redis 不可用时降级为内存缓存
class CacheService:
    """基于 Redis 的缓存服务，支持 TTL 自动过期。Redis 不可用时自动降级为内存缓存。"""

    # 初始化缓存服务，读取默认 TTL 并准备内存降级容器
    def __init__(self):
        self.redis: Optional[aioredis.Redis] = None
        self.default_ttl = settings.cache_default_ttl
        self._memory: dict = {}          # Redis 不可用时的内存降级
        self._connected = False

    # 连接 Redis 并校验连通性，失败时降级为内存缓存
    async def connect(self):
        try:
            self.redis = aioredis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                max_connections=settings.redis_max_connections,
                decode_responses=True,
            )
            await self.redis.ping()
            self._connected = True
            logger.info("✅ Redis 缓存连接成功")
        except Exception as e:
            logger.warning(f"⚠️ Redis 连接失败，降级为内存缓存: {e}")
            self.redis = None
            self._connected = False

    # 按 key 读取缓存值，Redis 不可用时走内存降级
    async def get(self, key: str) -> Optional[Any]:
        if self.redis:
            try:
                value = await self.redis.get(key)
                return json.loads(value) if value else None
            except Exception:
                return None
        # 内存降级
        item = self._memory.get(key)
        if item and item["expire"] > 0:
            return item["value"]
        return None

    # 写入缓存并设置 TTL，Redis 失败时降级为内存存储
    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        ttl = ttl or self.default_ttl
        if self.redis:
            try:
                await self.redis.set(key, json.dumps(value, ensure_ascii=False, default=str), ex=ttl)
                return True
            except Exception as e:
                logger.error(f"缓存写入失败 [{key}]: {e}")
                return False
        # 内存降级（简化：无真实 TTL 清理，够开发用）
        import time
        self._memory[key] = {"value": value, "expire": time.time() + ttl if ttl else 0}
        return True

    # 删除指定 key 的缓存，内存降级模式同样清理
    async def delete(self, key: str):
        if self.redis:
            await self.redis.delete(key)
        else:
            self._memory.pop(key, None)

    # 原子自增计数（限流用），Redis 不可用时用内存计数
    async def incr(self, key: str, amount: int = 1) -> int:
        """原子自增（限流用）。Redis 不可用时用内存计数。"""
        if self.redis:
            try:
                return await self.redis.incr(key)
            except Exception:
                return 0
        self._memory[key] = self._memory.get(key, 0) + amount
        return self._memory[key]

    # 为指定 key 设置过期时间（仅 Redis 模式）
    async def expire(self, key: str, seconds: int):
        if self.redis:
            await self.redis.expire(key, seconds)

    # 健康检查：探测 Redis 连接是否可用
    async def health_check(self) -> bool:
        if not self.redis:
            return False
        try:
            return await self.redis.ping()
        except Exception:
            return False


cache_service = CacheService()
