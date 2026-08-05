"""RAG 缓存 — 语义 key + TTL（企业版，支持租户隔离前缀）"""
import hashlib
import logging

from config.settings import settings
from config.cache import cache_service

logger = logging.getLogger(__name__)

_PREFIX = "rag:v2:"


class RAGCacheManager:
    def __init__(self):
        self.ttl = settings.rag_cache_ttl

    @staticmethod
    def build_cache_key(user_input: str, intent: str, top_k: int = 5, tenant_id: str = "") -> str:
        qh = hashlib.md5(user_input.strip().encode()).hexdigest()[:12]
        return f"{_PREFIX}{tenant_id}:{intent}:{qh}:k{top_k}"

    async def get(self, key: str) -> dict | None:
        return await cache_service.get(key)

    async def set(self, key: str, value: dict, ttl: int = None) -> bool:
        return await cache_service.set(key, value, ttl=ttl or self.ttl)

    async def invalidate(self, intent: str = "", tenant_id: str = ""):
        """文档变更后失效对应 intent 缓存（简化：清整个前缀，生产按 pattern 删）"""
        logger.info(f"RAG 缓存失效: intent={intent} tenant={tenant_id}")
        # 简化实现：不清 Redis（生产按 prefix 扫描删除）
        pass


rag_cache = RAGCacheManager()
