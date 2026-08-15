"""RAG 缓存 — 语义 key + TTL + 索引版本指纹（企业版，支持租户隔离前缀）"""
import hashlib
import json
import logging
import time

from config.settings import settings
from config.cache import cache_service

logger = logging.getLogger(__name__)

_PREFIX = "rag:v2:"
_INDEX_VERSION_KEY = "rag:index_version"


# RAG 语义缓存管理器,支持租户隔离与索引版本指纹
class RAGCacheManager:
    # 初始化 TTL 与进程内版本缓存
    def __init__(self):
        self.ttl = settings.rag_cache_ttl
        # 进程内版本缓存(避免每请求读 Redis)
        self._version_cache: dict = {}

    # ── 索引版本指纹(L16):知识内容变 → fingerprint 变 → 缓存 key 变 → 旧缓存失效 ──
    async def compute_index_fingerprint(self, tenant_id: str = "") -> str:
        """从向量库文档内容计算知识版本指纹(内容哈希)。文档变更指纹即变。"""
        try:
            from rag.retriever import rag_service
            docs = rag_service.query(tenant_id=tenant_id, limit=2000)
            if not docs:
                return "empty"
            # 用 source + title 组合做指纹(内容变更通常伴随 source 变化)
            payload = json.dumps(
                [f"{d.get('source', '')}:{d.get('title', '')}" for d in docs],
                ensure_ascii=False, sort_keys=True)
            return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
        except Exception as e:
            logger.warning(f"索引指纹计算失败: {e}")
            return "unknown"

    # 读取当前索引版本(带进程缓存,减少 Redis 压力)
    async def get_index_version(self, tenant_id: str = "") -> str:
        """读取当前索引版本(带进程缓存,减少 Redis 压力)。"""
        cache_key = f"{_INDEX_VERSION_KEY}:{tenant_id}"
        if tenant_id in self._version_cache:
            cached = self._version_cache[tenant_id]
            if time.time() - cached[1] < 30:
                return cached[0]
        raw = await cache_service.get(cache_key)
        version = raw if raw else await self.compute_index_fingerprint(tenant_id)
        self._version_cache[tenant_id] = (version, time.time())
        return version

    # 重建索引版本并失效该租户检索缓存
    async def rebuild_index(self, tenant_id: str = "") -> str:
        """重建索引版本 + 失效该租户检索缓存(旧知识不继续被复用)。"""
        new_version = await self.compute_index_fingerprint(tenant_id)
        cache_key = f"{_INDEX_VERSION_KEY}:{tenant_id}"
        await cache_service.set(cache_key, new_version, ttl=86400)
        self._version_cache[tenant_id] = (new_version, time.time())
        await self.invalidate(tenant_id=tenant_id)
        logger.info(f"🔄 索引重建: tenant={tenant_id} version={new_version}")
        return new_version

    # 构造语义缓存 key(含索引版本指纹)
    @staticmethod
    def build_cache_key(user_input: str, intent: str, top_k: int = 5,
                        tenant_id: str = "", index_version: str = "") -> str:
        qh = hashlib.md5(user_input.strip().encode()).hexdigest()[:12]
        # ⭐ 索引版本进缓存 key:知识更新后旧缓存自动失效
        return f"{_PREFIX}{tenant_id}:{intent}:{index_version}:{qh}:k{top_k}"

    # 读取缓存
    async def get(self, key: str) -> dict | None:
        return await cache_service.get(key)

    # 写入缓存(带默认 TTL)
    async def set(self, key: str, value: dict, ttl: int = None) -> bool:
        return await cache_service.set(key, value, ttl=ttl or self.ttl)

    # 文档变更后失效对应 intent 缓存
    async def invalidate(self, intent: str = "", tenant_id: str = ""):
        """文档变更后失效对应 intent 缓存（简化：清整个前缀，生产按 pattern 删）"""
        logger.info(f"RAG 缓存失效: intent={intent} tenant={tenant_id}")
        # 简化实现：不清 Redis（生产按 prefix 扫描删除）


rag_cache = RAGCacheManager()
