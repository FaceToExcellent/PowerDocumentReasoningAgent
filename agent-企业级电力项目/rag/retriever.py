"""RAG 统一检索服务 — 工厂模式：Milvus / Chroma 一行切换，接口兼容"""
import logging
from typing import Dict, List, Any, Optional

from config.settings import settings

logger = logging.getLogger(__name__)


class RAGService:
    """统一 RAG 检索：向量检索 + 混合 + Rerank"""

    def __init__(self):
        from rag.vector_store.milvus_store import MilvusVectorStore
        from rag.vector_store.chroma_store import ChromaVectorStore
        if settings.vector_store_type == "milvus":
            self.store = MilvusVectorStore()
        else:
            self.store = ChromaVectorStore(persist_dir=settings.chroma_persist_dir)
        logger.info(f"向量库模式: {settings.vector_store_type}")

    # ── 检索（兼容原 RAGService.search 接口 + tenant_id）──
    def search(self, query: str, top_k: int = 5, filters: Optional[Dict] = None,
               intent: str = "", tenant_id: str = "") -> Dict[str, Any]:
        results = self.store.search(query, top_k=top_k, filters=filters, tenant_id=tenant_id)
        results = self._hybrid_and_rerank(query, results, top_k)
        return {"results": results, "total_found": len(results), "query": query}

    def _hybrid_and_rerank(self, query: str, dense_results: list, top_k: int) -> list:
        """轻量混合 + Rerank：dense 结果按 score 重排（本机够用，生产接 BM25+RRF）"""
        if not dense_results:
            return []
        ranked = sorted(dense_results, key=lambda r: r.get("score", 0), reverse=True)
        return ranked[:top_k]

    # ── 入库 ──
    def add_documents(self, docs: List[Dict[str, Any]], tenant_id: str = "") -> int:
        return self.store.add_documents(docs, tenant_id=tenant_id)

    def delete_by_tenant(self, tenant_id: str) -> bool:
        return self.store.delete_by_tenant(tenant_id)

    def count(self, tenant_id: str = "") -> int:
        return self.store.count(tenant_id)

    def query(self, tenant_id: str = "", limit: int = 20) -> list:
        return self.store.query(tenant_id=tenant_id, limit=limit)


# 全局单例
rag_service = RAGService()
