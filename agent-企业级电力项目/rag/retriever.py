"""RAG 统一检索服务 — 工厂模式：Milvus / Chroma 一行切换，接口兼容
Hybrid RAG(L15):向量召回(dense)+ 关键词召回(BM25直觉)+ RRF 融合
"""
import logging
import math
from typing import Dict, List, Any, Optional

from config.settings import settings

logger = logging.getLogger(__name__)

# RRF 融合常数(rank 倒数融合,不依赖两路分数量纲)
RRF_K = 60


def _rrf_merge(vector_hits: List[Dict], keyword_hits: List[Dict], top_k: int) -> List[Dict]:
    """Reciprocal Rank Fusion:按排名融合两路召回,去重合并来源。"""
    merged: Dict[str, Dict[str, Any]] = {}

    def _accumulate(hits: List[Dict], source: str):
        for rank, hit in enumerate(hits):
            doc = hit.get("doc", {})
            meta = doc.get("metadata", {}) or {}
            key = meta.get("chunk_id") or meta.get("source") or doc.get("content", "")[:40]
            entry = merged.setdefault(key, {"doc": doc, "sources": [], "rrf": 0.0,
                                            "score": 0.0, "vector_score": 0.0,
                                            "keyword_score": 0.0, "chunk_id": meta.get("chunk_id", "")})
            entry["rrf"] += 1.0 / (RRF_K + rank + 1)
            if source not in entry["sources"]:
                entry["sources"].append(source)
            entry["vector_score"] = max(entry["vector_score"], hit.get("vector_score", 0) or hit.get("score", 0))
            entry["keyword_score"] = max(entry["keyword_score"], hit.get("keyword_score", 0))

    _accumulate(vector_hits, "vector")
    _accumulate(keyword_hits, "keyword")

    for entry in merged.values():
        entry["score"] = round(entry["rrf"], 4)
        entry["hybrid"] = True
    ranked = sorted(merged.values(), key=lambda e: e["rrf"], reverse=True)
    return ranked[:top_k]


def _keyword_terms(query: str) -> List[str]:
    """从查询抽关键词(中文按业务词,ASCII 按 token)。"""
    import re
    terms = []
    # 电力业务词(来自知识文档高频词)
    for w in ["主变", "变压器", "母线", "断路器", "检修", "保护", "规程", "电压", "线路",
              "电缆", "造价", "定额", "故障", "跳闸", "巡视", "运维", "安规", "DL/T", "GB",
              "110", "220", "kV", "KVA"]:
        if w in query:
            terms.append(w)
    # ASCII 词
    terms += re.findall(r"[a-zA-Z0-9]{2,}", query)
    return list(dict.fromkeys(terms))


class RAGService:
    """统一 RAG 检索：向量召回 + 关键词召回 + RRF 融合 + Rerank"""

    def __init__(self):
        from rag.vector_store.milvus_store import MilvusVectorStore
        from rag.vector_store.chroma_store import ChromaVectorStore
        if settings.vector_store_type == "milvus":
            self.store = MilvusVectorStore()
        else:
            self.store = ChromaVectorStore(persist_dir=settings.chroma_persist_dir)
        self._corpus_cache: Optional[Dict[str, List[Dict]]] = None  # tenant_id -> docs(关键词召回用)
        logger.info(f"向量库模式: {settings.vector_store_type}")

    def _get_corpus(self, tenant_id: str) -> List[Dict]:
        """加载租户文档语料(关键词召回用),缓存避免重复查询。"""
        if self._corpus_cache is None or tenant_id not in self._corpus_cache:
            try:
                self._corpus_cache = self._corpus_cache or {}
                self._corpus_cache[tenant_id] = self.store.query(tenant_id=tenant_id, limit=1000)
            except Exception as e:
                logger.warning(f"关键词语料加载失败: {e}")
                self._corpus_cache[tenant_id] = []
        return self._corpus_cache.get(tenant_id, [])

    # ── 检索（兼容原 RAGService.search 接口 + tenant_id）──
    def search(self, query: str, top_k: int = 5, filters: Optional[Dict] = None,
               intent: str = "", tenant_id: str = "") -> Dict[str, Any]:
        # ① 向量召回(dense)
        dense = self.store.search(query, top_k=top_k * 2, filters=filters, tenant_id=tenant_id)
        for hit in dense:
            hit["vector_score"] = hit.get("score", 0)

        # ② 关键词召回(BM25 直觉:关键词命中文本)
        keyword = self._keyword_retrieve(query, tenant_id, top_k=top_k * 2)

        # ③ RRF 融合 + Rerank
        hybrid = _rrf_merge(dense, keyword, top_k * 2)
        results = self._hybrid_and_rerank(query, hybrid, top_k)
        return {"results": results, "total_found": len(results), "query": query,
                "sources_dist": self._source_distribution(dense, keyword)}

    def _keyword_retrieve(self, query: str, tenant_id: str, top_k: int) -> List[Dict]:
        """BM25 直觉:关键词匹配文档 title/source(向量库标量字段)。稀疏词加权。"""
        terms = _keyword_terms(query)
        if not terms:
            return []
        hits = []
        for doc in self._get_corpus(tenant_id):
            title = str(doc.get("title", ""))
            source = str(doc.get("source", ""))
            searchable = f"{title} {source}"
            matched = [t for t in terms if t.lower() in searchable.lower()]
            if not matched:
                continue
            score = sum(2.0 if len(t) >= 4 else 1.0 for t in set(matched))
            hits.append({
                "doc": {"content": title, "metadata": {"title": title, "source": source,
                                                       "chunk_id": str(doc.get("id", ""))}},
                "score": round(score, 3), "keyword_score": round(score, 3),
                "chunk_id": str(doc.get("id", "")),
            })
        hits.sort(key=lambda h: h["keyword_score"], reverse=True)
        return hits[:top_k]

    def _source_distribution(self, dense: List[Dict], keyword: List[Dict]) -> Dict[str, int]:
        return {"vector": len(dense), "keyword": len(keyword)}

    def _hybrid_and_rerank(self, query: str, hybrid_results: list, top_k: int) -> list:
        """Rerank:对混合结果精排(调用 Reranker,含 cross-encoder 可选)。"""
        if not hybrid_results:
            return []
        from rag.reranker import reranker
        return reranker.rerank(query, hybrid_results, top_k)

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
