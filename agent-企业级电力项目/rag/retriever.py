"""RAG 统一检索服务 — 工厂模式：Milvus / Chroma 一行切换，接口兼容
Hybrid RAG：向量召回(dense)+ 关键词召回(BM25直觉)+ RRF 融合
"""
import logging
import math
from typing import Dict, List, Any, Optional

from config.settings import settings

logger = logging.getLogger(__name__)

# RRF 融合常数(rank 倒数融合,不依赖两路分数量纲)
RRF_K = 60


# RRF融合：按排名倒数融合向量/关键词两路召回，去重合并来源
def _rrf_merge(vector_hits: List[Dict], keyword_hits: List[Dict], top_k: int) -> List[Dict]:
    """Reciprocal Rank Fusion:按排名融合两路召回,去重合并来源。"""
    merged: Dict[str, Dict[str, Any]] = {}

    # 单路命中累加进融合字典(按chunk_id去重)
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


# 中文分词(jieba,BM25 全文索引/查询用)
def _tokenize(text: str) -> List[str]:
    """jieba 中文分词 + 小写,过滤单字/空白,产出 BM25 词项。"""
    import jieba
    tokens = []
    for seg in jieba.cut(str(text or "")):
        seg = seg.strip().lower()
        if len(seg) >= 2 and not seg.isspace():
            tokens.append(seg)
    return tokens


# 统一RAG检索服务：向量+关键词召回、RRF融合、Rerank，兼容Milvus/Chroma
class RAGService:
    """统一 RAG 检索：向量召回 + 关键词召回 + RRF 融合 + Rerank"""

    # 按配置选择Milvus/Chroma向量库并初始化语料缓存
    def __init__(self):
        from rag.vector_store.milvus_store import MilvusVectorStore
        from rag.vector_store.chroma_store import ChromaVectorStore
        if settings.vector_store_type == "milvus":
            self.store = MilvusVectorStore()
        else:
            self.store = ChromaVectorStore(persist_dir=settings.chroma_persist_dir)
        self._corpus_cache: Optional[Dict[str, List[Dict]]] = None  # tenant_id -> docs(关键词召回用)
        self._bm25_index: Dict[str, Any] = {}       # tenant_id -> BM25Okapi
        self._bm25_docs: Dict[str, List[Dict]] = {} # tenant_id -> docs(与索引对齐)
        logger.info(f"向量库模式: {settings.vector_store_type}")

    # 加载租户文档语料(关键词召回用)，带缓存避免重复查询
    def _get_corpus(self, tenant_id: str) -> List[Dict]:
        """加载租户文档语料(关键词召回用),缓存避免重复查询。"""
        if self._corpus_cache is None or tenant_id not in self._corpus_cache:
            try:
                self._corpus_cache = self._corpus_cache or {}
                self._corpus_cache[tenant_id] = self.store.query(tenant_id=tenant_id, limit=1000, include_content=True)
            except Exception as e:
                logger.warning(f"关键词语料加载失败: {e}")
                self._corpus_cache[tenant_id] = []
        return self._corpus_cache.get(tenant_id, [])

    # ── 检索（兼容原 RAGService.search 接口 + tenant_id）──
    def search(self, query: str, top_k: int = 5, filters: Optional[Dict] = None,
               intent: str = "", tenant_id: str = "") -> Dict[str, Any]:
        from observability.tracing import tracer
        with tracer.span("rag_search", query=query[:100], intent=intent,
                         top_k=top_k, tenant_id=tenant_id):
            # ① 向量召回(dense)
            with tracer.span("rag_vector_search", tenant_id=tenant_id):
                dense = self.store.search(query, top_k=top_k * 2, filters=filters,
                                          tenant_id=tenant_id)
                for hit in dense:
                    hit["vector_score"] = hit.get("score", 0)

            # ② 关键词召回(BM25 全文:jieba 分词 + rank_bm25)
            keyword = self._bm25_retrieve(query, tenant_id, top_k=top_k * 2)

            # ③ RRF 融合 + Rerank
            with tracer.span("rag_rrf_merge", vector=len(dense), keyword=len(keyword)):
                hybrid = _rrf_merge(dense, keyword, top_k * 2)
            results = self._hybrid_and_rerank(query, hybrid, top_k)
        return {"results": results, "total_found": len(results), "query": query,
                "sources_dist": self._source_distribution(dense, keyword)}

    # BM25 全文召回:jieba 分词 + rank_bm25 打分(基于文档 content)
    def _bm25_retrieve(self, query: str, tenant_id: str, top_k: int) -> List[Dict]:
        """真 BM25:对全文 content 建索引打分(rank_bm25),替代旧 title/source 关键词匹配。"""
        terms = _tokenize(query)
        if not terms:
            return []
        index, docs = self._bm25(tenant_id)
        if index is None or not docs:
            return []
        try:
            from observability.tracing import tracer
            with tracer.span("rag_bm25", terms=terms[:10], docs=len(docs)):
                scores = index.get_scores(terms)
        except Exception as e:
            logger.warning(f"BM25 打分失败: {e}")
            return []
        # 小语料兜底:BM25Okapi 的 idf 在 df≈N/2 时归零(如 2 篇文档各命中 1 篇),
        # 整体归零时退化为词频命中,保证关键词召回不空手
        if not any(s > 0 for s in scores):
            scores = [float(sum(1 for t in terms if t in _tokenize(d.get("content", ""))))
                      for d in docs]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return self._hits_from(docs, ranked, scores)

    # 按打分排名组装 keyword 命中(hit 结构供 RRF 融合消费)
    def _hits_from(self, docs: List[Dict], indices: List[int], scores: List[float]) -> List[Dict]:
        hits = []
        for i in indices:
            if scores[i] <= 0:
                continue
            doc = docs[i]
            hits.append({
                "doc": {"content": str(doc.get("content", "")),
                        "metadata": {"title": str(doc.get("title", "")),
                                     "source": str(doc.get("source", "")),
                                     "chunk_id": str(doc.get("id", ""))}},
                "score": round(float(scores[i]), 4),
                "keyword_score": round(float(scores[i]), 4),
                "chunk_id": str(doc.get("id", "")),
            })
        return hits

    # 构建/复用 BM25 索引(按租户,惰性;语料变化由 add_documents 失效)
    def _bm25(self, tenant_id: str):
        """按租户惰性构建 BM25Okapi 索引并缓存。"""
        if tenant_id not in self._bm25_index:
            corpus = self._get_corpus(tenant_id)
            tokenized = [_tokenize(d.get("content", "")) for d in corpus]
            from rank_bm25 import BM25Okapi
            self._bm25_index[tenant_id] = BM25Okapi(tokenized) if any(tokenized) else None
            self._bm25_docs[tenant_id] = corpus
        return self._bm25_index.get(tenant_id), self._bm25_docs.get(tenant_id, [])

    # 统计两路召回的命中数分布
    def _source_distribution(self, dense: List[Dict], keyword: List[Dict]) -> Dict[str, int]:
        return {"vector": len(dense), "keyword": len(keyword)}

    # 对混合结果调用Reranker精排后返回TopK
    def _hybrid_and_rerank(self, query: str, hybrid_results: list, top_k: int) -> list:
        """Rerank:对混合结果精排(调用 Reranker,含 cross-encoder 可选)。"""
        if not hybrid_results:
            return []
        from rag.reranker import reranker
        from observability.tracing import tracer
        with tracer.span("rag_rerank", candidates=len(hybrid_results)):
            return reranker.rerank(query, hybrid_results, top_k)

    # ── 入库 ──
    def add_documents(self, docs: List[Dict[str, Any]], tenant_id: str = "") -> int:
        # 语料变化 → 失效 BM25 索引与语料缓存,下次查询重建
        self._bm25_index.pop(tenant_id, None)
        self._bm25_docs.pop(tenant_id, None)
        if self._corpus_cache is not None:
            self._corpus_cache.pop(tenant_id, None)
        return self.store.add_documents(docs, tenant_id=tenant_id)

    # 按租户删除全部文档
    def delete_by_tenant(self, tenant_id: str) -> bool:
        return self.store.delete_by_tenant(tenant_id)

    # 统计文档数(可选租户)
    def count(self, tenant_id: str = "") -> int:
        return self.store.count(tenant_id)

    # 按租户查询原始数据(管理/调试用)
    def query(self, tenant_id: str = "", limit: int = 20) -> list:
        return self.store.query(tenant_id=tenant_id, limit=limit)


# 全局单例
rag_service = RAGService()
