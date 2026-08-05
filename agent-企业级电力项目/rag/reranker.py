"""重排器 — 轻量：按 score 排序 + 关键词加分（本机够用）"""
from typing import List, Dict, Any


class Reranker:
    def rerank(self, query: str, results: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        if not results:
            return []
        # 关键词命中加分：标题/来源含 query 关键词的排前
        for r in results:
            meta = r.get("doc", {}).get("metadata", {})
            title = str(meta.get("title", ""))
            if query and any(kw in title for kw in query[:6]):
                r["score"] = r.get("score", 0) + 0.1
        ranked = sorted(results, key=lambda r: r.get("score", 0), reverse=True)
        return ranked[:top_k]


reranker = Reranker()
