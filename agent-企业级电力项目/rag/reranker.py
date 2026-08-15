"""重排器 — cross-encoder 精排(可选) + 本地规则兜底

对齐课程 Reranker(14课):对初召回候选重新排序,用交叉编码器成对判断"谁更适合当前问题"。
- 有 cross-encoder 模型 → 精排(BAAI/bge-reranker-base)
- 无模型/失败 → 本地规则兜底(关键词/来源加分)
分数分层:vector_score(初召回)/ rerank_score(精排)分离,rerank_reasons 可解释。
"""
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


# 重排器：cross-encoder精排(可选) + 本地规则兜底
class Reranker:
    # 初始化模型槽位
    def __init__(self):
        self._model = None

    # 懒加载cross-encoder重排模型，失败返回False走本地兜底
    def _load_model(self):
        """懒加载 cross-encoder(bge-reranker)。失败返回 None,走本地兜底。"""
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
            from config.settings import settings
            model_name = getattr(settings, "reranker_model", "BAAI/bge-reranker-base")
            self._model = CrossEncoder(model_name, device=getattr(settings, "embedding_device", "cpu"))
            logger.info(f"✅ cross-encoder 重排器就绪: {model_name}")
        except Exception as e:
            logger.warning(f"⚠️ cross-encoder 加载失败({e}),用本地规则兜底")
            self._model = False  # 标记不可用
        return self._model

    # 对初召回候选精排，返回TopK带rerank_score的结果
    def rerank(self, query: str, results: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        if not results:
            return []

        # ── ① cross-encoder 精排(优先) ──
        model = self._load_model()
        if model:
            try:
                pairs = [(query, str(r.get("doc", {}).get("content", ""))[:512]) for r in results]
                scores = model.predict(pairs)
                for r, s in zip(results, scores):
                    r["rerank_score"] = round(float(s), 4)
                    r["rerank_reasons"] = ["cross-encoder 精排"]
                ranked = sorted(results, key=lambda r: r.get("rerank_score", 0), reverse=True)
                return ranked[:top_k]
            except Exception as e:
                logger.warning(f"cross-encoder 精排失败,回退本地: {e}")

        # ── ② 本地规则兜底:关键词/来源加分 ──
        for r in results:
            meta = r.get("doc", {}).get("metadata", {}) or {}
            title = str(meta.get("title", ""))
            source = str(meta.get("source", ""))
            r["rerank_reasons"] = []
            base = r.get("score", 0)
            if query:
                for kw in query[:6]:
                    if kw and kw in title:
                        base += 0.1
                        r["rerank_reasons"].append("标题关键词命中")
            if query and any(kw in source for kw in query[:4]):
                base += 0.05
                r["rerank_reasons"].append("来源关键词命中")
            r["rerank_score"] = round(base, 4)
        ranked = sorted(results, key=lambda r: r.get("rerank_score", 0), reverse=True)
        return ranked[:top_k]


reranker = Reranker()
