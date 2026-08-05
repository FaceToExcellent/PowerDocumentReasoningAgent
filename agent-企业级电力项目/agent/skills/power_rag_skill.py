"""电力 RAG 检索 Skill — 封装 Milvus 检索（企业版带租户隔离）"""
import logging
from typing import Any, Dict

from agent.skills.base_skill import BaseSkill, SkillMetadata
from agent.harness.risk_level import RiskLevel

logger = logging.getLogger(__name__)


class PowerRAGSkill(BaseSkill):
    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="power_rag",
            description="从电力向量库检索规程、图纸、造价、故障等文档",
            tags=["规程", "标准", "检索", "文档"],
            risk_level=RiskLevel.LOW,
            category="检索",
        )

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        query = context.get("query", "")
        top_k = context.get("top_k", 5)
        intent = context.get("intent", "")
        user_ctx = context.get("user_context", {})
        tenant_id = user_ctx.get("tenant_id", context.get("tenant_id", ""))

        try:
            from rag.retriever import rag_service
            r = rag_service.search(query, top_k=top_k, intent=intent, tenant_id=tenant_id)
            results = r.get("results", [])
            docs = [{
                "content": it.get("doc", {}).get("content", "")[:800],
                "metadata": it.get("doc", {}).get("metadata", {}),
                "score": round(it.get("score", 0), 3),
            } for it in results]
            return {
                "success": True,
                "result": {"documents": docs, "total_found": len(docs), "query": query},
                "confidence": min(0.9, len(docs) / max(top_k, 1)),
            }
        except Exception as e:
            logger.warning(f"RAG 检索失败: {e}")
            return {"success": True, "result": {"documents": [], "total_found": 0},
                    "confidence": 0.0}
