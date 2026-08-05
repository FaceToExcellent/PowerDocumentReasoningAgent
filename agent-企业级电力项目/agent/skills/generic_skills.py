"""通用领域 Skill — 文档问答 / 对比 / 总结，证明底座领域无关"""
import logging
from typing import Any, Dict

from agent.skills.base_skill import BaseSkill, SkillMetadata
from agent.harness.risk_level import RiskLevel
from llm.adapter import unified_llm

logger = logging.getLogger(__name__)


def _retrieve(query: str, tenant_id: str, top_k: int = 4) -> str:
    """复用 RAG 底座检索（领域无关）"""
    try:
        from rag.retriever import rag_service
        r = rag_service.search(query, top_k=top_k, tenant_id=tenant_id)
        lines = [it.get("doc", {}).get("content", "")[:400]
                 for it in r.get("results", [])]
        return "\n".join(f"- {x}" for x in lines if x) or "（未检索到相关文档）"
    except Exception as e:
        logger.warning(f"检索失败: {e}")
        return "（检索不可用）"


class DocQASkill(BaseSkill):
    """通用文档问答"""

    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="doc_qa",
            description="基于文档内容回答问题，引用文档出处",
            tags=["文档", "问答", "规定", "内容"],
            risk_level=RiskLevel.LOW,
            category="问答",
        )

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        query = context.get("query", "")
        user_ctx = context.get("user_context", {})
        tenant_id = user_ctx.get("tenant_id", context.get("tenant_id", ""))
        evidence = _retrieve(query, tenant_id)
        try:
            result = await unified_llm.ainvoke("chat", [
                {"role": "system", "content": "你是通用文档问答助理。基于证据回答并引用出处，检索不到就明确说明。"},
                {"role": "user", "content": f"证据：\n{evidence}\n\n问题：{query}"},
            ])
            return {"success": True, "result": {"answer": result.content,
                                                "evidence": evidence},
                    "confidence": 0.7}
        except Exception as e:
            return {"success": False, "error": str(e)}


class DocCompareSkill(BaseSkill):
    """通用对比分析"""

    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="doc_compare",
            description="对比两份文档/两种方案的关系与差异，支持'假设换成X'分析",
            tags=["对比", "比较", "区别", "差异", "分析"],
            risk_level=RiskLevel.MEDIUM,
            category="分析",
        )

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        query = context.get("query", "")
        user_ctx = context.get("user_context", {})
        tenant_id = user_ctx.get("tenant_id", context.get("tenant_id", ""))
        evidence = _retrieve(query, tenant_id)
        try:
            result = await unified_llm.ainvoke("chat", [
                {"role": "system", "content": "你是通用分析助理。基于证据做结构化对比，逐维度输出并标注【事实】与【推演】。"},
                {"role": "user", "content": f"证据：\n{evidence}\n\n对比需求：{query}"},
            ])
            return {"success": True, "result": {"analysis": result.content, "evidence": evidence},
                    "confidence": 0.7}
        except Exception as e:
            return {"success": False, "error": str(e)}


class DocSummarySkill(BaseSkill):
    """通用文档总结"""

    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="doc_summary",
            description="总结文档要点，分条列出，标注依据章节",
            tags=["总结", "摘要", "要点", "概括"],
            risk_level=RiskLevel.LOW,
            category="总结",
        )

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        query = context.get("query", "")
        user_ctx = context.get("user_context", {})
        tenant_id = user_ctx.get("tenant_id", context.get("tenant_id", ""))
        evidence = _retrieve(query, tenant_id)
        try:
            result = await unified_llm.ainvoke("chat", [
                {"role": "system", "content": "你是通用文档总结助理。基于证据提炼要点，分条列出，标注依据章节。"},
                {"role": "user", "content": f"文档内容：\n{evidence}\n\n总结要求：{query}"},
            ])
            return {"success": True, "result": {"summary": result.content, "evidence": evidence},
                    "confidence": 0.7}
        except Exception as e:
            return {"success": False, "error": str(e)}
