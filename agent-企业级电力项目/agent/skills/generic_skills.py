"""通用领域 Skill — 文档问答 / 对比 / 总结，证明底座领域无关

专项转化(步骤6):文档推理三专项 — 引用溯源(citations)+ KG 关系证据 + 按场景走 reasoner
"""
import logging
from typing import Any, Dict, List

from agent.skills.base_skill import BaseSkill, SkillMetadata
from agent.harness.risk_level import RiskLevel
from llm.adapter import unified_llm

logger = logging.getLogger(__name__)


def _retrieve_with_citations(query: str, tenant_id: str, top_k: int = 4) -> Dict[str, Any]:
    """复用 RAG 底座检索,返回 evidence 文本 + 结构化 citations(引用溯源)。"""
    try:
        from rag.retriever import rag_service
        r = rag_service.search(query, top_k=top_k, tenant_id=tenant_id)
        results = r.get("results", [])
        lines = [it.get("doc", {}).get("content", "")[:400] for it in results]
        citations = [
            {
                "source": it.get("doc", {}).get("metadata", {}).get("source", ""),
                "title": it.get("doc", {}).get("metadata", {}).get("title", ""),
                "chunk_id": it.get("chunk_id", ""),
                "score": round(float(it.get("score", 0)), 4),
            }
            for it in results[:4]
        ]
        return {
            "evidence": "\n".join(f"- {x}" for x in lines if x) or "（未检索到相关文档）",
            "citations": citations,
        }
    except Exception as e:
        logger.warning(f"检索失败: {e}")
        return {"evidence": "（检索不可用）", "citations": []}


def _kg_evidence(query: str) -> str:
    """从问题抽电力实体 → 查 KG 一跳关系 → 证据文本(向量补不了的精确关系)。"""
    try:
        from rag.kg.entity_index import entity_index
        entities = entity_index.extract_entities(query)
        lines = []
        for ent in entities:
            for r in entity_index.query(ent):
                lines.append(f"  {r['subject']} --[{r['relation']}]--> {r['object']}")
        return "\n".join(lines) if lines else ""
    except Exception as e:
        logger.debug(f"KG 证据跳过: {e}")
        return ""


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
        r = _retrieve_with_citations(query, tenant_id)
        evidence, citations = r["evidence"], r["citations"]
        kg = _kg_evidence(query)
        try:
            result = await unified_llm.ainvoke("doc_qa", [
                {"role": "system", "content": "你是通用文档问答助理。基于证据回答并引用出处（规程名/章节），检索不到就明确说明'知识库中未找到'，不要编造。"},
                {"role": "user", "content": f"证据：\n{evidence}\n{('实体关系：\n' + kg) if kg else ''}\n\n问题：{query}"},
            ])
            return {"success": True,
                    "result": {"answer": result.content, "evidence": evidence,
                               "citations": citations, "kg_evidence": kg},
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
        r = _retrieve_with_citations(query, tenant_id)
        evidence, citations = r["evidence"], r["citations"]
        kg = _kg_evidence(query)
        try:
            # 对比/分析走 reasoner(核心推理)
            result = await unified_llm.ainvoke("doc_compare", [
                {"role": "system", "content": "你是通用分析助理。基于证据做结构化对比，逐维度输出并明确标注【事实】(来自文档)与【推演】(我的推断)，引用来源。"},
                {"role": "user", "content": f"证据：\n{evidence}\n{('实体关系：\n' + kg) if kg else ''}\n\n对比需求：{query}"},
            ])
            return {"success": True,
                    "result": {"analysis": result.content, "evidence": evidence,
                               "citations": citations, "kg_evidence": kg},
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
        r = _retrieve_with_citations(query, tenant_id)
        evidence, citations = r["evidence"], r["citations"]
        try:
            result = await unified_llm.ainvoke("doc_summary", [
                {"role": "system", "content": "你是通用文档总结助理。基于证据提炼要点，分条列出，每条标注依据章节，不添加文档外的内容。"},
                {"role": "user", "content": f"文档内容：\n{evidence}\n\n总结要求：{query}"},
            ])
            return {"success": True,
                    "result": {"summary": result.content, "evidence": evidence,
                               "citations": citations},
                    "confidence": 0.7}
        except Exception as e:
            return {"success": False, "error": str(e)}
