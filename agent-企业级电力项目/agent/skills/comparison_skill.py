"""对比/影响分析 Skill — 关系 / 影响 / 反事实（M5.6 核心）
拆实体对 → 多域检索（复用 RAG）→ 维度化对比 → 反事实投影 → 结论分级
"""
import logging
from typing import Any, Dict, List

from agent.skills.base_skill import BaseSkill, SkillMetadata
from agent.harness.risk_level import RiskLevel
from llm.adapter import unified_llm

logger = logging.getLogger(__name__)

# 维度化推理的 Prompt 骨架
_COMPARE_PROMPT = """你是电力分析助理。请对下面两个对象做结构化对比分析。
对象A：{entity_a}
对象B：{entity_b}
检索到的证据：
{evidence}

请按以下维度逐项对比，每项给出结论 + 依据：
1. 安全合规影响
2. 供电可靠性影响
3. 改造成本影响
4. 运维差异
最后给出总结论，并明确标注：哪些是【事实】（有证据）、哪些是【推演】（推测+建议核实）。"""


# 对比/影响分析 Skill:拆实体对→检索→维度化推理→结论分级
class ComparisonAnalysisSkill(BaseSkill):
    # 元数据:对比分析 Skill 的名称/描述/标签等
    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="comparison_analysis",
            description="分析两个电力对象的关系/影响/差异，支持'假设换成X会怎样'的反事实分析",
            tags=["对比", "影响", "关系", "假设", "分析"],
            risk_level=RiskLevel.MEDIUM,
            category="分析",
        )

    # 执行对比分析:抽取实体、检索证据、调用 LLM 维度推理
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        query = context.get("query", "")
        user_ctx = context.get("user_context", {})
        tenant_id = user_ctx.get("tenant_id", context.get("tenant_id", ""))

        # 1. 抽取实体对（简化：A 和 B，或单个主体 A）
        entities = self._extract_entities(query)
        if not entities:
            return {"success": False, "error": "未识别出对比对象"}

        # 2. 多域检索（复用 RAG，不新建）
        evidence = await self._retrieve_evidence(query, tenant_id)

        # 3. 维度化推理（核心推理 → DeepSeek；无 key 降级本地）
        prompt = _COMPARE_PROMPT.format(
            entity_a=entities[0], entity_b=entities[1] if len(entities) > 1 else "（无，单对象影响分析）",
            evidence=evidence,
        )
        try:
            result = await unified_llm.ainvoke("comparison_analysis", [
                {"role": "system", "content": "你是严谨的电力分析助理，结论必须标注事实/推演。"},
                {"role": "user", "content": prompt},
            ])
            content = result.content
        except Exception as e:
            logger.error(f"对比分析推理失败: {e}")
            content = "（推理服务不可用，以下为基于证据的归纳）\n" + evidence

        # 4. 结论分级（在文本中体现事实/推演标注）
        return {
            "success": True,
            "result": {
                "entities": entities,
                "analysis": content,
                "evidence_count": len(evidence.split("\n")),
            },
            "confidence": 0.7,
        }

    # 从查询中抽取待对比的电力设备实体(简化的关键词抽取)
    @staticmethod
    def _extract_entities(query: str) -> List[str]:
        """简化抽取：常见电力设备词 + 用'和/与/换成/对比'分隔"""
        candidates = []
        for kw in ["主变", "变压器", "母线", "母线保护", "主变保护", "熔断器", "断路器",
                   "隔离开关", "避雷器", "互感器", "电容器", "电抗器"]:
            if kw in query:
                candidates.append(kw)
        return candidates[:2]

    # 多域检索证据(复用 RAG 服务,失败时降级返回)
    async def _retrieve_evidence(self, query: str, tenant_id: str) -> str:
        try:
            from rag.retriever import rag_service
            r = rag_service.search(query, top_k=4, tenant_id=tenant_id)
            lines = [it.get("doc", {}).get("content", "")[:300]
                     for it in r.get("results", [])]
            return "\n".join(f"- {x}" for x in lines if x)
        except Exception:
            return "（检索不可用）"
