"""FactCheck Skill — 对回答做证据支撑校验（graph 内也有 fact_check_node，这是 Skill 形式）"""
import logging
from typing import Any, Dict

from agent.skills.base_skill import BaseSkill, SkillMetadata
from agent.harness.risk_level import RiskLevel

logger = logging.getLogger(__name__)


class FactCheckSkill(BaseSkill):
    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="fact_check",
            description="对 Agent 输出做事实校验：证据支撑 / 数值一致性 / 无证据标注",
            tags=["校验", "幻觉", "事实"],
            risk_level=RiskLevel.LOW,
            category="校验",
        )

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        from agent.fact_checker import check_output
        output = context.get("content", "")
        intent = context.get("intent", "")
        if not output:
            return {"success": True, "result": {"passed": True, "errors": []}, "confidence": 1.0}
        result = check_output(output, {"intent": intent})
        return {
            "success": True,
            "result": result,
            "confidence": 0.9 if result.get("passed") else 0.3,
        }
