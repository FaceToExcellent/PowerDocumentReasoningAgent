"""定额匹配 Skill — 区分新建/改造/大修造价系数"""
import logging
from typing import Any, Dict

from agent.skills.base_skill import BaseSkill, SkillMetadata
from agent.harness.risk_level import RiskLevel

logger = logging.getLogger(__name__)

QUOTA_RULES = {
    "新建": {"multiplier": 1.0, "desc": "新建工程使用标准定额"},
    "改造": {"multiplier": 1.15, "desc": "改造工程增加 15% 拆除+措施费"},
    "大修": {"multiplier": 0.7, "desc": "大修按新建 70% 定额"},
}


# 定额匹配 Skill:区分新建/改造/大修工程造价系数
class QuotaMatchSkill(BaseSkill):
    # 元数据:定额匹配 Skill 的名称/描述/标签等
    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="quota_match",
            description="电力定额匹配，区分新建/改造/大修工程的造价系数",
            tags=["造价", "定额", "费用", "预算"],
            risk_level=RiskLevel.LOW,
            category="计算",
        )

    # 执行定额匹配:识别工程类型并返回对应造价系数
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        query = context.get("query", "")
        project_type = "新建"
        for kw, rule in [("改造", "改造"), ("大修", "大修"), ("扩建", "改造")]:
            if kw in query:
                project_type = rule
                break
        rule = QUOTA_RULES.get(project_type, QUOTA_RULES["新建"])
        return {
            "success": True,
            "result": {
                "project_type": project_type,
                "quota_base": "《电力建设工程预算定额（2023版）》",
                "multiplier": rule["multiplier"],
                "note": rule["desc"],
            },
            "confidence": 0.9,
            "duration_ms": 5,
        }
