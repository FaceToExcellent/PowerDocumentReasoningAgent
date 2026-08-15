"""风险等级定义 + 高危 Skill 清单（Harness 前置拦截依据）"""
from enum import Enum


# 风险等级枚举(低/中/高/严重)
class RiskLevel(Enum):
    LOW = "low"            # 查询/检索：自动执行
    MEDIUM = "medium"      # 生成报告/分析：自动 + 提示
    HIGH = "high"          # 批量操作/删除/改配置：强制人工确认
    CRITICAL = "critical"  # 影响生产/停电：强制确认 + 二次鉴权


# Skill → 风险等级
HIGH_RISK_SKILLS = {
    "batch_doc_delete": RiskLevel.HIGH,
    "equipment_config_modify": RiskLevel.HIGH,
    "power_outage_plan": RiskLevel.CRITICAL,
    "cost_adjustment": RiskLevel.MEDIUM,
}

# 意图 → 风险等级（supervisor 判定意图后，执行前拦截依据）
INTENT_RISK_MAP = {
    "cost_audit": RiskLevel.HIGH,               # 造价核算/调整涉及资金，强制人工确认
    "comparison_analysis": RiskLevel.LOW,       # 对比分析纯检索，自动执行
}


# 按 skill 名或意图名查询风险等级(未登记默认 LOW)
def get_risk_level(skill_or_intent: str) -> RiskLevel:
    """按 skill 名或意图名查风险等级（未登记默认 LOW）"""
    if skill_or_intent in HIGH_RISK_SKILLS:
        return HIGH_RISK_SKILLS[skill_or_intent]
    return INTENT_RISK_MAP.get(skill_or_intent, RiskLevel.LOW)
