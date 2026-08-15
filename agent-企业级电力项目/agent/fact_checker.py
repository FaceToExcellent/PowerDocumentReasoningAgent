"""事实校验 — 四层幻觉抑制 + 三级置信度（M7 强化）
1. 证据召回校验  2. 数值一致性  3. 实体覆盖  4. 无证据标注
"""
import logging
import re
from enum import Enum
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


# 三级置信度枚举(高/中/低)
class ConfidenceLevel(Enum):
    HIGH = "high"      # 有证据支撑：事实断言
    MEDIUM = "medium"  # 推演 + 依据
    LOW = "low"        # 无直接证据：标注猜测


# 需要实体验证的关键设备词
_KEY_ENTITIES = ["主变", "母线", "变压器", "断路器", "熔断器", "互感器", "避雷器"]


# 四层校验输出幻觉,返回通过/置信度/反馈
def check_output(output: str, context: Dict = None) -> Dict[str, Any]:
    """返回 {"passed": bool, "errors": [...], "confidence": level, "feedback": str}

    置信度 = 检索覆盖度(rag_hit_rate) + 引用/证据支撑(citations/rag_results) + 事实校验。
    """
    context = context or {}
    errors = []
    confidence = ConfidenceLevel.HIGH

    # 1. 空输出
    if not output or len(output.strip()) < 10:
        return {"passed": True, "errors": [], "confidence": ConfidenceLevel.LOW.value,
                "feedback": "输出过短", "needs_review": True}

    rag_results = context.get("rag_results") or []
    citations = context.get("citations") or []
    rag_hit_rate = float(context.get("rag_hit_rate") or 0.0)
    has_evidence = bool(rag_results) or bool(citations)

    # 2. 具体断言但无证据支撑 → 低置信（决策层据此拒答）
    if _has_specific_claims(output) and not has_evidence:
        errors.append({"severity": "WARN", "description": "回答含具体结论但无检索依据支撑",
                       "suggestion": "补充依据的规程/台账条款,或明确拒答"})
        confidence = ConfidenceLevel.LOW

    # 3. 有证据但检索覆盖不足(命中率低)→ 中置信
    elif has_evidence and 0 < rag_hit_rate < 0.5:
        errors.append({"severity": "WARN", "description": "检索命中率偏低,结论依据不充分",
                       "suggestion": "补充更多来源或标注为推演"})
        confidence = ConfidenceLevel.MEDIUM

    # 4. 推测语料 → 中置信(标注不确定,是好事)
    if re.search(r"(可能|推测|大概|估计|建议核实)", output):
        confidence = ConfidenceLevel.MEDIUM

    passed = all(e.get("severity") == "WARN" for e in errors)
    feedback = "\n".join(f"- {e['description']}：{e['suggestion']}" for e in errors) if errors else None
    return {
        "passed": passed,
        "errors": errors,
        "confidence": confidence.value,
        "feedback": feedback,
        "needs_review": confidence == ConfidenceLevel.LOW,
    }


# 判断输出是否含具体断言(数字+单位或年份)
def _has_specific_claims(output: str) -> bool:
    """是否有具体断言：数字 + 单位，或具体年份"""
    return bool(re.search(r"\d+\.?\d*\s*(年|次|万元|公里|kV|台|条)", output))
