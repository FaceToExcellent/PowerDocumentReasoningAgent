"""工具澄清 — 缺参/歧义时先追问,不猜测执行

对齐课程工具澄清(19课):
- 工具前澄清:必填参数缺失时先追问,不执行
- 工具后澄清:候选不唯一时保留结果让用户确认
- 澄清是可恢复暂停,不是工具失败
"""
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class Clarification:
    """结构化澄清请求。"""

    def __init__(self, clarification_field: str, question: str,
                 candidates: List[str] = None):
        self.clarification_field = clarification_field
        self.question = question
        self.candidates = candidates or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clarification_field": self.clarification_field,
            "question": self.question,
            "candidates": self.candidates,
        }


def pre_tool_clarification(intent: str, query: str, params: Dict[str, Any],
                           required: List[str]) -> Optional[Clarification]:
    """工具前澄清:检查必填参数,缺失则追问。"""
    missing = [p for p in required if not params.get(p)]
    if not missing:
        return None
    # 对比分析缺第二个对象
    if intent == "comparison_analysis" and "entity_b" in missing:
        return Clarification(
            clarification_field="entity_b",
            question="请补充要对比的第二个对象(如'对比主变和断路器检修要求')。",
        )
    return Clarification(
        clarification_field=",".join(missing),
        question=f"需要补充: {', '.join(missing)} 才能继续。",
    )


def post_tool_clarification(result: Dict[str, Any], candidates_key: str = "candidates") -> Optional[Clarification]:
    """工具后澄清:返回多个候选且目标不唯一时,让用户确认。"""
    candidates = result.get(candidates_key) or []
    if len(candidates) <= 1:
        return None
    return Clarification(
        clarification_field="target",
        question="检索到多个相关结果,请确认你要哪一个:",
        candidates=[str(c)[:60] for c in candidates[:5]],
    )
