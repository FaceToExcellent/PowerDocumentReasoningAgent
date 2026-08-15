"""错误分类与降级决策 — error_category 8 类枚举 + degradation_policy

对齐课程错误降级(21课):
- error_category 8 类:timeout/validation_error/not_found/forbidden/business_error/
  model_unavailable/system_error/high_risk_write_blocked
- degradation_policy:决策层定 next_action,fallback 话术层定口径
- 只读工具可重试一次;高风险写动作不重试直接转人工
"""
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# 8 类错误分类
ERROR_CATEGORIES = [
    "timeout", "validation_error", "not_found", "forbidden",
    "business_error", "model_unavailable", "system_error", "high_risk_write_blocked",
]

# 降级决策:每类错误的 retry/next_action 口径
_ERROR_POLICY: Dict[str, Dict[str, Any]] = {
    "timeout": {"retry": True, "max_attempts": 2, "next_action": "fallback_answer",
                "fallback": "服务暂时繁忙，请稍后再试。"},
    "validation_error": {"retry": False, "next_action": "ask_clarification",
                         "fallback": "参数不合法，请补充或修正后重试。"},
    "not_found": {"retry": False, "next_action": "fallback_answer",
                  "fallback": "未找到对应记录。"},
    "forbidden": {"retry": False, "next_action": "fallback_answer",
                  "fallback": "当前账号无权访问该资源。"},
    "business_error": {"retry": False, "next_action": "fallback_answer",
                       "fallback": "当前业务状态不允许该操作。"},
    "model_unavailable": {"retry": False, "next_action": "fallback_answer",
                          "fallback": "模型服务暂时不可用，保留已确认事实。"},
    "system_error": {"retry": False, "next_action": "fallback_answer",
                     "fallback": "系统暂时异常，建议稍后再试或转人工。"},
    "high_risk_write_blocked": {"retry": False, "next_action": "transfer_to_human",
                                "fallback": "该操作属于高风险动作，需要人工审批。"},
}


# 降级决策结果:分类/是否重试/下一步动作/兜底话术
class DegradationDecision:
    # 初始化降级决策字段
    def __init__(self, category: str, retry: bool = False, max_attempts: int = 1,
                 next_action: str = "fallback_answer", fallback_message: str = ""):
        self.category = category
        self.retry = retry
        self.max_attempts = max_attempts
        self.next_action = next_action
        self.fallback_message = fallback_message

    # 序列化为字典
    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_category": self.category, "retry": self.retry,
            "max_attempts": self.max_attempts, "next_action": self.next_action,
            "fallback_message": self.fallback_message,
        }


# 把异常归类为 8 类错误之一
def classify_error(err: Exception) -> str:
    """把异常归类为 8 类错误。"""
    msg = str(err).lower()
    if isinstance(err, TimeoutError) or "timeout" in msg:
        return "timeout"
    if "not found" in msg or "no such" in msg or "不存在" in msg:
        return "not_found"
    if "forbidden" in msg or "permission" in msg or "无权" in msg or "401" in msg or "403" in msg:
        return "forbidden"
    if "model" in msg or "llm" in msg or "api key" in msg:
        return "model_unavailable"
    return "system_error"


# 错误分类 → 降级决策(只读工具才允许重试)
def degradation_policy(category: str, *, is_read_only: bool = True) -> DegradationDecision:
    """错误分类 → 降级决策(决策层)。"""
    policy = _ERROR_POLICY.get(category, _ERROR_POLICY["system_error"])
    retry = policy["retry"] and is_read_only  # 只读工具才允许重试
    return DegradationDecision(
        category=category, retry=retry,
        max_attempts=policy.get("max_attempts", 1) if retry else 1,
        next_action=policy["next_action"],
        fallback_message=policy["fallback"],
    )
