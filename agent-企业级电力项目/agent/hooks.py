"""Hooks 治理层 — Skill 生命周期统一治理点(pre/post/error/completion)

对齐课程 Hooks(23课):把参数校验、脱敏、降级、审计从每个 Skill 里抽出来统一管理。
不改变 Skill 业务,只收口治理动作。
"""
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class HookEvent:
    """单个 Hook 事件(结构化,供 trace/审计读取)。"""

    def __init__(self, hook_type: str, target_name: str, result: str,
                 reason: str, safe_summary: Dict[str, Any],
                 redacted: bool = False, degraded: bool = False):
        self.hook_type = hook_type
        self.target_name = target_name
        self.result = result
        self.reason = reason
        self.safe_summary = safe_summary
        self.redacted = redacted
        self.degraded = degraded

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hook_type": self.hook_type, "target_name": self.target_name,
            "result": self.result, "reason": self.reason,
            "safe_summary": self.safe_summary, "redacted": self.redacted,
            "degraded": self.degraded,
        }


class SkillHooks:
    """Skill 生命周期治理点。"""

    def __init__(self):
        self.events: List[HookEvent] = []
        self.touched_skills: set = set()

    # ── 脱敏:手机号/邮箱/token ──
    @staticmethod
    def redact(text: str) -> str:
        import re
        # 手机号:1 开头 11 位数字(避免中文边界 \b 失效,直接匹配)
        text = re.sub(r"1[3-9]\d{9}", "[phone-redacted]", text)
        # 邮箱
        text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "[email-redacted]", text)
        # token/key/secret 等密钥字段
        text = re.sub(r"(?i)(api[_-]?key|secret|token)\s*[:=]\s*\S+", r"\1=[secret-redacted]", text)
        return text

    # ── pre_tool_call:执行前校验必填参数 + 风险边界 ──
    def pre_tool_call(self, skill_name: str, meta: Any, params: Dict[str, Any]) -> HookEvent:
        self.touched_skills.add(skill_name)
        missing = [p for p in (meta.required_params or []) if not params.get(p)]
        result = "blocked" if missing else "allowed"
        event = HookEvent(
            hook_type="pre_tool_call", target_name=skill_name, result=result,
            reason="Skill 执行前校验必填参数与只读边界",
            safe_summary={
                "skill": skill_name,
                "risk_level": getattr(meta, "risk_level", None).value if getattr(meta, "risk_level", None) else "low",
                "read_only": getattr(meta, "read_only", True),
                "required": getattr(meta, "required_params", []),
                "missing": missing,
            },
        )
        self.events.append(event)
        return event

    # ── post_tool_call:结果脱敏 + 摘要 ──
    def post_tool_call(self, skill_name: str, result: Dict[str, Any]) -> HookEvent:
        output = str(result.get("result") or result.get("output") or "")
        safe_output = self.redact(output)
        redacted = safe_output != output
        event = HookEvent(
            hook_type="post_tool_call", target_name=skill_name,
            result="sanitized" if redacted else "passed",
            reason="Skill 结果脱敏后进入上下文,不暴露隐私/密钥",
            safe_summary={"skill": skill_name,
                          "output_preview": safe_output[:120],
                          "redacted": redacted},
            redacted=redacted,
        )
        self.events.append(event)
        return event

    # ── on_error:异常归一成降级信号 ──
    def on_error(self, skill_name: str, error: Exception) -> HookEvent:
        event = HookEvent(
            hook_type="on_error", target_name=skill_name, result="degraded",
            reason="Skill 执行异常,归一成可读降级信号,不让链路静默失败",
            safe_summary={"skill": skill_name, "error": str(error)[:120]},
            degraded=True,
        )
        self.events.append(event)
        return event

    # ── on_completion:本轮公开治理摘要 ──
    def on_completion(self) -> Dict[str, Any]:
        event = HookEvent(
            hook_type="on_completion", target_name="skill_chain", result="completed",
            reason="本轮 Skill 链路治理完成,输出公开摘要",
            safe_summary={
                "touched_skills": sorted(self.touched_skills),
                "event_count": len(self.events),
            },
        )
        self.events.append(event)
        return event.to_dict()


skill_hooks = SkillHooks()
