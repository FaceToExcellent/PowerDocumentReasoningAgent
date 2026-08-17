"""Hooks 治理层 — Skill 生命周期统一治理点(pre/post/error/completion)

把参数校验、脱敏、降级、审计从每个 Skill 里抽出来统一管理。
不改变 Skill 业务,只收口治理动作。
"""
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Hook 事件结构化载体,供 trace/审计读取
class HookEvent:
    """单个 Hook 事件(结构化,供 trace/审计读取)。"""

    # 初始化事件字段
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

    # 将事件转成字典供审计读取
    def to_dict(self) -> Dict[str, Any]:
        return {
            "hook_type": self.hook_type, "target_name": self.target_name,
            "result": self.result, "reason": self.reason,
            "safe_summary": self.safe_summary, "redacted": self.redacted,
            "degraded": self.degraded,
        }


# Skill 生命周期治理点集合
class SkillHooks:
    """Skill 生命周期治理点。"""

    # 初始化事件列表与已触达 Skill 集合
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

    # ── post_tool_call:结果脱敏 + 摘要 + omitted_fields 字段裁剪 ──
    def post_tool_call(self, skill_name: str, result) -> HookEvent:
        from agent.tool_result import ToolResult
        if isinstance(result, ToolResult):
            # ToolResult → Observation(已脱敏摘要 + 省略字段),L20
            obs = result.to_observation()
            output, omitted_fields, redacted = obs.text, list(obs.omitted_fields), obs.redacted
        else:
            output = str(result.get("result") or result.get("output") or "")
            safe_output = self.redact(output)
            redacted = safe_output != output
            output = safe_output
            # ⭐ Observation 治理:记录省略的内部字段,只留摘要进模型
            omitted_fields = self._collect_omitted(result)
        event = HookEvent(
            hook_type="post_tool_call", target_name=skill_name,
            result="sanitized" if (redacted or omitted_fields) else "passed",
            reason="Skill 结果脱敏 + 字段裁剪后进入上下文,不暴露隐私/密钥/内部字段",
            safe_summary={
                "skill": skill_name,
                "output_preview": output[:120],
                "omitted_fields": omitted_fields,
                "redacted": redacted,
            },
            redacted=redacted,
        )
        self.events.append(event)
        return event

    # 识别应省略的内部字段
    @staticmethod
    def _collect_omitted(result: Dict[str, Any]) -> List[str]:
        """识别应省略的内部字段(诊断/内部细节不进上下文)。"""
        omitted = []
        # 整对象返回:除白名单外都算省略
        safe_keys = {"success", "result", "output", "answer", "analysis",
                     "summary", "evidence", "citations", "confidence", "duration_ms",
                     "entities", "kg_evidence", "documents", "total_found", "query",
                     "answer_path", "matched_chunk_ids"}
        for key in result.keys():
            if key not in safe_keys and key not in omitted:
                omitted.append(key)
        return omitted[:10]

    # ── on_error:异常归一成降级信号(带 error_category 分类) ──
    def on_error(self, skill_name: str, error: Exception) -> HookEvent:
        from agent.degradation import classify_error
        category = classify_error(error)
        event = HookEvent(
            hook_type="on_error", target_name=skill_name, result="degraded",
            reason="Skill 执行异常,归一成可读降级信号,不让链路静默失败",
            safe_summary={"skill": skill_name, "error": str(error)[:120],
                          "error_category": category},
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
