"""ToolResult + Observation — 工具调用结果的结构化封装与 Observation 治理(L18/L20)

- ToolResult:工具/ Skill 一次调用的统一返回结构(success/tool_name/result/error/duration)
- Observation:进模型的"观察" = 脱敏摘要 + 省略的内部字段(L20:记录省略字段,只留摘要进模型)
"""
import re
from dataclasses import dataclass, field
from typing import Any, List, Tuple

# 敏感信息正则:手机号 / 邮箱 / 密钥(Observation 治理的脱敏规则)
_SENSITIVE_RULES = [
    (r"1[3-9]\d{9}", "[手机号已脱敏]"),
    (r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "[邮箱已脱敏]"),
    (r"[\"']?(api[_-]?key|access[_-]?token|secret|password|密钥)[\"']?\s*[=:：]\s*[\"']?\S+", r"\1=[已隐藏]"),
]


@dataclass
class ToolResult:
    """工具调用结果(替代裸 dict,统一 success/result/error/duration)。"""

    success: bool
    tool_name: str
    result: Any = None
    error: str = ""
    duration_ms: int = 0

    @classmethod
    def ok(cls, tool_name: str, result: Any = None, duration_ms: int = 0) -> "ToolResult":
        return cls(success=True, tool_name=tool_name, result=result, duration_ms=duration_ms)

    @classmethod
    def fail(cls, tool_name: str, error: str, duration_ms: int = 0) -> "ToolResult":
        return cls(success=False, tool_name=tool_name, error=str(error)[:500], duration_ms=duration_ms)

    def to_observation(self, max_len: int = 300) -> "Observation":
        """L20:把工具结果转成给模型的脱敏观察(摘要 + 省略字段说明)。"""
        raw = _stringify(self.result if self.success else self.error)
        text, omitted = _sanitize_and_summarize(raw, max_len)
        return Observation(text=text, source=self.tool_name,
                           omitted_fields=omitted, redacted=bool(omitted),
                           success=self.success)

    def to_dict(self) -> dict:
        """兼容 dict 消费方(如 hooks/audit)。"""
        return {"success": self.success, "tool_name": self.tool_name,
                "result": self.result, "error": self.error,
                "duration_ms": self.duration_ms}


@dataclass
class Observation:
    """进模型的"观察":脱敏摘要 + 省略字段(只留安全的给模型)。"""

    text: str
    source: str = ""
    omitted_fields: List[str] = field(default_factory=list)
    redacted: bool = False
    success: bool = True


def _stringify(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    try:
        import json
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception:
        return str(result)


def _sanitize_and_summarize(text: str, max_len: int) -> Tuple[str, List[str]]:
    """脱敏 + 截断,返回 (安全文本, 省略/脱敏字段说明)。"""
    omitted: List[str] = []
    for pattern, repl in _SENSITIVE_RULES:
        if re.search(pattern, text, re.IGNORECASE):
            text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
            if "[已隐藏]" not in repl:
                omitted.append(repl.strip("[]"))
            else:
                omitted.append("密钥已隐藏")
    if len(text) > max_len:
        omitted.append(f"长度超限(截断到{max_len})")
        text = text[:max_len] + "…"
    return text, omitted
