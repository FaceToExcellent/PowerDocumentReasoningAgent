"""统一 LLM 结果结构"""
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class LLMResult:
    content: str = ""
    thinking: str = ""                    # deepseek reasoning_content
    backend: str = ""                     # deepseek / local_small / local_reasoning
    model: str = ""
    token_usage: Dict = field(default_factory=dict)
    raw: Optional[object] = None

    @property
    def success(self) -> bool:
        return bool(self.content)
