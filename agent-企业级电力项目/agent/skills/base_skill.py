"""Skill 基类 — 带元数据（动态筛选基础）"""
import time
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from agent.harness.risk_level import RiskLevel

logger = logging.getLogger(__name__)


class SkillMetadata:
    """Skill 元数据 — 供 ToolSelector 三层筛选（权限 → 租户可见 → 语义）"""

    def __init__(self, name: str, description: str, tags: List[str] = None,
                 risk_level: RiskLevel = RiskLevel.LOW, category: str = "检索",
                 required_permission: str = "", tenant_visible: List[str] = None):
        self.name = name
        self.description = description
        self.tags = tags or []
        self.risk_level = risk_level
        self.category = category
        self.required_permission = required_permission
        self.tenant_visible = tenant_visible or []   # 空=全部租户可见

    def to_dict(self) -> dict:
        return {
            "name": self.name, "description": self.description, "tags": self.tags,
            "risk_level": self.risk_level.value, "category": self.category,
            "required_permission": self.required_permission,
        }


class BaseSkill(ABC):
    """所有电力 Agent Skill 必须继承"""

    @property
    @abstractmethod
    def metadata(self) -> SkillMetadata:
        ...

    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        ...

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """运行 Skill，带计时和异常保护"""
        start = time.time()
        try:
            result = await self.execute(context)
            duration_ms = int((time.time() - start) * 1000)
            result.setdefault("success", True)
            result.setdefault("duration_ms", duration_ms)
            return result
        except Exception as e:
            logger.error(f"Skill [{self.metadata.name}] 执行失败: {e}", exc_info=True)
            return {
                "success": False, "error": str(e), "confidence": 0.0,
                "duration_ms": int((time.time() - start) * 1000),
            }
