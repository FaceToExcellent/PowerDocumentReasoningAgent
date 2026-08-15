"""Skill 基类 — 带元数据（动态筛选基础）"""
import time
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from agent.harness.risk_level import RiskLevel

logger = logging.getLogger(__name__)


# Skill 元数据类 — 承载筛选字段与工具契约
class SkillMetadata:
    """Skill 元数据 — 供 ToolSelector 三层筛选（权限 → 租户可见 → 语义）+ 工具契约"""

    # 初始化元数据字段(缺省值保证安全:只读、空权限、全租户可见)
    def __init__(self, name: str, description: str, tags: List[str] = None,
                 risk_level: RiskLevel = RiskLevel.LOW, category: str = "检索",
                 required_permission: str = "", tenant_visible: List[str] = None,
                 read_only: bool = True, required_params: List[str] = None):
        self.name = name
        self.description = description
        self.tags = tags or []
        self.risk_level = risk_level
        self.category = category
        self.required_permission = required_permission
        self.tenant_visible = tenant_visible or []   # 空=全部租户可见
        self.read_only = read_only                   # 工具契约:是否只读(默认 True,安全)
        self.required_params = required_params or [] # 工具契约:必填参数

    # 将元数据序列化为字典,供工具列表/筛选使用
    def to_dict(self) -> dict:
        return {
            "name": self.name, "description": self.description, "tags": self.tags,
            "risk_level": self.risk_level.value, "category": self.category,
            "required_permission": self.required_permission,
            "read_only": self.read_only, "required_params": self.required_params,
        }


# Skill 抽象基类 — 所有电力 Agent Skill 必须继承实现
class BaseSkill(ABC):
    """所有电力 Agent Skill 必须继承"""

    # 元数据属性:子类须返回 SkillMetadata 供动态筛选
    @property
    @abstractmethod
    def metadata(self) -> SkillMetadata:
        ...

    # 抽象执行方法:子类实现具体 Skill 逻辑
    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        ...

    # 运行 Skill 入口:执行 execute 并附加计时与异常保护
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
