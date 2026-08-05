"""领域配置基类 + 领域工厂

领域化设计核心：**领域只在配置层，不在代码层**。
- 意图关键词、领域提示词、领域 Skill、演示文档全部由 DomainConfig 提供
- 换领域 = 换一个 DomainConfig 子类，底座（记忆/HITL/检索/多租户/SSE）零改动
"""
from abc import ABC
from typing import Dict, List


class DomainConfig(ABC):
    """领域配置：定义某领域的意图词、提示词、Skill、演示文档。

    子类直接声明类属性覆盖默认值即可，无需调用父类构造。
    """

    name: str = "generic"
    label: str = "通用文档分析"
    description: str = ""
    intent_keywords: Dict[str, List[str]] = {}
    intent_prompts: Dict[str, str] = {}
    skill_classes: List[str] = []
    demo_docs: List[Dict] = []
    chat_intent: str = "chat"

    def get_intents(self) -> List[str]:
        return list(self.intent_keywords.keys())

    def intent_to_skill(self, intent: str) -> str:
        """意图 → 主 Skill 名（供 agent_execute 用）；子类可覆盖"""
        return ""


def get_domain(name: str) -> DomainConfig:
    """领域工厂：按名字实例化领域配置"""
    if name == "power":
        from config.domains.power import PowerDomainConfig
        return PowerDomainConfig()
    if name == "generic":
        from config.domains.generic import GenericDomainConfig
        return GenericDomainConfig()
    from config.domains.generic import GenericDomainConfig
    return GenericDomainConfig()
