"""领域配置基类 + 领域工厂

领域化设计核心：**领域只在配置层，不在代码层**。
- 意图关键词、领域提示词、领域 Skill、演示文档全部由 DomainConfig 提供
- 换领域 = 换一个 DomainConfig 子类，底座（记忆/HITL/检索/多租户/SSE）零改动
"""
from abc import ABC
from typing import Dict, List


# 领域配置基类：定义意图词、提示词、Skill 与演示文档等领域化内容
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

    # 返回当前领域支持的意图关键词列表
    def get_intents(self) -> List[str]:
        return list(self.intent_keywords.keys())

    # 意图映射到主 Skill 名（供 agent_execute 使用），子类可覆盖
    def intent_to_skill(self, intent: str) -> str:
        """意图 → 主 Skill 名（供 agent_execute 用）；子类可覆盖"""
        return ""

    # ⭐ Prompt Registry:把意图提示词片段化,带版本,支持按意图选择
    @property
    def prompt_registry(self) -> Dict:
        """返回结构化 Prompt 注册表:片段ID/优先级/适用意图/内容 + 版本指纹。"""
        import hashlib
        fragments = []
        for intent, content in self.intent_prompts.items():
            fragments.append({
                "fragment_id": f"{self.name}_{intent}",
                "priority": 10,
                "enabled": True,
                "applies_to": [intent],
                "content": content,
            })
        payload = "|".join(self.intent_prompts.values())
        return {
            "domain": self.name,
            "version": hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8],
            "template": "domain_intent_prompts_v1",
            "fragments": fragments,
            "fragment_count": len(fragments),
        }

    # 按意图挑选启用的 Prompt 片段，未匹配时回退到 chat 或第一个片段
    def select_prompt_fragments(self, intent: str) -> List[Dict]:
        """按意图选择 Prompt 片段。"""
        reg = self.prompt_registry
        selected = [f for f in reg["fragments"]
                    if f["enabled"] and intent in f["applies_to"]]
        # 兜底:意图未匹配时至少给 chat 或第一个
        if not selected:
            fallback = [f for f in reg["fragments"] if "chat" in f["applies_to"]]
            selected = fallback or reg["fragments"][:1]
        return selected


# 领域工厂：按名称实例化对应的领域配置
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
