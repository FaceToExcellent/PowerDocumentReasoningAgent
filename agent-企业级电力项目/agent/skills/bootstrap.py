"""Skill 引导注册 — 按当前领域配置注册 Skill
领域化：领域 Skill 由 DomainConfig.skill_classes 指定，底座代码零领域字眼。
"""
import logging

from config.settings import settings
from config.domain import get_domain
from agent.skills.registry import skill_registry

logger = logging.getLogger(__name__)


# 按当前领域注册全部 Skill(默认用 settings.domain)
def register_all_skills(domain_name: str = None) -> skill_registry.__class__:
    """按领域注册 Skill。domain_name 缺省用 settings.domain"""
    domain = get_domain(domain_name or settings.domain)
    for class_ref in domain.skill_classes:
        try:
            mod_path, cls_name = class_ref.split(":")
            import importlib
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name)
            skill_registry.register(cls())
        except Exception as e:
            logger.error(f"注册 Skill [{class_ref}] 失败: {e}")
    return skill_registry


# 模块导入时自动按当前领域注册
_ = register_all_skills()
