"""Skill 注册中心 — 统一注册、查询、权限过滤"""
from typing import Dict, List, Optional

from agent.skills.base_skill import BaseSkill, SkillMetadata


# Skill 注册中心:统一注册、查询、权限过滤
class SkillRegistry:
    # 初始化技能与元数据字典
    def __init__(self):
        self._skills: Dict[str, BaseSkill] = {}
        self._metadata: Dict[str, SkillMetadata] = {}

    # 注册 Skill 到注册中心并缓存其元数据
    def register(self, skill: BaseSkill):
        meta = skill.metadata
        self._skills[meta.name] = skill
        self._metadata[meta.name] = meta

    # 按名称获取已注册的 Skill 实例
    def get(self, name: str) -> Optional[BaseSkill]:
        return self._skills.get(name)

    # 返回全部 Skill 的元数据列表
    def all_metadata(self) -> List[SkillMetadata]:
        return list(self._metadata.values())

    # 返回全部已注册 Skill 的名称列表
    def all_names(self) -> List[str]:
        return list(self._skills.keys())

    # 执行指定名称的 Skill(拼装上下文并调用 run)
    async def execute(self, name: str, params: dict, user_context: dict = None) -> dict:
        """执行 Skill（走 Harness 拦截在 graph 层做，这里直接执行）"""
        skill = self.get(name)
        if not skill:
            return {"success": False, "error": f"未知 Skill: {name}"}
        context = dict(params)
        if user_context:
            context["user_context"] = user_context
        return await skill.run(context)


skill_registry = SkillRegistry()
