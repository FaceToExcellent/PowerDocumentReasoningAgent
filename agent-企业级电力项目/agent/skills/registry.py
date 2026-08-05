"""Skill 注册中心 — 统一注册、查询、权限过滤"""
from typing import Dict, List, Optional

from agent.skills.base_skill import BaseSkill, SkillMetadata


class SkillRegistry:
    def __init__(self):
        self._skills: Dict[str, BaseSkill] = {}
        self._metadata: Dict[str, SkillMetadata] = {}

    def register(self, skill: BaseSkill):
        meta = skill.metadata
        self._skills[meta.name] = skill
        self._metadata[meta.name] = meta

    def get(self, name: str) -> Optional[BaseSkill]:
        return self._skills.get(name)

    def all_metadata(self) -> List[SkillMetadata]:
        return list(self._metadata.values())

    def all_names(self) -> List[str]:
        return list(self._skills.keys())

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
