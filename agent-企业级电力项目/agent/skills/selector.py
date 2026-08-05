"""Skill 动态选择器 — 三层筛选（权限 → 租户可见 → 语义 Top-K）
先筛选后使用，避免全量注入 Prompt 导致上下文膨胀。
"""
import logging
from typing import List, Dict

from agent.skills.base_skill import SkillMetadata
from agent.skills.registry import skill_registry

logger = logging.getLogger(__name__)


class SkillSelector:
    def __init__(self):
        # 简单语义打分：query 关键词与 skill 描述/标签的重叠度
        # 生产可换 BGE-M3 向量语义排序（M5.6 计划）
        pass

    def select_skills(self, query: str, user_context: Dict = None,
                      top_k: int = 5) -> List[SkillMetadata]:
        user_context = user_context or {}
        perms = user_context.get("permissions", [])
        tenant = user_context.get("tenant_id", "")

        all_skills = skill_registry.all_metadata()

        # L1 权限过滤
        s1 = [s for s in all_skills
              if not s.required_permission or s.required_permission in perms]
        # L2 租户可见性过滤
        s2 = [s for s in s1 if not s.tenant_visible or tenant in s.tenant_visible]
        # L3 语义打分：query 与 description/tags 关键词重叠
        scored = [(self._score(query, s), s) for s in s2]
        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [s for score, s in scored if score > 0][:top_k]
        # 兜底：一个都选不中时取前 top_k 个
        if not selected:
            selected = s2[:top_k]
        return selected

    @staticmethod
    def _score(query: str, meta: SkillMetadata) -> int:
        q = query.lower()
        score = 0
        # 标签优先加权：查询词命中标签，说明这就是该 Skill 的意图域
        for tag in meta.tags:
            if tag in q:
                score += 3
        # 描述次要
        if meta.name in q:
            score += 2
        pool = (meta.description + " " + " ".join(meta.tags)).lower()
        for ch in set(q):
            if len(ch) > 1 and ch in pool:
                score += 1
        return score

    def format_for_prompt(self, skills: List[SkillMetadata]) -> str:
        lines = []
        for i, s in enumerate(skills, 1):
            lines.append(f"{i}. {s.name}：{s.description}（{s.category}）")
        return "\n".join(lines) if lines else "（无可用工具）"


skill_selector = SkillSelector()
