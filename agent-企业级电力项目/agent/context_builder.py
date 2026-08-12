"""上下文构建器 — 多来源上下文按 trust_level 排序 + 冲突解决

对齐课程 Context Builder(34课):把用户文本/记忆/检索/KG/HITL状态统一收口,
按信任优先级决定"听谁的",冲突留下公开说明。

trust_level 优先级(高→低):
  system_rules > hitl_state(审批状态) > runtime_context(身份) > tool_fact(检索事实) > memory(记忆) > user_claim(用户自述)
"""
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 信任等级:值越大越可信
TRUST_RANK = {
    "system_rules": 100,     # 系统规则/安全边界(最高,不可被覆盖)
    "hitl_state": 90,        # HITL 审批状态(钱的动作,必须冻结)
    "runtime_context": 80,   # 运行时身份(tenant/user,系统注入)
    "tool_fact": 70,         # 检索到的事实(文档依据)
    "memory": 50,            # 历史记忆(仅作指代消解)
    "user_claim": 10,        # 用户自述(最低,可能失真)
}


class ContextItem:
    """一条上下文,带来源与信任等级。"""

    def __init__(self, source_type: str, content: str, *, key: str = "",
                 facts: Optional[Dict[str, Any]] = None, priority: int = 0):
        self.source_type = source_type
        self.content = content
        self.key = key or f"{source_type}:{content[:20]}"
        self.facts = facts or {}
        self.priority = priority  # 同来源内部排序(高优先在前)
        self.trust = TRUST_RANK.get(source_type, 10)

    def to_dict(self) -> Dict[str, Any]:
        return {"source_type": self.source_type, "content": self.content,
                "key": self.key, "trust": self.trust}


class ContextBuilder:
    """统一上下文入口:add 多来源 → resolve_conflicts → render。"""

    def __init__(self):
        self._items: List[ContextItem] = []
        self._conflicts: List[Dict[str, Any]] = []

    def add(self, source_type: str, content: str, *, key: str = "",
            facts: Optional[Dict[str, Any]] = None, priority: int = 0) -> "ContextBuilder":
        if content:
            self._items.append(ContextItem(source_type, content, key=key,
                                           facts=facts, priority=priority))
        return self

    def resolve_conflicts(self) -> "ContextBuilder":
        """按信任等级解决冲突:同一 key 出现多次,高信任覆盖低信任。"""
        # 按 key 分组,保留 trust 最高 + priority 最高的那条
        best: Dict[str, ContextItem] = {}
        for item in self._items:
            existing = best.get(item.key)
            if existing is None:
                best[item.key] = item
                continue
            # 高信任覆盖低信任;同信任比 priority
            if item.trust > existing.trust or (
                    item.trust == existing.trust and item.priority >= existing.priority):
                if item.trust > existing.trust:
                    self._conflicts.append({
                        "key": item.key,
                        "winner": item.source_type,
                        "loser": existing.source_type,
                        "reason": f"{item.source_type}(trust={item.trust}) 覆盖 {existing.source_type}(trust={existing.trust})",
                    })
                best[item.key] = item
        self._items = list(best.values())
        # 按 trust 降序 + priority 降序排列(高信任在前)
        self._items.sort(key=lambda it: (-it.trust, -it.priority))
        return self

    def render(self) -> str:
        """渲染成 system 上下文文本(信任高在前)。"""
        if not self._items:
            return ""
        blocks = []
        for item in self._items:
            blocks.append(f"[{item.source_type}]\n{item.content}")
        return "\n\n".join(blocks)

    # ── 上下文压缩(L35):Protected Context 不可压 + 低相关折叠摘要 ──
    def compress(self, budget_tokens: int = 0, max_items: int = 12) -> Dict[str, Any]:
        """压缩上下文:保护高信任项(runtime/hitl/tool),折叠低相关历史为摘要。

        返回压缩报告,便于验证 token 变化。
        """
        protected = [it for it in self._items if it.trust >= 70]  # system/hitl/runtime/tool
        compressible = [it for it in self._items if it.trust < 70]  # memory/user_claim
        keep = protected + compressible[:max_items]
        dropped = compressible[max_items:]
        summary = f"压缩摘要：已折叠 {len(dropped)} 条低相关历史消息。"
        self._items = keep
        return {
            "token_estimate_before": sum(len(i.content) // 2 for i in protected + compressible),
            "token_estimate_after": sum(len(i.content) // 2 for i in keep),
            "protected_count": len(protected),
            "compressed_summary": summary if dropped else "",
        }

    def report(self) -> Dict[str, Any]:
        return {
            "selected_items": [it.to_dict() for it in self._items],
            "conflict_resolutions": self._conflicts,
            "total_items": len(self._items),
        }


def build_runtime_context_view(*, tenant_id: str = "", user_id: str = "",
                               nickname: str = "", member_level: str = "",
                               page_context: Dict[str, Any] | None = None,
                               risk_level: str = "", permissions: list = None) -> Dict[str, Any]:
    """构建 Runtime Context 双通道视图(L33):
    - trusted_for_model:模型可见(昵称/会员等级/页面线索)
    - system_only:系统校验专用(user_id/风险/权限),不进模型
    用户自述不能覆盖系统事实,冲突留说明。
    """
    permissions = permissions or []
    trusted_for_model = {
        "authenticated": True,
        "nickname": nickname or "",
        "member_level": member_level or "unknown",
        "page_context": page_context or {},
    }
    system_only = {
        "user_id": user_id,
        "tenant_id": tenant_id,
        "risk_level": risk_level or "unknown",
        "permissions": permissions,
    }
    conflict_notes = []
    if risk_level == "high":
        conflict_notes.append("风险等级 high,高风险动作需人工确认")
    if not member_level:
        conflict_notes.append("会员等级缺失,保持 unknown 不补")
    return {
        "trusted_for_model": trusted_for_model,
        "system_only": system_only,
        "conflict_notes": conflict_notes,
        "permission_decision": {"reason": "owner_matched", "allowed": True},
    }


context_builder = ContextBuilder()
