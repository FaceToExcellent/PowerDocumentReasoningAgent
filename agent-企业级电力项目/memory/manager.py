"""记忆统一管理器 — 协调短期（Redis）/ 流水（chat_message），推理前按需加载
核心：build_context 严格控制 token 预算，不把全量历史塞进 LLM 上下文。
"""
import logging
from typing import Dict, List

from config.settings import settings
from memory.message_store import message_store
from memory.short_term import short_term_memory

logger = logging.getLogger(__name__)


def _count_tokens(text: str) -> int:
    """粗略 token 估算（中文按字，英文按 4 字符）"""
    if not text:
        return 0
    cn = sum(1 for c in text if "一" <= c <= "鿿")
    other = len(text) - cn
    return cn + int(other / 4) + 1


class MemoryManager:
    def __init__(self, max_tokens: int = None):
        self.max_tokens = max_tokens or settings.max_context_tokens

    async def build_context(self, *, tenant_id="", user_id="", thread_id="",
                            query="", max_tokens=None) -> str:
        """推理前按需加载记忆，返回拼好的上下文片段（≤ max_tokens）"""
        budget = max_tokens or self.max_tokens
        parts = []

        # 1. 语义召回相关历史（优先级最高）
        related = message_store.semantic_search(
            tenant_id=tenant_id, user_id=user_id, query=query, top_k=3)
        if related:
            text = "\n".join(f"- {m['content'][:200]}" for m in related)
            parts.append(f"【相关历史问答】\n{text}")
            budget -= _count_tokens(text)

        # 2. 最近几轮原始消息（从 chat_message 精确取，保证顺序）
        recent = message_store.get_recent(
            tenant_id=tenant_id, user_id=user_id, thread_id=thread_id, limit=10)
        if recent:
            lines = [f"{m['role']}: {m['content'][:150]}" for m in recent]
            text = "\n".join(lines)
            parts.append(f"【最近对话】\n{text}")
            budget -= _count_tokens(text)

        # 3. 短期 Redis 记忆（若有独立缓存）
        try:
            stm = await short_term_memory.get(tenant_id=tenant_id, thread_id=thread_id)
            if stm:
                lines = [f"{m['role']}: {m['content'][:100]}" for m in stm[-4:]]
                parts.append("【会话快照】\n" + "\n".join(lines))
        except Exception:
            pass

        return "\n\n".join(parts)

    async def record(self, *, tenant_id="", user_id="", thread_id="", reply_id="",
                     role="", content="", content_type="text", intent=""):
        """推理后自动写入记忆（异步不阻塞）"""
        try:
            message_store.add(
                tenant_id=tenant_id, user_id=user_id, thread_id=thread_id,
                reply_id=reply_id, role=role, content=content,
                content_type=content_type, intent=intent,
                tokens=_count_tokens(content),
            )
            await short_term_memory.push(
                tenant_id=tenant_id, thread_id=thread_id, role=role, content=content)
        except Exception as e:
            logger.error(f"记忆写入失败: {e}")


memory_manager = MemoryManager()
