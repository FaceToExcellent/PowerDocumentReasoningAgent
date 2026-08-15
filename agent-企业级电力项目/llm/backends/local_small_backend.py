"""本地小模型后端 — Ollama（轻量任务 + 兜底）"""
import logging
from typing import Dict, Optional

from config.settings import settings
from llm.llm_result import LLMResult

logger = logging.getLogger(__name__)


# 本地小模型后端：基于 langchain_ollama 的 qwen2.5，承担轻量任务与兜底
class LocalSmallBackend:
    """基于 langchain_ollama，本地 qwen2.5 小模型（意图/闲聊/校验）"""

    # 初始化小模型的模型名与 Ollama 地址
    def __init__(self, model: str = "", base_url: str = ""):
        self.model = model or settings.local_small_model
        self.base_url = base_url or settings.ollama_host

    # 创建并返回 ChatOllama 客户端
    def _client(self):
        from langchain_ollama import ChatOllama
        return ChatOllama(
            base_url=self.base_url, model=self.model,
            temperature=settings.llm_temperature,
            num_predict=settings.llm_max_tokens,
            num_ctx=8192,
        )

    # 调用本地小模型非流式接口并封装为 LLMResult
    async def ainvoke(self, messages: list, **kwargs) -> LLMResult:
        client = self._client()
        resp = await client.ainvoke(messages)
        usage = getattr(resp, "usage_metadata", {}) or {}
        return LLMResult(
            content=resp.content or "",
            thinking="",
            backend="local_small",
            model=self.model,
            token_usage=usage,
        )

    # 流式调用小模型，逐块 yield (thinking, content) 片段
    async def astream(self, messages: list):
        """流式（小模型），yield (thinking, content) 片段"""
        client = self._client()
        async for chunk in client.astream(messages):
            yield "", chunk.content or ""
