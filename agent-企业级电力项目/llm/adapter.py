"""统一 LLM 适配器 — 模型分级 + 智能降级 + 流式
调用约定：unified_llm.ainvoke(task, messages) / astream(task, messages)
"""
import asyncio
import logging
import time
from typing import Optional, AsyncIterator

from config.settings import settings
from config.logging_config import logger
from llm.llm_result import LLMResult
from llm.model_router import model_router
from observability.metrics import metrics

logger = logging.getLogger(__name__)


class UnifiedLLM:
    """统一推理入口。按任务路由到对应后端，失败自动降级。"""

    def __init__(self):
        self._backends = {}
        self._health = {}

    def _get_backend(self, name: str):
        if name not in self._backends:
            if name == "deepseek_reasoner":
                from llm.backends.deepseek_backend import DeepSeekBackend
                self._backends[name] = DeepSeekBackend(kind="reasoner")
            elif name == "deepseek_chat":
                from llm.backends.deepseek_backend import DeepSeekBackend
                self._backends[name] = DeepSeekBackend(kind="chat")
            elif name == "local_reasoning":
                from llm.backends.local_reasoning_backend import LocalReasoningBackend
                self._backends[name] = LocalReasoningBackend()
            else:
                from llm.backends.local_small_backend import LocalSmallBackend
                self._backends[name] = LocalSmallBackend()
        return self._backends[name]

    def _chain_for(self, task: str) -> list:
        """返回按优先级排列的后端链（含降级）"""
        primary = model_router.route(task)
        chain = [primary] + model_router.fallback_for(primary)
        # 过滤不可用的（如 DeepSeek 无 key）
        filtered = []
        for name in chain:
            backend = self._get_backend(name)
            if getattr(backend, "available", True) or name in ("local_small", "local_reasoning"):
                filtered.append(name)
        if not filtered:
            filtered = ["local_small"]
        return filtered

    async def ainvoke(self, task: str, messages: list, **kwargs) -> LLMResult:
        last_error = None
        for name in self._chain_for(task):
            if not self._health.get(name, True):
                continue
            backend = self._get_backend(name)
            start = time.time()
            try:
                result = await asyncio.wait_for(backend.ainvoke(messages, **kwargs),
                                                timeout=settings.llm_timeout)
                metrics.observe("llm_latency_ms", (time.time() - start) * 1000, {"backend": name})
                metrics.incr("llm_tokens_total", result.token_usage.get("total_tokens", 0) or 0)
                return result
            except asyncio.TimeoutError:
                last_error = "timeout"
                logger.warning(f"[{task}] 后端 {name} 超时，降级…")
                self._health[name] = False
                metrics.incr("llm_fallback_total", labels={"backend": name})
            except Exception as e:
                last_error = str(e)
                logger.warning(f"[{task}] 后端 {name} 失败: {str(e)[:120]}，降级…")
                self._health[name] = False
                metrics.incr("llm_fallback_total", labels={"backend": name})
        raise RuntimeError(f"所有 LLM 后端均不可用: {last_error}")

    async def astream(self, task: str, messages: list):
        """流式调用。yield dict: {type: thinking|content, text: str, is_final: bool}"""
        chain = self._chain_for(task)
        for i, name in enumerate(chain):
            backend = self._get_backend(name)
            try:
                if name in ("deepseek_reasoner", "deepseek_chat"):
                    async for chunk in backend.astream_raw(messages):
                        yield {"type": "thinking" if chunk["thinking"] else "content",
                               "text": chunk["thinking"] or chunk["content"],
                               "is_final": chunk.get("is_final", False)}
                    return
                else:
                    # 本地小模型/推理：流式，不支持 thinking 分段就全当 content
                    async for thinking, content in backend.astream(messages):
                        if content:
                            yield {"type": "content", "text": content, "is_final": False}
                    yield {"type": "content", "text": "", "is_final": True}
                    return
            except Exception as e:
                logger.warning(f"[{task}] 流式后端 {name} 失败: {str(e)[:100]}，降级…")
                continue
        raise RuntimeError("所有 LLM 后端流式均不可用")


unified_llm = UnifiedLLM()
