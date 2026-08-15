"""本地推理后端 — Ollama deepseek-r1（无 API key 时的本地兜底推理）"""
import logging

from config.settings import settings
from llm.llm_result import LLMResult
from llm.backends.local_small_backend import LocalSmallBackend

logger = logging.getLogger(__name__)


# 本地推理后端：基于 Ollama deepseek-r1，离线可拿 reasoning 思考
class LocalReasoningBackend(LocalSmallBackend):
    """基于本地 deepseek-r1:7b，离线可拿思考（Ollama 原生返回 reasoning）"""

    # 初始化推理后端，沿用本地小模型基类并标记后端名
    def __init__(self, model: str = "", base_url: str = ""):
        super().__init__(model or settings.local_reasoning_model, base_url)
        self.backend_name = "local_reasoning"

    # 调用本地推理模型，提取 reasoning 并封装为 LLMResult
    async def ainvoke(self, messages: list, **kwargs) -> LLMResult:
        client = self._client()
        resp = await client.ainvoke(messages)
        thinking = ""
        # Ollama deepseek-r1 在额外 kwargs 里返回 reasoning
        try:
            thinking = (getattr(resp, "additional_kwargs", {}) or {}).get("reasoning_content", "")
            if not thinking:
                thinking = (getattr(resp, "response_metadata", {}) or {}).get("reasoning_content", "")
        except Exception:
            thinking = ""
        return LLMResult(
            content=resp.content or "",
            thinking=thinking,
            backend="local_reasoning",
            model=self.model,
            token_usage=getattr(resp, "usage_metadata", {}) or {},
        )
