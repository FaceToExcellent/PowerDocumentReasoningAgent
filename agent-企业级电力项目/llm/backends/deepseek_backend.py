"""DeepSeek v4 API 后端 — 核心推理（reasoner），原生流式拿 thinking"""
import logging
from typing import Optional, AsyncIterator

import httpx

from config.settings import settings
from llm.llm_result import LLMResult

logger = logging.getLogger(__name__)


class DeepSeekBackend:
    """DeepSeek v4 API。API key 从环境变量注入，代码不落盘。"""

    def __init__(self, model: str = "", base_url: str = "", api_key: str = ""):
        self.model = model or settings.deepseek_reasoner_model
        self.base_url = (base_url or settings.deepseek_base_url).rstrip("/")
        self.api_key = api_key or settings.deepseek_api_key

    @property
    def available(self) -> bool:
        """无 API key 时不可用（自动降级本地）"""
        return bool(self.api_key)

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    async def ainvoke(self, messages: list, **kwargs) -> LLMResult:
        if not self.available:
            raise RuntimeError("DeepSeek API key 未配置")
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", settings.llm_temperature),
            "max_tokens": kwargs.get("max_tokens", settings.llm_max_tokens),
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=settings.llm_timeout) as client:
            resp = await client.post(f"{self.base_url}/chat/completions",
                                     json=payload, headers=self._headers())
            resp.raise_for_status()
            data = resp.json()
        choice = data["choices"][0]
        return LLMResult(
            content=choice["message"].get("content", ""),
            thinking=choice["message"].get("reasoning_content", ""),
            backend="deepseek",
            model=self.model,
            token_usage=data.get("usage", {}),
        )

    async def astream_raw(self, messages: list) -> AsyncIterator[dict]:
        """原生 HTTP 流式，完整获取 reasoning_content（thinking）。
        yield {"thinking": str, "content": str, "is_final": bool}
        """
        if not self.available:
            raise RuntimeError("DeepSeek API key 未配置")
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": settings.llm_temperature,
            "max_tokens": settings.llm_max_tokens,
            "stream": True,
        }
        async with httpx.AsyncClient(timeout=settings.llm_timeout) as client:
            async with client.stream("POST", f"{self.base_url}/chat/completions",
                                     json=payload, headers=self._headers()) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    if line == "data: [DONE]":
                        yield {"thinking": "", "content": "", "is_final": True}
                        return
                    try:
                        delta = json_loads(line[6:])["choices"][0]["delta"]
                    except Exception:
                        continue
                    yield {
                        "thinking": delta.get("reasoning_content", ""),
                        "content": delta.get("content", ""),
                        "is_final": False,
                    }


def json_loads(s: str) -> dict:
    import json
    return json.loads(s)
