"""网关限流中间件 — 滑动窗口（Redis，本机降级内存计数）"""
import time
import logging

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from config.settings import settings
from config.cache import cache_service

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not settings.gateway_enabled:
            return await call_next(request)
        tenant_id = getattr(request.state, "tenant_id", "anonymous")
        path = request.url.path
        limit_key = f"rate:{tenant_id}:{path}"

        # 滑动窗口简化：1 分钟窗口计数（本机够用；生产用 Lua 脚本）
        count = await cache_service.incr(limit_key)
        if count == 1:
            await cache_service.expire(limit_key, 60)

        if count > settings.rate_limit_per_minute:
            raise HTTPException(status_code=429, detail="Too Many Requests",
                                headers={"Retry-After": "60"})
        return await call_next(request)
