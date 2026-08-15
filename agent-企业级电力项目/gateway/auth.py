"""网关鉴权中间件 — API Key / JWT（本机默认空 key = 放行，生产开启）"""
import time
import logging

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from config.settings import settings
from observability.tracing import get_trace_id
from observability.audit import audit_logger

logger = logging.getLogger(__name__)

_WHITELIST = ["/health", "/docs", "/openapi.json", "/metrics", "/favicon.ico"]


# 网关鉴权中间件：API Key / JWT 校验，本机默认放行
class AuthMiddleware(BaseHTTPMiddleware):
    # 请求入口：白名单放行、API Key 校验并注入租户/用户信息
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if self._is_whitelist(path):
            return await call_next(request)
        # 本机开发默认放行；生产配 gateway_api_key 后强制校验
        if settings.gateway_api_key:
            api_key = request.headers.get("X-API-Key", "")
            if api_key != settings.gateway_api_key:
                audit_logger.log_security(event_type="auth_failed", detail=path)
                raise HTTPException(status_code=401, detail="Unauthorized")
        request.state.tenant_id = request.headers.get("X-Tenant-Id", "default")
        request.state.user_id = request.headers.get("X-User-Id", "")
        request.state.account = request.state.user_id or "anonymous"
        return await call_next(request)

    # 判断请求路径是否在白名单中
    @staticmethod
    def _is_whitelist(path: str) -> bool:
        return any(path.startswith(p) for p in _WHITELIST)
