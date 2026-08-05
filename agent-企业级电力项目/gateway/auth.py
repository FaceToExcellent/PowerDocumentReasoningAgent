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


class AuthMiddleware(BaseHTTPMiddleware):
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

    @staticmethod
    def _is_whitelist(path: str) -> bool:
        return any(path.startswith(p) for p in _WHITELIST)
