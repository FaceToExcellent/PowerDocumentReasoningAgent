"""网关请求日志中间件 — 记录耗时 + TraceID 注入响应头"""
import time
import logging

from starlette.middleware.base import BaseHTTPMiddleware

from config.logging_config import logger
from observability.tracing import get_trace_id, set_trace_id, new_trace_id


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        incoming = request.headers.get("X-Trace-ID", "")
        set_trace_id(incoming or new_trace_id())
        start = time.time()
        response = await call_next(request)
        duration_ms = int((time.time() - start) * 1000)
        logger.info(f"{request.method} {request.url.path} → {response.status_code} "
                    f"{duration_ms}ms trace={get_trace_id()}")
        response.headers["X-Trace-ID"] = get_trace_id()
        return response
