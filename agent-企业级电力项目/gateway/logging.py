"""网关请求日志中间件 — OTel 根 span + TraceID 注入响应头"""
import time
import logging

from starlette.middleware.base import BaseHTTPMiddleware

from config.logging_config import logger
from observability.tracing import (
    HAS_OTEL, get_trace_id, set_trace_id, new_trace_id, request_span,
)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start = time.time()
        if HAS_OTEL:
            # OTel 模式：根 span 决定 trace_id，支持 W3C traceparent 跨服务传播
            # 必须在 with 块内取 trace_id，退出后当前 span 已不 active
            with request_span(request, method=request.method, path=request.url.path):
                response = await call_next(request)
                tid = get_trace_id()
        else:
            # 降级模式：与升级前一致，contextvar 手动透传 trace_id
            incoming = request.headers.get("X-Trace-ID", "")
            set_trace_id(incoming or new_trace_id())
            response = await call_next(request)
            tid = get_trace_id()
        duration_ms = int((time.time() - start) * 1000)
        logger.info(f"{request.method} {request.url.path} → {response.status_code} "
                    f"{duration_ms}ms trace={tid}")
        response.headers["X-Trace-ID"] = tid
        return response
