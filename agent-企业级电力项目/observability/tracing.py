"""轻量 OpenTelemetry 风格追踪 — 本机用简单实现，生产可换 OTLP
本机避免引入 OTel SDK 重依赖，用 contextvar 传递 TraceID + 结构化 span 记录。
"""
import time
import uuid
import contextvars
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# 当前 TraceID / SpanID，通过 contextvar 全链路传递（协程安全）
_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="")
_span_stack: contextvars.ContextVar[list] = contextvars.ContextVar("span_stack", default=[])


@dataclass
class Span:
    name: str
    start: float
    attrs: dict = field(default_factory=dict)
    parent: Optional["Span"] = None
    children: list = field(default_factory=list)
    duration_ms: float = 0.0

    def finish(self):
        self.duration_ms = (time.time() - self.start) * 1000


class Tracer:
    """极简 Tracer：管理 TraceID 和 span 栈"""

    def start_span(self, name: str, **attrs) -> Span:
        stack = list(_span_stack.get())
        parent = stack[-1] if stack else None
        span = Span(name=name, start=time.time(), attrs=attrs, parent=parent)
        if parent is not None:
            parent.children.append(span)
        stack.append(span)
        _span_stack.set(stack)
        return span

    def end_span(self, span: Span):
        span.finish()
        stack = list(_span_stack.get())
        if stack and stack[-1] is span:
            stack.pop()
            _span_stack.set(stack)
        # 根 span 结束时打印一条链路摘要
        if span.parent is None and span.duration_ms > 50:
            logger.debug(f"[trace:{get_trace_id()}] {span.name} {span.duration_ms:.0f}ms "
                         f"attrs={span.attrs} children={len(span.children)}")

    def span(self, name: str, **attrs):
        """context manager 用法"""
        class _Ctx:
            def __enter__(self):
                self.span = tracer.start_span(name, **attrs)
                return self.span
            def __exit__(self, *exc):
                tracer.end_span(self.span)
        return _Ctx()

    def get_current_span(self) -> Optional[Span]:
        stack = _span_stack.get()
        return stack[-1] if stack else None


tracer = Tracer()


def new_trace_id() -> str:
    """生成新 TraceID"""
    return uuid.uuid4().hex[:16]


def set_trace_id(tid: str):
    _trace_id.set(tid)


def get_trace_id() -> str:
    return _trace_id.get() or ""


def trace_middleware_factory(app=None):
    """FastAPI 中间件：为每个请求生成 TraceID 并注入响应头"""
    from fastapi import Request

    async def middleware(request: Request, call_next):
        incoming = request.headers.get("X-Trace-ID", "")
        set_trace_id(incoming or new_trace_id())
        with tracer.span("http_request", path=request.url.path, method=request.method):
            response = await call_next(request)
        response.headers["X-Trace-ID"] = get_trace_id()
        return response
    return middleware
