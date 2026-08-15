"""OpenTelemetry 追踪 — 对外保持原 API，内部换成 OTel SDK

对外契约（graph/audit/gateway 的调用点零改动）：
- tracer.span(name, **attrs)   → OTel start_as_current_span（父子自动继承，协程安全）
- get_trace_id()               → 当前 span 的 trace_id（audit 落库 / 日志直接用）
- set_trace_id() / new_trace_id() → 仅无 OTel 时降级用，OTel 模式下由根 span 决定

未安装 OTel SDK 时自动降级：span 变成 no-op，trace_id 走 contextvar，
保证本机在装依赖前也能跑（行为与升级前一致）。
"""
import contextvars
import uuid
from contextlib import contextmanager

try:
    from opentelemetry import trace as _otel_trace
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    HAS_OTEL = True
except ImportError:  # pragma: no cover - 依赖未装时的降级路径
    _otel_trace = None
    HAS_OTEL = False

# 无 OTel 时的手动 TraceID 透传（与升级前行为一致）
_manual_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="")


# OTel 缺失时的 no-op span，兼容 with 语法
class _NoopSpan:
    """OTel 缺失时的 no-op span，兼容 `with tracer.span(...)` 语法"""

    # 进入上下文管理器，返回自身
    def __enter__(self):
        return self

    # 退出上下文管理器，不吞异常
    def __exit__(self, *exc):
        return False


# 追踪器：保持旧 API tracer.span(name, **attrs)
class Tracer:
    """保持旧 API：tracer.span(name, **attrs)"""

    # 开启一个 span（OTel 用 start_as_current_span，否则返回 no-op）
    def span(self, name: str, **attrs):
        if not HAS_OTEL:
            return _NoopSpan()
        return _otel_trace.get_tracer("power-agent").start_as_current_span(
            name, attributes=attrs or None)


tracer = Tracer()


# 获取当前 span 的 trace_id（无 OTel 时读 contextvar）
def get_trace_id() -> str:
    if not HAS_OTEL:
        return _manual_trace_id.get()
    ctx = _otel_trace.get_current_span().get_span_context()
    if not ctx.is_valid:
        return ""
    return format(ctx.trace_id, "032x")


# 手动设置 trace_id（仅无 OTel 的降级路径使用）
def set_trace_id(tid: str) -> None:
    """OTel 模式下由根 span 决定 trace_id，此函数仅用于降级路径。"""
    _manual_trace_id.set(tid)


# 生成新的随机 trace_id
def new_trace_id() -> str:
    return uuid.uuid4().hex


# 为一次 HTTP 请求创建 OTel 根 span（支持 traceparent 桥接）
def request_span(request, **attrs):
    """为一次 HTTP 请求创建 OTel 根 span。

    - 入站带 W3C `traceparent` → 作为父上下文（跨服务串联）
    - 入站只有旧 `X-Trace-ID` → 桥接成 traceparent，保持连续
    - 都没有 → 新开一条 trace
    """
    if not HAS_OTEL:
        return _NoopSpan()
    carrier = {k.lower(): v for k, v in request.headers.items()}
    if "traceparent" not in carrier and "x-trace-id" in carrier:
        carrier["traceparent"] = _legacy_to_traceparent(carrier["x-trace-id"])
    ctx = TraceContextTextMapPropagator().extract(carrier)
    return _otel_trace.get_tracer("power-agent").start_as_current_span(
        "http_request", context=ctx, attributes=attrs or None)


# 将旧 X-Trace-ID 桥接为 W3C traceparent 格式
def _legacy_to_traceparent(tid: str) -> str:
    """旧 X-Trace-ID(<=16 hex) → W3C traceparent 格式，span_id 用占位。"""
    return f"00-{tid:0>32}-0000000000000001-01"
