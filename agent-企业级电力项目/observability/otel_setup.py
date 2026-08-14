"""OTel 初始化：TracerProvider + OTLP Exporter + 采样（幂等）

在 api/main.py 的 lifespan 启动时调用一次 setup_otel()，
退出时调用 shutdown_otel() 刷空待导出的 span。

本地验证：起一个 Jaeger（已默认开启 OTLP）后访问 16686 端口看 span 树。
    docker run -d --name jaeger -p 16686:16686 -p 4317:4317 -p 4318:4318 \
      jaegertracing/all-in-one:latest
"""
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

_initialized = False


def setup_otel(
    service_name: str = "power-agent",
    sample_ratio: float = 1.0,
    otlp_endpoint: str = "http://localhost:4318/v1/traces",
) -> None:
    """初始化全局 TracerProvider。重复调用幂等（第二次直接跳过）。"""
    global _initialized
    if _initialized:
        return

    resource = Resource.create({"service.name": service_name})
    # 父采样：根按比例采样，子 span 跟随父决策（健康链路保留/慢链路可由 collector 端补充）
    provider = TracerProvider(
        resource=resource,
        sampler=ParentBased(root=TraceIdRatioBased(sample_ratio)),
    )
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint))
    )
    trace.set_tracer_provider(provider)
    _initialized = True


def shutdown_otel() -> None:
    """应用退出时刷空待导出的 span。"""
    provider = trace.get_tracer_provider()
    shutdown = getattr(provider, "shutdown", None)
    if shutdown:
        shutdown()
