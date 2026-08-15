"""企业版全局配置 — pydantic-settings，.env 加载"""
from pydantic_settings import BaseSettings
from typing import List, Optional


# 全局配置：pydantic-settings 从 .env 加载，集中管理各模块参数
class Settings(BaseSettings):
    # ── 推理层 ──
    deepseek_api_key: str = ""                     # 从环境变量注入，不落盘
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_reasoner_model: str = "deepseek-reasoner"
    deepseek_chat_model: str = "deepseek-chat"
    ollama_host: str = "http://127.0.0.1:11434"
    local_small_model: str = "qwen2.5:7b-instruct-q4_K_M"   # 轻量任务（意图/闲聊/校验）
    local_reasoning_model: str = "deepseek-r1:7b"           # 本地兜底推理（无 API key 时）
    llm_timeout: int = 120
    llm_max_tokens: int = 2048
    llm_temperature: float = 0.3

    # ── 向量库（Milvus Lite 本地 / 生产 tcp）──
    # pymilvus 3.x：传 .db 文件路径即自动本地模式（无需 lite:// 前缀）；生产传 tcp://host:19530
    milvus_db_uri: str = "./data/milvus.db"
    vector_store_type: str = "milvus"                 # milvus / chroma
    chroma_persist_dir: str = "./data/chroma_db"
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "mps"                     # Apple MPS；Linux 改 cuda / cpu
    milvus_dim: int = 1024                            # BGE-M3 输出维度
    reranker_model: str = "BAAI/bge-reranker-base"    # cross-encoder 重排(可选,无则本地兜底)

    # ── 缓存 ──
    redis_host: str = "127.0.0.1"
    redis_port: int = 6379
    redis_db: int = 0
    redis_max_connections: int = 10
    cache_default_ttl: int = 3600
    rag_cache_ttl: int = 1800

    # ── 服务 ──
    api_host: str = "0.0.0.0"
    api_port: int = 8090
    api_reload: bool = False
    log_level: str = "INFO"
    log_dir: str = "./logs"

    # ── 记忆 / 图 ──
    max_iterations: int = 3
    max_context_tokens: int = 2000
    checkpoint_db: str = "./data/checkpoints.db"
    sqlite_audit_db: str = "./data/audit.db"
    max_recent_rounds: int = 8
    graph_timeout_seconds: int = 180

    # ── 安全 / 网关 ──
    gateway_enabled: bool = True
    gateway_api_key: str = ""          # 预留：MCP/租户 API Key（本机默认空=放行）
    rate_limit_per_minute: int = 60    # 单租户每分钟限流

    # ── 观测 / 追踪（OTel）──
    otel_enabled: bool = False         # true 时把 span 推到 OTLP 后端（.env 用 OTEL_ENABLED 覆盖）

    # ── 电力业务 ──
    power_voltage_levels: List[str] = ["10", "35", "110", "220", "500", "750"]
    default_doc_chunk_size: int = 512
    doc_ingest_workers: int = 2

    # ── 领域化 ──
    domain: str = "power"          # power（电力）/ generic（通用文档）——换领域只改这里

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


settings = Settings()
