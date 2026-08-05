"""电力智能运维 Agent 企业版 — 一键启动入口

用法:
    uv run python main.py              # 默认 INFO
    uv run python main.py --debug      # DEBUG
    uv run python main.py --reload     # 热重载

本机依赖（已预装或 Docker）:
    Redis（已跑）· Milvus Lite（进程内）· Ollama（本地小模型）
    DeepSeek API（可选，配 .env DEEPSEEK_API_KEY 后核心推理走 v4 API）
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def _banner():
    print(r"""
  ╔══════════════════════════════════════════╗
  ║   ⚡ 电力智能运维 Agent  v5.0 企业版      ║
  ║   Milvus Lite · DeepSeek v4 · 分层记忆    ║
  ║   LangGraph · HITL · SSE · 多租户         ║
  ╚══════════════════════════════════════════╝
""")


def _env_check():
    from config.settings import settings
    print("🔍 环境预检 …")
    ok, fail = 0, 0

    # 1) Redis
    import redis.asyncio as aioredis
    import asyncio
    try:
        async def _p():
            r = aioredis.Redis(host=settings.redis_host, port=settings.redis_port, db=settings.redis_db)
            await r.ping(); await r.aclose()
        asyncio.run(_p())
        print(f"   ✅ Redis ({settings.redis_host}:{settings.redis_port})"); ok += 1
    except Exception as e:
        print(f"   ⚠️ Redis 不可用 ({e}) — 缓存降级内存"); ok += 1

    # 2) Ollama（本地小模型）
    import httpx
    try:
        r = httpx.get(f"{settings.ollama_host}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        print(f"   ✅ Ollama ({len(models)} 模型: {', '.join(models[:2])}…)"); ok += 1
    except Exception:
        print("   ⚠️ Ollama 不可用 — 本地小模型降级不可用"); ok += 1

    # 3) Milvus Lite
    try:
        from rag.vector_store.milvus_store import MilvusVectorStore
        _ = MilvusVectorStore()
        print("   ✅ Milvus Lite 就绪"); ok += 1
    except Exception as e:
        print(f"   ❌ Milvus 不可用: {str(e)[:80]}"); fail += 1

    # 4) DeepSeek API
    if settings.deepseek_api_key:
        print("   ✅ DeepSeek v4 API（已配置 key，核心推理走云端）")
    else:
        print("   ⚠️ DeepSeek API key 未配置 — 核心推理降级本地 deepseek-r1")
    ok += 1

    print(f"   结果: {ok} 通过, {fail} 失败")
    return fail == 0


def main():
    import argparse
    parser = argparse.ArgumentParser(description="电力智能运维 Agent 企业版")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--seed", action="store_true", help="启动前灌入演示电力文档")
    args = parser.parse_args()

    _banner()

    from config.logging_config import setup_logging
    setup_logging(level="DEBUG" if args.debug else "INFO")
    from config.logging_config import logger

    from config.settings import settings
    host = args.host or settings.api_host
    port = args.port or settings.api_port

    if not _env_check():
        logger.error("环境检查未通过")
        sys.exit(1)

    # 演示文档
    if args.seed:
        from scripts.seed_docs import seed_demo_docs
        seed_demo_docs()

    import uvicorn
    from api.main import app
    logger.info(f"🚀 启动服务: http://{host}:{port}")
    logger.info(f"   📍 健康检查: http://localhost:{port}/health")
    logger.info(f"   📍 API 文档:  http://localhost:{port}/docs")
    logger.info(f"   📍 SSE 流式:  POST /chat/stream")
    logger.info("━" * 50)
    uvicorn.run(app, host=host, port=port, log_level="info", reload=args.reload)


if __name__ == "__main__":
    main()
