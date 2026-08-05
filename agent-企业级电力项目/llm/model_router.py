"""模型分级路由 — 核心推理 DeepSeek v4 API / 轻量任务本地小模型"""
import logging

logger = logging.getLogger(__name__)


class ModelRouter:
    """按任务复杂度选模型（M8.x 延迟优化的核心）"""

    # 任务 → 后端名
    TASK_BACKEND = {
        "supervisor": "local_small",        # 意图识别：本地小模型够用
        "chat": "local_small",              # 闲聊：本地小模型
        "spec_retrieval": "deepseek",       # 规程问答：DeepSeek v4 API
        "cost_audit": "deepseek",
        "doc_archive": "local_small",
        "grid_op": "deepseek",
        "fault_disposal": "deepseek",
        "comparison_analysis": "deepseek",  # 分析类：DeepSeek v4 API
        "fact_check": "local_small",        # 校验：本地小模型够用
    }

    # 允许降级的后端
    FALLBACK_CHAIN = {
        "deepseek": ["local_reasoning", "local_small"],   # API 挂了 → 本地推理 → 本地小模型
        "local_reasoning": ["local_small"],
        "local_small": [],
    }

    def route(self, task: str) -> str:
        return self.TASK_BACKEND.get(task, "local_small")

    def fallback_for(self, backend: str) -> list:
        return self.FALLBACK_CHAIN.get(backend, [])


model_router = ModelRouter()
