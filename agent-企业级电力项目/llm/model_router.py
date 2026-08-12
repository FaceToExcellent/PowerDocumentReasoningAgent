"""模型分级路由 — DeepSeek v4 API(reasoner 推理 / chat 轻量)+ 本地降级"""
import logging

logger = logging.getLogger(__name__)


class ModelRouter:
    """按任务复杂度选模型。轻量任务走 deepseek-chat(便宜快),核心推理走 deepseek-reasoner,本地仅降级。"""

    # 任务 → 后端名
    TASK_BACKEND = {
        "supervisor": "deepseek_chat",      # 意图识别：DeepSeek chat（语义路由优于关键词）
        "chat": "deepseek_chat",            # 闲聊：DeepSeek chat
        "spec_retrieval": "deepseek_reasoner",  # 规程问答：DeepSeek reasoner
        "cost_audit": "deepseek_reasoner",
        "doc_archive": "deepseek_chat",
        "grid_op": "deepseek_reasoner",
        "fault_disposal": "deepseek_reasoner",
        "comparison_analysis": "deepseek_reasoner",  # 分析类：DeepSeek reasoner
        "fact_check": "deepseek_chat",      # 校验：DeepSeek chat
        # 文档推理专项(步骤6):问答走 chat,对比/总结走 reasoner
        "doc_qa": "deepseek_chat",
        "doc_compare": "deepseek_reasoner",
        "doc_summary": "deepseek_reasoner",
    }

    # 允许降级的后端（API 挂了 → 本地推理 → 本地小模型）
    FALLBACK_CHAIN = {
        "deepseek_reasoner": ["deepseek_chat", "local_reasoning", "local_small"],
        "deepseek_chat": ["local_small"],
        "local_reasoning": ["local_small"],
        "local_small": [],
    }

    def route(self, task: str) -> str:
        return self.TASK_BACKEND.get(task, "deepseek_chat")

    def fallback_for(self, backend: str) -> list:
        return self.FALLBACK_CHAIN.get(backend, [])


model_router = ModelRouter()
