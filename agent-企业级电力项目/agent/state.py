"""AgentState — LangGraph 共享状态（企业版：tenant_id / reply_id / routing）"""
from typing import TypedDict, Optional, List, Dict, Any, Annotated
import operator


# ── 并行分支合并 reducer ──────────────────────────
# 多 agent 并行(fan_out_parallel → Send,子图分支)时,多个分支会写同一批共享通道;
# 无 reducer 的通道会在扇入合并时报 INVALID_CONCURRENCY。所有通道统一给合并策略。
def _first_non_empty(current, new):
    """保留第一个非空值(身份型/不变字段,并行分支值相同)。"""
    return current if current not in (None, "") else new


def _last_value(current, new):
    """保留最后一个值(输出/校验/状态迁移,顺序与重试语义一致)。"""
    return new


def _concat_lists(current, new):
    """列表拼接(工具调用/校验错误/钩子事件等)。"""
    return list(current or []) + list(new or [])


def _max_value(current, new):
    """取最大值(迭代次数等)。"""
    try:
        return max(current, new)
    except (TypeError, ValueError):
        return new


# AgentState — LangGraph 全局共享状态字段定义(TypedDict)
# 注:并行分支(子图)间共享的通道都必须带 reducer,否则并行扇入会冲突。
class AgentState(TypedDict, total=False):
    # 会话标识(身份型,并行分支值相同 → first)
    thread_id: Annotated[str, _first_non_empty]
    user_id: Annotated[str, _first_non_empty]
    account: Annotated[str, _first_non_empty]
    employee_id: Annotated[str, _first_non_empty]
    tenant_id: Annotated[str, _first_non_empty]
    reply_id: Annotated[str, _first_non_empty]

    # 用户输入
    user_input: Annotated[str, _first_non_empty]

    # Supervisor 路由
    intent: Annotated[str, _first_non_empty]
    routing_plan: Annotated[Optional[Dict], _first_non_empty]  # {"mode": "fast"|"parallel", "intents": [...]}
    next_agent: Annotated[Optional[str], _first_non_empty]

    # RAG 检索
    rag_results: Annotated[Optional[List[Dict]], _first_non_empty]
    rag_hit_rate: Annotated[float, _first_non_empty]
    citations: Annotated[Optional[List[Dict]], _first_non_empty]  # 引用依据(来源/分数/片段) — 证据治理层
    cost_summary: Annotated[Optional[Dict], _first_non_empty]     # 请求级成本摘要 — 证据治理层
    hook_events: Annotated[Optional[List[Dict]], _concat_lists]   # Hooks 治理事件 — 能力执行层
    runtime_context_view: Annotated[Optional[Dict], _first_non_empty]  # Runtime Context 双通道
    context_compression: Annotated[Optional[Dict], _first_non_empty]  # 上下文压缩报告
    tool_calls: Annotated[Optional[List[Dict]], _concat_lists]    # Tool Calling 记录

    # Agent 输出
    agent_output: Annotated[Optional[str], _last_value]
    confidence: Annotated[float, _last_value]

    # FactCheck（三级置信度）
    fact_check_passed: Annotated[bool, _last_value]
    fact_check_errors: Annotated[Optional[List], _concat_lists]
    fact_check_feedback: Annotated[Optional[str], _last_value]
    confidence_level: Annotated[Optional[str], _last_value]   # high / medium / low

    # HITL
    need_human_confirm: Annotated[bool, _last_value]
    confirm_payload: Annotated[Optional[Dict], _first_non_empty]  # interrupt 挂起前的确认信息
    human_intervened: Annotated[bool, _last_value]
    human_action: Annotated[Optional[str], _last_value]
    human_approved: Annotated[bool, _last_value]              # 人工确认后放行标记
    approved_params: Annotated[Optional[Dict], _first_non_empty]  # modify 后人工修改的参数
    approval_mode: Annotated[Optional[str], _first_non_empty]     # inline / external
    approval_id: Annotated[Optional[str], _first_non_empty]       # 外部审批单号

    # 工具澄清
    need_clarification: Annotated[bool, _last_value]
    clarification: Annotated[Optional[Dict], _first_non_empty]

    # 缓存
    cache_hit: Annotated[bool, _last_value]

    # 迭代控制
    iteration_count: Annotated[int, _max_value]
    max_iterations: Annotated[int, _first_non_empty]
    retrieve_count: Annotated[int, _max_value]   # 检索轮次(Agentic RAG 防循环硬守卫)

    # 图超时保护
    start_time: Annotated[float, _first_non_empty]
    timeout_seconds: Annotated[int, _first_non_empty]

    # 时间统计
    duration_ms: Annotated[int, _last_value]

    # 对话历史
    messages: Annotated[List[Dict], operator.add]
    context_summary: Annotated[Optional[str], _first_non_empty]
    recent_rounds: Annotated[List[Dict], _first_non_empty]

    # 并行子任务结果
    sub_results: Annotated[List[Dict], operator.add]

    # 安全
    security_events: Annotated[List[str], operator.add]
    safety_decision: Annotated[Optional[Dict], _first_non_empty]  # Prompt 注入扫描决策

    # 错误
    error: Annotated[Optional[str], _last_value]


# 创建带默认值的初始 AgentState
def create_initial_state(thread_id="", user_id="", account="anonymous", employee_id="",
                         tenant_id="default", user_input="") -> AgentState:
    return AgentState(
        thread_id=thread_id, user_id=user_id, account=account, employee_id=employee_id,
        tenant_id=tenant_id, reply_id="", user_input=user_input,
        intent="", routing_plan=None, next_agent=None,
        rag_results=None, rag_hit_rate=0.0,
        citations=None, cost_summary=None,
        agent_output=None, confidence=0.0,
        fact_check_passed=True, fact_check_errors=None, fact_check_feedback=None,
        confidence_level="high",
        need_human_confirm=False, confirm_payload=None,
        human_intervened=False, human_action=None, human_approved=False,
        approved_params=None, approval_mode=None, approval_id=None,
        need_clarification=False, clarification=None,
        cache_hit=False, iteration_count=0, max_iterations=3,
        retrieve_count=0,
        start_time=0.0, timeout_seconds=180, duration_ms=0,
        messages=[], context_summary=None, recent_rounds=[],
        sub_results=[], security_events=[], error=None,
    )
