"""AgentState — LangGraph 共享状态（企业版：tenant_id / reply_id / routing）"""
from typing import TypedDict, Optional, List, Dict, Any, Annotated
import operator


# AgentState — LangGraph 全局共享状态字段定义(TypedDict)
class AgentState(TypedDict, total=False):
    # 会话标识
    thread_id: str
    user_id: str
    account: str
    employee_id: str
    tenant_id: str
    reply_id: str

    # 用户输入
    user_input: str

    # Supervisor 路由
    intent: str
    routing_plan: Optional[Dict]     # {"mode": "fast"|"parallel", "intents": [...]}
    next_agent: Optional[str]

    # RAG 检索
    rag_results: Optional[List[Dict]]
    rag_hit_rate: float
    citations: Optional[List[Dict]]       # 引用依据(来源/分数/片段) — 证据治理层
    cost_summary: Optional[Dict]          # 请求级成本摘要 — 证据治理层
    hook_events: Optional[List[Dict]]     # Hooks 治理事件 — 能力执行层
    runtime_context_view: Optional[Dict]  # Runtime Context 双通道(模型可见/系统校验)
    context_compression: Optional[Dict]   # 上下文压缩报告(protected/折叠)
    tool_calls: Optional[List[Dict]]      # Tool Calling 记录(名称/参数/观察)

    # Agent 输出
    agent_output: Optional[str]
    confidence: float

    # FactCheck（三级置信度）
    fact_check_passed: bool
    fact_check_errors: Optional[List]
    fact_check_feedback: Optional[str]
    confidence_level: Optional[str]   # high / medium / low

    # HITL
    need_human_confirm: bool
    confirm_payload: Optional[Dict]    # interrupt 挂起前保存的确认信息（推给前端）
    human_intervened: bool
    human_action: Optional[str]
    human_approved: bool               # 人工确认后放行标记，避免二次拦截
    approved_params: Optional[Dict]    # modify 后人工修改的参数
    approval_mode: Optional[str]       # inline / external
    approval_id: Optional[str]         # 外部审批单号

    # 工具澄清
    need_clarification: bool
    clarification: Optional[Dict]

    # 缓存
    cache_hit: bool

    # 迭代控制
    iteration_count: int
    max_iterations: int

    # 图超时保护
    start_time: float
    timeout_seconds: int

    # 时间统计
    duration_ms: int

    # 对话历史
    messages: Annotated[List[Dict], operator.add]
    context_summary: Optional[str]
    recent_rounds: List[Dict]

    # 并行子任务结果
    sub_results: Annotated[List[Dict], operator.add]

    # 安全
    security_events: Annotated[List[str], operator.add]
    safety_decision: Optional[Dict]       # Prompt 注入扫描决策

    # 错误
    error: Optional[str]


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
        start_time=0.0, timeout_seconds=180, duration_ms=0,
        messages=[], context_summary=None, recent_rounds=[],
        sub_results=[], security_events=[], error=None,
    )
