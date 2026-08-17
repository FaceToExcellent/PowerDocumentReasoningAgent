"""领域 Agent 子图 — 每个业务领域一个独立子图(独立重试/校验/人工复核)

P2:把"单 agent + supervisor 路由"升级为"项目内多 agent":
  父图(cache/supervisor/aggregate/memory/log)只做编排;
  每个领域意图一个独立编译子图,内部: 检索→执行(领域prompt,带重试)→校验(可重试)→(高危)人工复核。
  子图共享父级 AgentState,节点函数由 graph.py 注入,避免循环导入。

区别于旧结构的关键:
  - 旧:同一个 agent_execute 节点按 intent 换 prompt(单 agent)
  - 新:每个领域是独立 agent 对象(独立 state/重试/降级/复核)
"""
import time
from typing import Any, Callable

from langgraph.graph import END, StateGraph
from langgraph.types import RetryPolicy

from config.settings import settings
from agent.state import AgentState


# 子图内:执行后,高危 → hitl/external,否则 → fact_check
def _route_after_execute(state: AgentState) -> str:
    if state.get("need_human_confirm"):
        risk = (state.get("confirm_payload") or {}).get("risk_level", "high")
        if settings.approval_external_enabled and settings.approval_external_endpoint \
                and risk == "critical":
            return "external_approval"
        return "hitl"
    return "fact_check"


# 子图内:校验后,超时/通过/超迭代 → 显式 done 终止,否则回执行重试
# (注意:条件边直接返回 END 在 langgraph 1.2.11 子图里会抛 KeyError('__end__'),
#  所以统一路由到显式 done 节点,再由 add_edge(done, END) 正常结束)
def _route_after_fact_check(state: AgentState) -> str:
    if time.time() - state.get("start_time", time.time()) > settings.graph_timeout_seconds:
        return "done"
    if state.get("fact_check_passed"):
        return "done"
    if state.get("iteration_count", 0) < settings.max_iterations:
        return "execute"
    return "done"


# 子图内:人工确认后,approve/modify 回执行,否则显式 done 终止
def _route_after_hitl(state: AgentState) -> str:
    return "execute" if state.get("human_action") in ("approve", "modify") else "done"


def build_domain_agent(
    intent: str,
    *,
    retrieve: Callable,
    execute: Callable,
    fact_check: Callable,
    hitl: Callable,
    external_approval: Callable,
    retry_policy: RetryPolicy,
) -> Any:
    """构建一个领域 Agent 的独立编译子图(共享 AgentState)。"""
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("execute", execute, retry_policy=retry_policy)
    workflow.add_node("fact_check", fact_check)
    workflow.add_node("hitl", hitl)
    workflow.add_node("external_approval", external_approval)
    workflow.add_node("done", lambda state: {})   # 显式终止节点(规避条件边返回 END 的子图 bug)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "execute")
    workflow.add_conditional_edges("execute", _route_after_execute,
                                   {"fact_check": "fact_check", "hitl": "hitl",
                                    "external_approval": "external_approval"})
    workflow.add_conditional_edges("fact_check", _route_after_fact_check,
                                   {"execute": "execute", "done": "done"})
    workflow.add_conditional_edges("hitl", _route_after_hitl,
                                   {"execute": "execute", "done": "done"})
    workflow.add_conditional_edges("external_approval", _route_after_hitl,
                                   {"execute": "execute", "done": "done"})
    workflow.add_edge("done", END)
    return workflow.compile(name=f"agent_{intent}")
