"""Planner 模式 — plan-then-execute:解耦任务 → 调度执行 → 汇总

可靠性原则(编排确定性,LLM 只在"提议"和"子任务内容"):
- 解耦:LLM 提议步骤,强制校验(意图白名单 + 步数上限 + JSON 解析)
- 调度:代码按依赖拓扑执行,无依赖步骤并行
- 执行:每步复用 run_agent(领域子图 + HITL + 引用 + 审计 + 重试)
- 汇总:LLM 整合已知步骤结果

防失控:MAX_PLAN_STEPS 硬上限、解耦失败降级单步、依赖环检测终止。
"""
import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List

logger = logging.getLogger(__name__)

MAX_PLAN_STEPS = 5


@dataclass
class PlanStep:
    """计划中的一步:子任务 + 路由意图 + 依赖 + 执行结果。"""

    id: str
    task: str
    intent: str
    depends_on: List[str] = field(default_factory=list)
    status: str = "pending"        # pending/running/done/error
    result: str = ""
    citations: List[Dict] = field(default_factory=list)


def _valid_intents() -> List[str]:
    """当前领域的合法意图白名单(解耦校验用)。"""
    from config.domain import get_domain
    from config.settings import settings
    domain = get_domain(settings.domain)
    return list(domain.get_intents()) + ["chat"]


async def decompose_task(user_input: str) -> List[PlanStep]:
    """LLM 提议步骤 → 校验(白名单/步数/JSON)→ PlanStep 列表;失败降级单步。"""
    from llm.adapter import unified_llm
    valid = _valid_intents()
    try:
        r = await unified_llm.ainvoke("chat", [
            {"role": "system", "content": (
                "你是任务规划器。把用户任务拆成 1-5 个可执行子步骤。"
                f"每步 intent 只能从:{', '.join(valid)} 里选。"
                '只输出 JSON:{"steps":[{"id":"s1","task":"子任务","intent":"...","depends_on":[]}]}'
                "无依赖的步骤可并行;依赖用 depends_on 填前面步骤的 id。")},
            {"role": "user", "content": user_input},
        ])
        data = json.loads(r.content.strip().strip("`").strip())
        raw = data.get("steps", [])
        valid_ids = {str(s.get("id", f"s{i+1}")) for i, s in enumerate(raw)}
        steps: List[PlanStep] = []
        for i, s in enumerate(raw[:MAX_PLAN_STEPS]):
            intent = str(s.get("intent", "chat"))
            if intent not in valid:
                intent = "chat"
            deps = [str(d) for d in (s.get("depends_on") or []) if str(d) in valid_ids]
            steps.append(PlanStep(id=str(s.get("id", f"s{i+1}")),
                                  task=str(s.get("task", user_input))[:500],
                                  intent=intent, depends_on=deps))
        if steps:
            return steps
    except Exception as e:
        logger.warning(f"planner 解耦失败,降级单步: {e}")
    return [PlanStep(id="s1", task=user_input, intent="chat")]


async def execute_plan(steps: List[PlanStep], *, tenant_id: str = "",
                       thread_id: str = "") -> List[PlanStep]:
    """按依赖拓扑调度:无依赖并行,有依赖串行;每步经 run_agent 执行。"""
    from agent.graph import run_agent
    pool = {s.id: s for s in steps}
    done: set = set()
    while pool:
        ready = [s for s in pool.values() if all(d in done for d in s.depends_on)]
        if not ready:
            logger.warning("planner 依赖环,中止剩余步骤")
            for s in pool.values():
                s.status = "error"
                s.result = "依赖环,未执行"
            break

        async def _run(s):
            s.status = "running"
            r = await run_agent(s.task, thread_id=f"{thread_id}-{s.id}",
                                tenant_id=tenant_id, account="planner")
            s.status = "done" if r.get("success") else "error"
            s.result = (r.get("reply") or r.get("error") or "")[:1500]
            s.citations = r.get("citations") or []

        await asyncio.gather(*[_run(s) for s in ready])
        for s in ready:
            done.add(s.id)
            pool.pop(s.id)
    return steps


async def aggregate_plan(steps: List[PlanStep], user_input: str) -> str:
    """LLM 整合各步骤结果 → 最终结构化答案。"""
    from llm.adapter import unified_llm
    parts = []
    for s in steps:
        mark = "✓" if s.status == "done" else "✗"
        parts.append(f"步骤{s.id}[{s.intent}]{mark}:\n{s.result or '(无结果)'}")
    combined = "\n\n".join(parts)
    r = await unified_llm.ainvoke("chat", [
        {"role": "system", "content": "你是汇总员。把多步骤结果整合成一份完整结构化的最终答案,按步骤组织;失败的步骤要明确说明。"},
        {"role": "user", "content": f"原任务:{user_input}\n\n步骤结果:\n{combined}"},
    ])
    return r.content.strip()
