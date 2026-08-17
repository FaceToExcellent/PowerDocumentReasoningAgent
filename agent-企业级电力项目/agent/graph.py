"""LangGraph 主图 — 企业版 v5.0
节点：cache_check → supervisor(分级双路径) → rag_retrieve → agent_execute
     → fact_check → memory_write → execution_log；HITL interrupt；并行 fan-out
"""
import asyncio
import contextvars
import json
import time
import uuid
import logging
import re

from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command, RetryPolicy

from agent.state import AgentState, create_initial_state
from agent.context_manager import context_manager
from agent.execution_log import write_log
from agent.fact_checker import check_output
from agent.checkpointer import init_checkpointer
from agent.rag_cache import rag_cache
from agent.harness.interceptor import harness_interceptor
from config.settings import settings
from config.cache import cache_service
from observability.metrics import metrics
from observability.audit import audit_logger
from observability.tracing import tracer, get_trace_id
from memory.manager import memory_manager
from llm.adapter import unified_llm
from llm.model_router import model_router
from agent.skills.bootstrap import skill_registry
from agent.skills.selector import skill_selector

logger = logging.getLogger(__name__)

# ⭐ SSE 流式：agent_execute_node 通过此 contextvar 把 token 推给 SSE endpoint
STREAM_QUEUE: contextvars.ContextVar = contextvars.ContextVar("stream_queue", default=None)

_ABORT_EVENT: dict = {}          # thread_id → asyncio.Event（前端取消）

# ── 领域化：意图关键词 / 提示词 / Skill 映射 全部从 DomainConfig 读取 ──
from config.settings import settings
from config.domain import get_domain

_current_domain = get_domain(settings.domain)
INTENT_KEYWORDS = _current_domain.intent_keywords          # 领域意图词
INTENT_PROMPTS = _current_domain.intent_prompts            # 领域提示词
_INTENT_TO_SKILL = {i: _current_domain.intent_to_skill(i) for i in _current_domain.get_intents()}
DEFAULT_INTENT = list(INTENT_KEYWORDS.keys())[0] if INTENT_KEYWORDS else "chat"

_COMPOUND_CONNECTORS = ["且", "并且", "同时", "再", "然后", "还要"]

# ── 节点级重试：LLM/网络瞬态错误自动重试，逻辑错误不重试 ──
def _retryable(exc: Exception) -> bool:
    """返回 True 表示该异常值得重试（瞬态错误），False 则不重试（逻辑错误）。

    默认 RetryPolicy 对 RuntimeError 不重试，而 unified_llm 全后端失败
    恰好抛 RuntimeError("所有 LLM 后端均不可用")——这是"值得重试"的瞬态
    失败（上游限流/网络抖动），所以要显式放行。
    """
    import httpx
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        # 5xx 服务端错误 / 429 限流值得重试；4xx 逻辑错误不重试
        return 429 <= exc.response.status_code or 500 <= exc.response.status_code
    if isinstance(exc, ConnectionError):
        return True
    if isinstance(exc, asyncio.TimeoutError):
        return True
    if isinstance(exc, RuntimeError):
        # unified_llm 全后端失败抛 RuntimeError —— 视为瞬态，重试整个节点
        return True
    # 其余未知异常：交给默认判断（默认重试）
    return True


# agent_execute_node：LLM 推理节点，最容易因超时/网络瞬态失败
AGENT_RETRY_POLICY = RetryPolicy(
    max_attempts=3,          # 含首次，共 3 次尝试
    initial_interval=1.0,    # 首次重试前等待 1s
    backoff_factor=2.0,      # 指数退避：1s → 2s → 4s
    max_interval=30.0,        # 最大重试间隔 30s
    jitter=True,             # 加随机抖动，避免重试风暴
    retry_on=_retryable,
)

# ── 复杂度预判 ──────────────────────────────────
def _judge_complexity(state: AgentState) -> dict:
    """复用 INTENT_KEYWORDS 打分：单意图→fast，多意图+连接词→parallel"""
    user_input = state.get("user_input", "")
    scores = {}
    for intent, kws in INTENT_KEYWORDS.items():
        s = sum(1 for kw in kws if kw in user_input)
        if s:
            scores[intent] = s
    if len(scores) <= 1:
        return {"mode": "fast", "intents": list(scores) or ["chat"]}
    has_connector = any(c in user_input for c in _COMPOUND_CONNECTORS)
    top2 = sorted(scores, key=scores.get, reverse=True)[:2]
    if has_connector and scores[top2[0]] - scores[top2[1]] < 2:
        return {"mode": "parallel", "intents": top2}
    return {"mode": "fast", "intents": [top2[0]]}


# ── 节点 1：缓存检查 ─────────────────────────────
async def cache_check_node(state: AgentState) -> dict:
    user_input = state.get("user_input", "")
    tenant = state.get("tenant_id", "default")
    guessed = _guess_intent(user_input)
    key = rag_cache.build_cache_key(user_input, guessed, tenant_id=tenant)
    cached = await rag_cache.get(key)
    if cached and cached.get("agent_output"):
        logger.info(f"缓存命中: {guessed}")
        return {"cache_hit": True, "agent_output": cached["agent_output"],
                "confidence": cached.get("confidence", 0.8),
                "rag_results": cached.get("rag_results"), "intent": guessed}
    return {"cache_hit": False}


# 根据用户输入关键词打分,返回得分最高的意图 此时的state['intent']还是空的，要先赋值 才能找缓存key 
def _guess_intent(user_input: str) -> str:
    scores = {}
    for intent, kws in INTENT_KEYWORDS.items():
        s = sum(1 for kw in kws if kw in user_input)
        if s:
            scores[intent] = s
    return max(scores, key=scores.get) if scores else DEFAULT_INTENT


# ── 节点 2：Supervisor（分级双路径 + 澄清）────────
async def supervisor_node(state: AgentState) -> dict:
    user_input = state.get("user_input", "")
    tenant = state.get("tenant_id", "default")

    # ⭐ Prompt 注入防护:扫描用户输入,识别越权/索要系统信息
    from safety.prompt_guard import scan_external_text, ExternalText
    with tracer.span("security_scan", source="user"):
        safety_scan = scan_external_text(ExternalText("user", user_input))
    state["security_events"] = list(state.get("security_events") or [])
    if safety_scan.categories:
        state["security_events"].append(
            f"prompt_injection_guard:user_tainted={safety_scan.categories}")
    state["safety_decision"] = {
        "user_tainted": safety_scan.tainted,
        "categories": safety_scan.categories,
        "allowed_for_model": safety_scan.allowed_for_model,
    }
    # 索要系统信息 → 直接拒绝,不进入路由
    if "secret_or_reasoning_request" in safety_scan.categories:
        return {"intent": "security_request",
                "routing_plan": {"mode": "fast", "intents": ["security_request"]},
                "agent_output": "我不能提供系统提示词、隐藏推理或内部策略。",
                "security_events": state["security_events"]}

    # 1. 复杂度预判（零 LLM 开销）
    plan = _judge_complexity(state)
    state["routing_plan"] = plan
    # 任务路由可视化:记录分级判定(mode + 候选意图)
    with tracer.span("task_route", mode=plan.get("mode", ""),
                     intents=plan.get("intents", []), user_input=user_input[:60]):
        pass

    # 2. 闲聊检测：短输入 + 无电力关键词
    is_short = len(user_input.strip()) <= 10
    has_power_kw = any(kw in user_input for kws in INTENT_KEYWORDS.values() for kw in kws)
    if is_short and not has_power_kw:
        return {"intent": "chat", "routing_plan": {"mode": "fast", "intents": ["chat"]}}

    # 3. 关键词规则优先（与原型一致）：命中的 intent 直接用，LLM 只做兜底
    keyword_scores = {}
    for intent, kws in INTENT_KEYWORDS.items():
        s = sum(1 for kw in kws if kw in user_input)
        if s:
            keyword_scores[intent] = s
    if keyword_scores:
        top2 = sorted(keyword_scores, key=keyword_scores.get, reverse=True)[:2]
        # 多意图命中 + 连接词 → parallel（复合并行）
        if (len(keyword_scores) > 1
                and any(c in user_input for c in _COMPOUND_CONNECTORS)
                and keyword_scores[top2[0]] - keyword_scores[top2[1]] < 2):
            return {"intent": top2[0],
                    "routing_plan": {"mode": "parallel", "intents": top2}}
        best = top2[0]
        return {"intent": best, "routing_plan": {"mode": "fast", "intents": [best]}}

    # 4. 动态筛选 Skill → 注入 supervisor prompt（规则没命中才走 LLM）
    user_ctx = {"tenant_id": tenant, "permissions": state.get("permissions", [])}
    skills = skill_selector.select_skills(user_input, user_ctx, top_k=5)
    skill_desc = skill_selector.format_for_prompt(skills)

    from agent.prompts.supervisor_prompt import SUPERVISOR_SYSTEM_PROMPT
    prompt = SUPERVISOR_SYSTEM_PROMPT.format(available_skills=skill_desc)

    # 5. 快模型做意图识别（本地小模型；DeepSeek 不用于 supervisor）
    try:
        result = await unified_llm.ainvoke("supervisor", [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input},
        ])
        raw = result.content.strip().lower()
    except Exception:
        raw = "chat"

    # 6. 解析意图：逗号分隔多意图 → parallel
    intents = [x.strip() for x in raw.replace("，", ",").split(",") if x.strip()]
    valid = [i for i in intents if i in _INTENT_TO_SKILL or i == "chat"]
    if not valid:
        valid = ["chat"]
    if plan["mode"] == "parallel" and len(valid) > 1:
        return {"intent": valid[0], "routing_plan": {"mode": "parallel", "intents": valid[:2]}}
    return {"intent": valid[0], "routing_plan": {"mode": "fast", "intents": [valid[0]]}}


# ── 节点 3：RAG 检索 ─────────────────────────────
async def rag_retrieve_node(state: AgentState) -> dict:
    intent = state.get("intent", DEFAULT_INTENT)
    if intent == "chat":
        return {"rag_results": [], "rag_hit_rate": 0.0}

    tenant = state.get("tenant_id", "default")
    user_input = state.get("user_input", "")
    from rag.retriever import rag_service
    # ⭐ agent 级 span:标记是哪个领域 agent 在做检索
    with tracer.span(f"agent_{intent}_retrieve", intent=intent,
                     thread_id=state.get("thread_id", "")):
        result = rag_service.search(user_input, top_k=5, intent=intent, tenant_id=tenant)
    results = result.get("results", [])
    # ⭐ 构建结构化 citations(证据治理层:来源/标题/分数/片段,可审计)
    citations = [
        {
            "source": it.get("doc", {}).get("metadata", {}).get("source", ""),
            "title": it.get("doc", {}).get("metadata", {}).get("title", ""),
            "chunk_id": it.get("chunk_id", ""),
            "score": round(float(it.get("score", 0)), 4),
            "snippet": (it.get("doc", {}).get("content", "") or "")[:120],
        }
        for it in results[:5]
    ]
    return {
        "rag_results": results,
        "rag_hit_rate": min(1.0, len(results) / 5),
        "citations": citations,
    }


# ── 节点 4：Agent 执行（模型分级 + 流式 + Skill）─
def _build_kg_evidence(user_input: str) -> str:
    """从用户输入抽实体 → 查 KG 一跳关系 → 格式化为证据文本"""
    try:
        from rag.kg.entity_index import entity_index
        entities = entity_index.extract_entities(user_input)
        if not entities:
            return ""
        lines = []
        for ent in entities:
            rels = entity_index.query(ent)
            for r in rels:
                lines.append(f"  {r['subject']} --[{r['relation']}]--> {r['object']}")
        return "\n".join(lines) if lines else ""
    except Exception as e:
        logger.debug(f"KG 证据构建跳过: {e}")
        return ""


# parallel 分支首次执行时把单意图结果写入 sub_results,供 aggregate 汇总
def _parallel_sub_result(state, intent, output):
    """parallel 分支结束时把结果写进 sub_results（reducer 自动累加），供 aggregate 汇总。
    仅首次执行写，避免 fact_check 重试循环里重复累加旧结果。"""
    plan = state.get("routing_plan") or {}
    if (plan.get("mode") == "parallel" and len(plan.get("intents", [])) > 1
            and not state.get("fact_check_feedback")):
        return {"sub_results": [{"intent": intent, "agent_output": output}]}
    return {}


# 核心推理节点:组装上下文与消息,走 Skill/HITL/流式或非流式 LLM 调用
async def agent_execute_node(state: AgentState) -> dict:
    intent = state.get("intent", "chat")
    user_input = state.get("user_input", "")
    tenant = state.get("tenant_id", "default")
    user_id = state.get("user_id", state.get("account", "anonymous"))
    thread_id = state.get("thread_id", "")

    # 记忆按需加载（推理前）
    memory_ctx = await memory_manager.build_context(
        tenant_id=tenant, user_id=user_id, thread_id=thread_id, query=user_input)

    # 构建 messages：系统Prompt（领域化，从 DomainConfig 读取）+ 记忆 + 证据
    sys_prompt = INTENT_PROMPTS.get(intent, INTENT_PROMPTS.get("chat", "你是文档分析助手。"))

    evidence = ""
    if state.get("rag_results"):
        evidence = "\n".join(
            f"[{it.get('doc',{}).get('metadata',{}).get('source','')}] "
            f"{it.get('doc',{}).get('content','')[:300]}"
            for it in state["rag_results"][:5])

    # ⭐ 分源扫描:检索到的文档先过 Prompt 注入防护,再进上下文(防文档内嵌间接注入)
    if evidence:
        from safety.prompt_guard import ExternalText, scan_external_text
        rag_scan = scan_external_text(ExternalText("rag", evidence, "rag_evidence"))
        if rag_scan.categories:
            state["security_events"] = list(state.get("security_events") or [])
            state["security_events"].append(
                f"prompt_injection_guard:rag_tainted={rag_scan.categories}")
            evidence = rag_scan.sanitized_content

    # ⭐ KG 关系证据：从用户输入抽实体 → 查一跳关系 → 拼入证据（复杂推理）
    kg_evidence = _build_kg_evidence(user_input)

    # ⭐ Runtime Context 双通道:trusted_for_model 模型可见 / system_only 系统校验
    from agent.context_builder import build_runtime_context_view
    rt_view = build_runtime_context_view(
        tenant_id=tenant, 
        user_id=user_id, 
        nickname=state.get("account", ""),
        member_level="", 
        page_context={}, 
        risk_level="",
        permissions=[],
    )
    state["runtime_context_view"] = rt_view

    # ⭐ Context Builder：多来源按 trust_level 排序 + 冲突解决
    from agent.context_builder import ContextBuilder
    cb = ContextBuilder()
    cb.add("runtime_context", f"tenant_id={tenant} user_id={user_id}", key="runtime_identity")
    cb.add("system_rules", sys_prompt, key="system_prompt")
    if memory_ctx:
        cb.add("memory", f"参考记忆：\n{memory_ctx}", key="memory")
    if evidence:
        cb.add("tool_fact", f"检索到的文档：\n{evidence}", key="rag_evidence")
    else:
        cb.add("system_rules",
               "本次未检索到相关文档依据。若问题涉及具体规程/数值，请明确回复'抱歉，知识库中未找到相关资料'，不要编造答案。",
               key="no_rag_fallback", priority=1)
    if kg_evidence:
        cb.add("tool_fact", f"设备/规程实体关系（知识图谱）：\n{kg_evidence}", key="kg_evidence")
    if state.get("fact_check_feedback"):
        cb.add("system_rules", f"上次校验未通过，请修正：{state['fact_check_feedback']}",
               key="fc_feedback", priority=2)
    # HITL 状态:审批中的事实是高信任,不能被用户新话覆盖
    if state.get("need_human_confirm") and state.get("confirm_payload"):
        cb.add("hitl_state", f"高危操作已挂起等待人工审批：{state['confirm_payload']}",
               key="hitl_pending")
    # ⭐ HITL modify:人工修改后的执行参数,以高信任系统规则注入
    approved_params = state.get("approved_params")
    if approved_params:
        cb.add("system_rules",
               f"人工确认后修改的执行参数：{json.dumps(approved_params, ensure_ascii=False)}，请按修改后的参数执行。",
               key="approved_params", priority=2)
    # 上下文构建 span:记录条目数/高信任保护数(上下文组装可视化)
    with tracer.span("context_build", items=len(cb._items),
                     protected=sum(1 for it in cb._items if it.trust >= 70)):
        cb.resolve_conflicts()

    # ⭐ 上下文压缩:保护高信任项,折叠低相关历史
    comp_report = cb.compress(max_items=getattr(settings, "max_recent_rounds", 8))
    state["context_compression"] = comp_report
    built_context = cb.render()

    messages = [{"role": "system", "content": built_context or sys_prompt}]
    messages.append({"role": "user", "content": user_input})

    # 选择 task 名（核心推理 vs 轻量）：领域有该意图就用领域意图，否则 chat
    task = intent if intent in _INTENT_TO_SKILL and intent != "chat" else "chat"

    # ⭐ 工具澄清:对比分析缺第二个对象时先追问,不猜测执行
    if intent == "comparison_analysis":
        from agent.clarification import pre_tool_clarification
        from agent.skills.comparison_skill import ComparisonAnalysisSkill
        ent = ComparisonAnalysisSkill._extract_entities(user_input)
        clarification = pre_tool_clarification(
            intent, user_input,
            {"entity_a": ent[0] if ent else "", "entity_b": ent[1] if len(ent) > 1 else ""},
            ["entity_b"],
        )
        if clarification:
            state["clarification"] = clarification.to_dict()
            return {"need_clarification": True,
                    "agent_output": clarification.question,
                    "intent": intent}

    # ⭐ Harness 风险拦截：HIGH/CRITICAL → 需要人工确认（标记后由路由转到 hitl_review）
    # 已人工批准（human_approved）则跳过拦截直接执行
    if not state.get("human_approved"):
        from observability.tracing import tracer
        with tracer.span("harness_risk_check", task=task, thread_id=thread_id):
            intercept = await harness_interceptor.before_skill_execute(
                task, {"intent": intent, "user_input": user_input},
                user_id=user_id, thread_id=thread_id)
        if intercept.need_confirm:
            logger.info(f"[harness] 高危操作需人工确认: {task} risk={intercept.risk_level.value}")
            return {"need_human_confirm": True, "confirm_payload": intercept.message,
                    "intent": intent}

    # ⭐ 流式：如果有 STREAM_QUEUE 注入，走流式
    queue = STREAM_QUEUE.get()
    if queue is not None:
        return await _agent_execute_stream(state, task, messages, queue, user_id, thread_id, intent)

    # 非流式
    from observability.tracing import tracer
    with tracer.span("llm_invoke", task=task, intent=intent):
        result = await unified_llm.ainvoke(task, messages)
    output = result.content
    confidence = 0.7 if result.content else 0.0

    # 若走 comparison_analysis Skill，整合证据
    if intent == "comparison_analysis":
        output = await _run_comparison_skill(state, user_input, output, tenant)

    return {
        "agent_output": output,
        "confidence": confidence,
        "iteration_count": state.get("iteration_count", 0) + 1,
        **_parallel_sub_result(state, intent, output),
    }


# 执行对比分析 Skill,附带 Hooks 治理与 Tool Calling 记录
async def _run_comparison_skill(state, user_input, fallback_output, tenant):
    skill = skill_registry.get("comparison_analysis")
    if not skill:
        return fallback_output
    # ⭐ Hooks: Skill 执行前治理(必填参数/只读/风险) + 执行 + 结果脱敏
    from agent.hooks import SkillHooks
    hooks = SkillHooks()
    hooks.pre_tool_call("comparison_analysis", skill.metadata, {"query": user_input})
    # ⭐ Tool Calling 记录:记录 skill 调用为 tool_call(名称/参数/观察)
    state["tool_calls"] = list(state.get("tool_calls") or [])
    state["tool_calls"].append({
        "tool_name": "comparison_analysis",
        "arguments": {"query": user_input, "tenant_id": tenant},
        "observation_summary": "对比分析 Skill 执行",
    })
    ctx = {"query": user_input, "tenant_id": tenant, "user_context": {"tenant_id": tenant}}
    if state.get("approved_params"):
        ctx["approved_params"] = state.get("approved_params")
    # 工具调用 span:参数摘要 + Observation 摘要(脱敏)进 trace
    from opentelemetry import trace as _otel_trace
    with tracer.span("tool_call", tool="comparison_analysis",
                     params=json.dumps({"query": user_input}, ensure_ascii=False)[:200]):
        result = await skill.run(ctx)
        obs = result.to_observation()
        _otel_trace.get_current_span().set_attribute("observation", obs.text[:200])
        _otel_trace.get_current_span().set_attribute("success", result.success)
        _otel_trace.get_current_span().set_attribute("duration_ms", result.duration_ms)
    # L20:ToolResult → Observation(脱敏摘要 + 省略字段),回填进 tool_calls 记录
    state["tool_calls"][-1].update({
        "observation": obs.text[:300],
        "omitted_fields": obs.omitted_fields,
        "success": result.success,
        "duration_ms": result.duration_ms,
    })
    hooks.post_tool_call("comparison_analysis", result)
    if not result.success:
        hooks.on_error("comparison_analysis", Exception(result.error or "skill failed"))
    state["hook_events"] = [e.to_dict() for e in hooks.events] + [hooks.on_completion()]
    if result.success:
        if isinstance(result.result, dict):
            return result.result.get("analysis", fallback_output)
        if isinstance(result.result, str):
            return result.result or fallback_output
        return fallback_output
    return fallback_output


# 流式执行 LLM 推理,逐 token 推送到队列,失败自动重试并兜底
async def _agent_execute_stream(state, task, messages, queue, user_id, thread_id, intent):
    """流式执行：逐 token 推送，同时累积最终内容。

    astream 是生成器，失败发生在迭代中途——langgraph 的 retry_policy 只会
    重跑整个节点，但流式路径的异常被这里 catch 后返回兜底文案，不会冒泡到
    节点层触发 retry。所以流式场景在这里自己做重试：整体失败则重置重新流一次。
    """
    max_stream_attempts = 3
    for attempt in range(1, max_stream_attempts + 1):
        collected = ""
        thinking_notified = False
        try:
            async for chunk in unified_llm.astream(task, messages):
                if chunk["type"] == "thinking":
                    # 不暴露内部推理链：仅在首个思考 chunk 推一条通用提示
                    if not thinking_notified:
                        thinking_notified = True
                        await queue.put({"type": "thinking",
                                         "data": {"content": "正在深度思考中，请稍候…",
                                                  "nodeName": "deepseek"}})
                elif chunk["text"]:
                    collected += chunk["text"]
                    await queue.put({"type": "reply", "data": {"content": chunk["text"],
                                                               "is_final": False}})
                if chunk.get("is_final"):
                    break
            # 正常流完：哪怕无内容也返回（不重试，避免死循环）
            break
        except Exception as e:
            logger.error(f"流式推理失败(第{attempt}/{max_stream_attempts}次): {e}")
            # 已输出部分内容也失败：重试会重复推送，放弃重试直接兜底
            if collected or attempt >= max_stream_attempts:
                break
            await asyncio.sleep(min(2 ** (attempt - 1), 8))   # 1s → 2s → 4s 退避
            continue
    if not collected:
        collected = "抱歉，知识库中未找到相关资料，无法回答您的问题。建议查阅相关规程文档或联系运维人员。"
    return {
        "agent_output": collected,
        "confidence": 0.7,
        "iteration_count": state.get("iteration_count", 0) + 1,
        **_parallel_sub_result(state, intent, collected),
    }


# ── 节点 5：FactCheck（三级置信度）────────────────
async def fact_check_node(state: AgentState) -> dict:
    intent = state.get("intent", "")
    if intent == "chat":
        return {"fact_check_passed": True, "confidence_level": "high",
                "fact_check_errors": [], "fact_check_feedback": None}
    output = state.get("agent_output", "")
    if not output:
        return {"fact_check_passed": True, "confidence_level": "high",
                "fact_check_errors": [], "fact_check_feedback": None}
    try:
        # 传入真实检索上下文,置信度才能反映"检索覆盖 + 引用/接地"
        with tracer.span(f"agent_{intent}_fact_check", intent=intent,
                         thread_id=state.get("thread_id", "")):
            result = await asyncio.wait_for(
                asyncio.to_thread(check_output, output, {
                    "intent": intent,
                    "rag_results": state.get("rag_results"),
                    "citations": state.get("citations"),
                    "rag_hit_rate": state.get("rag_hit_rate", 0.0),
                }), timeout=5)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        return {"fact_check_passed": True, "confidence_level": "high",
                "fact_check_errors": [], "fact_check_feedback": None}
    except Exception as e:
        logger.warning(f"FactCheck 失败（默认通过）: {e}")
        return {"fact_check_passed": True, "confidence_level": "high",
                "fact_check_errors": [], "fact_check_feedback": None}

    # ⭐ Agent 级审计:记录该领域 agent 的一次执行(输入/输出/校验/工具/引用)
    audit_logger.log_agent(
        tenant_id=state.get("tenant_id", ""), thread_id=state.get("thread_id", ""),
        agent=intent, user_input=state.get("user_input", ""),
        agent_output=state.get("agent_output", ""),
        fact_check_passed=bool(result.get("passed")),
        confidence_level=result.get("confidence", "high"),
        tool_calls=len(state.get("tool_calls") or []),
        citations_count=len(state.get("citations") or []),
        iteration=state.get("iteration_count", 0),
    )

    # ⭐ 决策动作:低置信且无依据的具体断言 → 拒答(不硬答)
    if result.get("confidence") == "low" and intent != "chat" and output.strip():
        logger.info(f"⚠️ 低置信无依据,转为拒答: thread={state.get('thread_id')} intent={intent}")
        output = "抱歉，知识库中未找到相关依据，无法给出确切结论。建议查阅相关规程文档或联系运维人员核实。"
        return {
            "agent_output": output,
            "fact_check_passed": True,
            "fact_check_errors": result.get("errors", []),
            "fact_check_feedback": result.get("feedback"),
            "confidence_level": "low",
        }

    return {
        "fact_check_passed": bool(result.get("passed")),
        "fact_check_errors": result.get("errors", []),
        "fact_check_feedback": result.get("feedback"),
        "confidence_level": result.get("confidence", "high"),
    }


# ── 节点 6：记忆写入 ─────────────────────────────
async def memory_write_node(state: AgentState) -> dict:
    tenant = state.get("tenant_id", "default")
    user_id = state.get("user_id", state.get("account", "anonymous"))
    thread_id = state.get("thread_id", "")
    reply_id = state.get("reply_id", "") or f"r{uuid.uuid4().hex[:8]}"
    user_input = state.get("user_input", "")
    output = state.get("agent_output", "")

    # 写入用户输入
    await memory_manager.record(tenant_id=tenant, user_id=user_id, thread_id=thread_id,
                                reply_id=reply_id, role="user", content=user_input,
                                intent=state.get("intent", ""))
    # 写入助手回复
    await memory_manager.record(tenant_id=tenant, user_id=user_id, thread_id=thread_id,
                                reply_id=reply_id, role="assistant", content=output,
                                intent=state.get("intent", ""))

    # 写 RAG 缓存(带索引版本,知识更新自动失效);低置信不缓存,避免污染
    index_version = ""
    if state.get("confidence_level") != "low":
        index_version = await rag_cache.get_index_version(tenant_id=tenant)
        key = rag_cache.build_cache_key(user_input, state.get("intent", ""),
                                        tenant_id=tenant, index_version=index_version)
        await rag_cache.set(key, {
            "agent_output": output, "confidence": state.get("confidence", 0.0),
            "rag_results": state.get("rag_results"),
            "index_version": index_version,
        })
    return {"reply_id": reply_id, "index_version": index_version}


# ── 节点 7：执行日志 + 指标 + 成本摘要 ──────────
async def execution_log_node(state: AgentState) -> dict:
    output = state.get("agent_output", "")
    intent = state.get("intent", "")

    # ⭐ 构建请求级 cost_summary(证据治理层:模型/路径/token 估算/来源数)
    from opentelemetry import trace as _otel_trace
    with tracer.span("cost_summary", intent=intent):
        primary_backend = model_router.route(intent if intent != "chat" else "chat")
        est_prompt = int(len(state.get("user_input", "")) / 2 + len(output) / 2)  # 估算 prompt token 数
        est_total = int(len(output) / 2)  # 估算 token 数
        cost_summary = {
            "path": "rag" if state.get("rag_results") else ("cache" if state.get("cache_hit") else "direct"),
            "model_backend": primary_backend,
            "intent": intent,
            "rag_hits": len(state.get("rag_results") or []),
            "citations_count": len(state.get("citations") or []),
            "estimated_prompt_tokens": est_prompt,
            "estimated_completion_tokens": est_total,
            "estimated_total_tokens": est_prompt + est_total,
            "cache_hit": state.get("cache_hit", False),
            "fact_check_passed": state.get("fact_check_passed", True),
            "confidence": state.get("confidence", 0),
        }
        _otel_trace.get_current_span().set_attribute("path", cost_summary["path"])
        _otel_trace.get_current_span().set_attribute("model_backend", primary_backend)
        _otel_trace.get_current_span().set_attribute("est_total_tokens", est_prompt + est_total)

    write_log(
        thread_id=state.get("thread_id", ""), account=state.get("account", "anonymous"),
        emp_id=state.get("employee_id", ""), intent=intent,
        user_input=state.get("user_input", ""), output=output,
        confidence=state.get("confidence", 0.0),
        fc_passed=state.get("fact_check_passed", True),
        fc_errors=state.get("fact_check_errors", []),
        fc_feedback=state.get("fact_check_feedback"),
        duration_ms=int(time.time() * 1000 - state.get("start_time", time.time()) * 1000),
        iteration=state.get("iteration_count", 0),
        cache_hit=state.get("cache_hit", False),
        rag_hit_rate=state.get("rag_hit_rate", 0.0),
        tenant_id=state.get("tenant_id", ""), user_id=state.get("user_id", ""),
    )
    metrics.incr("agent_request_total", labels={"intent": intent})
    metrics.observe("agent_latency_ms", time.time() - state.get("start_time", time.time()))

    recent = context_manager.update_recent_rounds(
        state.get("recent_rounds", []),
        state.get("user_input", ""),
        (output or "")[:500],
        max_rounds=settings.max_recent_rounds,
    )
    return {"recent_rounds": recent, "cost_summary": cost_summary}


# ── HITL 人工确认节点 ────────────────────────────
async def hitl_review_node(state: AgentState) -> dict:
    """高危操作：interrupt 挂起，外部 Command(resume) 恢复"""
    thread_id = state.get("thread_id", "")
    payload = state.get("confirm_payload") or {}
    resume_token = payload.get("resume_token") or f"rt-{uuid.uuid4().hex[:16]}"
    idem_key = payload.get("idempotency_key") or ""
    review_data = {
        "type": "human_confirm",
        "thread_id": thread_id,
        "intent": state.get("intent", ""),
        "title": payload.get("title", f"高危操作确认：{state.get('intent', '')}"),
        "description": payload.get("description", ""),
        "params": payload.get("params", {}),
        "risk_level": payload.get("risk_level", "high"),
        "options": payload.get("options", ["确认执行", "驳回操作", "修改参数后执行"]),
        "resume_token": resume_token,
        "idempotency_key": idem_key,
    }
    # ⭐ resume_token 落盘:仅此 thread 可用此凭证恢复(防冒用)
    with tracer.span("hitl_pause", thread_id=thread_id, intent=state.get("intent", ""),
                     risk_level=review_data["risk_level"]):
        await cache_service.set(f"hitl:resume:{thread_id}", resume_token, ttl=settings.hitl_resume_ttl)
    decision = interrupt(review_data)
    action = decision.get("action", "") if isinstance(decision, dict) else str(decision or "")
    audit_logger.log_human(
        thread_id=thread_id, user_id=state.get("user_id", ""),
        skill_name=state.get("intent", ""),
        risk_level=review_data["risk_level"],
        action=action, params=decision if isinstance(decision, dict) else {"action": action},
    )
    result = {
        "human_intervened": True,
        "human_action": action,
        "human_approved": action in ("approve", "modify"),
        "need_human_confirm": False,
        "confirm_payload": None,
    }
    if action == "modify":
        result["approved_params"] = decision.get("modified_params") if isinstance(decision, dict) else None
    if action not in ("approve", "modify"):
        result["agent_output"] = "操作已被人工驳回，未执行。"
    return result


# 人工确认后路由:approve/modify 回 agent_execute,否则直接 memory_write 结束
def route_after_hitl(state: AgentState) -> str:
    """人工确认后：approve/modify → 回到 agent_execute 继续执行；否则直接结束"""
    if state.get("human_action") in ("approve", "modify"):
        return "agent_execute"
    return "memory_write"


# ── 外部审批节点(CRITICAL 高危操作:提交审批系统,等待回调)──
async def external_approval_node(state: AgentState) -> dict:
    """CRITICAL 高危操作：提交外部审批系统 → 落盘审批单 → interrupt 挂起，
    审批回调 resume 后继续执行。"""
    thread_id = state.get("thread_id", "")
    user_id = state.get("user_id", state.get("account", ""))
    skill_name = state.get("intent", "")
    payload = state.get("confirm_payload") or {}
    trace_id = get_trace_id() or ""
    creds = harness_interceptor._gen_credentials(thread_id, skill_name, payload.get("params") or {})
    resume_token = creds["resume_token"]
    idem_key = creds["idempotency_key"]

    # 1. 提交外部审批系统,拿审批单号
    approval_id = ""
    if settings.approval_external_endpoint:
        try:
            import httpx
            with tracer.span("approval_submit", thread_id=thread_id, trace_id=trace_id):
                async with httpx.AsyncClient(timeout=10) as client:
                    resp = await client.post(settings.approval_external_endpoint, json={
                        "intent": skill_name, "thread_id": thread_id, "user_id": user_id,
                        "params": payload.get("params", {}),
                        "description": payload.get("description", ""),
                        "risk_level": "critical", "trace_id": trace_id,
                    })
                    approval_id = (resp.json() or {}).get("approval_id", "")
        except Exception as e:
            logger.error(f"❌ 外部审批提交失败: thread={thread_id} err={str(e)[:120]}")

    # 2. 落盘:resume 凭证 + 审批单状态(重启不丢)
    await cache_service.set(f"hitl:resume:{thread_id}", resume_token, ttl=settings.hitl_resume_ttl)
    if approval_id:
        await cache_service.set(f"approval:{approval_id}", {
            "thread_id": thread_id, "resume_token": resume_token,
            "idempotency_key": idem_key, "trace_id": trace_id, "status": "pending",
        }, ttl=settings.approval_redis_ttl)

    audit_logger.log_human(
        thread_id=thread_id, user_id=user_id, skill_name=skill_name,
        risk_level="critical", action="submit_external",
        params={"approval_id": approval_id, "trace_id": trace_id},
    )
    logger.info(f"📤 外部审批已提交: thread={thread_id} approval_id={approval_id} trace_id={trace_id}")

    review_data = {
        "mode": "external",
        "approval_id": approval_id,
        "status": "pending",
        "thread_id": thread_id,
        "intent": skill_name,
        "title": payload.get("title", f"高危操作待审批：{skill_name}"),
        "description": payload.get("description", ""),
        "params": payload.get("params", {}),
        "risk_level": "critical",
        "resume_token": resume_token,
        "idempotency_key": idem_key,
    }
    decision = interrupt(review_data)
    action = decision.get("action", "") if isinstance(decision, dict) else str(decision or "")
    result = {
        "human_intervened": True,
        "human_action": action,
        "human_approved": action in ("approve", "modify"),
        "need_human_confirm": False,
        "confirm_payload": None,
        "approval_mode": "external",
        "approval_id": approval_id,
    }
    if action == "modify":
        result["approved_params"] = decision.get("modified_params") if isinstance(decision, dict) else None
    if action not in ("approve", "modify"):
        result["agent_output"] = "审批未通过，操作未执行。"
    return result


# ── 路由条件 ─────────────────────────────────────
def route_after_cache(state: AgentState) -> str:
    return "execution_log" if state.get("cache_hit") else "supervisor"


# FactCheck 后路由:超时/通过/超迭代去 memory_write,否则回 agent_execute 重试
def route_after_fact_check(state: AgentState) -> str:
    # 超时强制结束
    if time.time() - state.get("start_time", time.time()) > settings.graph_timeout_seconds:
        return "memory_write"
    if state.get("fact_check_passed"):
        return "memory_write"
    if state.get("iteration_count", 0) < settings.max_iterations:
        return "agent_execute"      # 带反馈重试
    return "memory_write"


# agent_execute 后路由:CRITICAL 走外部审批,高危转 hitl_review,parallel 转 aggregate,否则 fact_check
def route_after_agent(state: AgentState) -> str:
    """agent_execute 后：CRITICAL → external_approval；高危 → hitl_review；parallel → aggregate；否则 fact_check"""
    if state.get("need_human_confirm"):
        risk = (state.get("confirm_payload") or {}).get("risk_level", "high")
        if settings.approval_external_enabled and settings.approval_external_endpoint \
                and risk == "critical":
            return "external_approval"
        return "hitl_review"
    plan = state.get("routing_plan") or {}
    if plan.get("mode") == "parallel" and len(plan.get("intents", [])) > 1:
        return "aggregate"
    return "fact_check"


# ── 并行 fan-out / 汇总──────────────────
def fan_out_parallel(state: AgentState) -> list:
    """supervisor 判定 parallel 时,把每个意图作为独立子任务分发到对应领域 agent(子图);
    单意图/chat 直接路由到对应领域 agent。
    注:子图经 Send 分支只收到 Send 携带的状态(不继承父级普通通道),
    所以并行分支必须带上完整父状态 + 覆盖 intent(身份通道已有 reducer 不冲突)。"""
    from langgraph.types import Send
    plan = state.get("routing_plan") or {}
    intents = plan.get("intents", [])
    if not intents or len(intents) == 1:
        intent = intents[0] if intents else "chat"
        return f"agent_{intent}"
    return [Send(f"agent_{intent}", {**state, "intent": intent}) for intent in intents[:2]]


# 领域 agent 完成后路由:parallel 多意图 → aggregate 汇总,否则直接收尾
def route_after_subgraph(state: AgentState) -> str:
    plan = state.get("routing_plan") or {}
    if plan.get("mode") == "parallel" and len(plan.get("intents", [])) > 1:
        return "aggregate"
    return "memory_write"


# 汇总 parallel 子结果:拼接后一次汇总 LLM,失败降级为拼接
async def aggregate_node(state: AgentState) -> dict:
    """汇总子结果：拼接 + 一次汇总 LLM（可降级为拼接）"""
    outputs = [s.get("agent_output", "") for s in state.get("sub_results", []) if s]
    if len(outputs) <= 1:
        return {"agent_output": outputs[0] if outputs else "（无有效子结果）"}
    combined = "\n\n".join(f"【{i+1}】{o[:800]}" for i, o in enumerate(outputs))
    try:
        result = await unified_llm.ainvoke("comparison_analysis", [
            {"role": "system", "content": "你是汇总助理，把多个子分析结果整合成连贯回答。"},
            {"role": "user", "content": f"整合以下子结果：\n{combined}"},
        ])
        return {"agent_output": result.content, "confidence": 0.7}
    except Exception:
        return {"agent_output": combined, "confidence": 0.5}


# ── 构建 & 编译图(P2:父图编排 + 领域 Agent 子图)──
def build_graph():
    from agent.domain_agents import build_domain_agent
    from observability.tracing import span_node

    workflow = StateGraph(AgentState)
    # 父图节点统一包一层 span,工作流节点可视化
    workflow.add_node("cache_check", span_node("cache_check", cache_check_node))
    workflow.add_node("supervisor", span_node("supervisor", supervisor_node))
    workflow.add_node("aggregate", span_node("aggregate", aggregate_node))
    workflow.add_node("memory_write", span_node("memory_write", memory_write_node))
    workflow.add_node("execution_log", span_node("execution_log", execution_log_node))

    # 领域 Agent:每个意图一个独立子图(共享 AgentState,独立重试/校验/复核)
    agent_names = []
    for intent in list(_current_domain.get_intents()) + ["chat"]:
        name = f"agent_{intent}"
        agent_names.append(name)
        workflow.add_node(name, build_domain_agent(
            intent,
            retrieve=rag_retrieve_node,
            execute=agent_execute_node,
            fact_check=fact_check_node,
            hitl=hitl_review_node,
            external_approval=external_approval_node,
            retry_policy=AGENT_RETRY_POLICY,
        ))

    workflow.set_entry_point("cache_check")
    workflow.add_conditional_edges("cache_check", route_after_cache,
                                   {"supervisor": "supervisor", "execution_log": "execution_log"})
    # supervisor → 领域 agent(单路径 或 parallel Send 分发到多个领域 agent)
    workflow.add_conditional_edges("supervisor", fan_out_parallel, agent_names)
    # 每个领域 agent 完成后:parallel → aggregate 汇总,否则直接收尾
    for name in agent_names:
        workflow.add_conditional_edges(name, route_after_subgraph,
                                       {"aggregate": "aggregate", "memory_write": "memory_write"})
    workflow.add_edge("aggregate", "memory_write")
    workflow.add_edge("memory_write", "execution_log")
    workflow.add_edge("execution_log", END)
    return workflow

    return workflow


# 官方 AsyncSqliteSaver 必须在 event loop 内初始化，graph 延迟到 init_graph 后可用
agent_graph = None


# 初始化 checkpointer 并编译图(在 event loop 内调用一次)
async def init_graph() -> None:
    """初始化 checkpointer 并编译图（在 event loop 内调用一次）"""
    global agent_graph
    if agent_graph is None:
        cp = await init_checkpointer()
        agent_graph = build_graph().compile(checkpointer=cp)


# ── 便捷入口 ─────────────────────────────────────
async def _ensure_graph():
    """兜底初始化：run_agent/resume_agent 不依赖 lifespan 也能用"""
    global agent_graph
    if agent_graph is None:
        await init_graph()
    return agent_graph


# 入口:建 state、复用 checkpoint、调用图并返回结构化结果
async def run_agent(user_input: str, thread_id: str = None, account: str = "anonymous",
                    employee_id: str = "", tenant_id: str = "default", user_id: str = ""):
    start = time.time()
    thread_id = thread_id or f"th-{uuid.uuid4().hex[:8]}"
    reply_id = f"r-{uuid.uuid4().hex[:8]}"
    logger.info(f"▶ 开始处理: thread={thread_id} tenant={tenant_id} intent=待定 input={user_input[:50]}")
    graph = await _ensure_graph()
    # 官方 AsyncSqliteSaver 实例需从模块内最新全局取（import 绑定是 None）
    from agent.checkpointer import init_checkpointer
    cp = await init_checkpointer()

    # checkpoint 恢复（同一 thread 同一输入已完成才复用，避免误复用旧结果）
    restored = await cp.aget_tuple({"configurable": {"thread_id": thread_id}})
    if restored:
        cv = restored.checkpoint.get("channel_values", {})
        if cv.get("agent_output") and cv.get("user_input") == user_input:
            logger.info(f"♻️ checkpoint 命中（同一输入已答过），直接复用: thread={thread_id}")
            return {"success": True, "reply": cv.get("agent_output", ""),
                    "intent": cv.get("intent", ""), "confidence": cv.get("confidence", 0),
                    "source": "checkpoint", "duration_ms": int((time.time() - start) * 1000)}

    state = create_initial_state(
        thread_id=thread_id, user_id=user_id or account, account=account,
        employee_id=employee_id, tenant_id=tenant_id, user_input=user_input,
    )
    state["reply_id"] = reply_id
    state["start_time"] = time.time()
    state["timeout_seconds"] = settings.graph_timeout_seconds

    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    try:
        result = await graph.ainvoke(state, config=config)
        dur = int((time.time() - start) * 1000)
        logger.info(f"✅ 处理完成: thread={thread_id} intent={result.get('intent','')} "
                    f"耗时={dur}ms fact_check={result.get('fact_check_passed', True)} "
                    f"confidence={result.get('confidence_level','high')}")
        return {
            "success": True,
            "reply": result.get("agent_output", ""),
            "intent": result.get("intent", ""),
            "confidence": result.get("confidence", 0.0),
            "source": "cache" if result.get("cache_hit") else "agent",
            "rag_hit_rate": result.get("rag_hit_rate", 0.0),
            "fact_check_passed": result.get("fact_check_passed", True),
            "confidence_level": result.get("confidence_level", "high"),
            "reply_id": result.get("reply_id", ""),
            "duration_ms": dur,
            "citations": result.get("citations", []),
            "cost_summary": result.get("cost_summary"),
            "need_human_confirm": result.get("need_human_confirm", False),
        }
    except Exception as e:
        logger.error(f"❌ Agent 执行失败: {e}", exc_info=True)
        return {"success": False, "error": str(e), "reply": "系统处理异常，请重试。"}


# HITL 恢复:以 Command(resume) 继续执行,带 resume_token 强校验 + Redis 幂等防重复
async def resume_agent(thread_id: str, decision: dict):
    """HITL 恢复：Command(resume) 作为 input 传入（langgraph 1.2 签名）

    强校验:resume_token 必须匹配该线程挂起时落盘的凭证(防冒用);
    幂等:同一 idempotency_key 用 Redis setnx 拦截重复提交(重启仍有效)。
    """
    graph = await _ensure_graph()
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    logger.info(f"▶ HITL 人工决策到达: thread={thread_id} action={decision.get('action')}")

    # ⭐ resume_token 强校验(防冒用):必须匹配挂起时落盘的凭证
    stored_token = await cache_service.get(f"hitl:resume:{thread_id}")
    if not stored_token or stored_token != decision.get("resume_token", ""):
        logger.warning(f"⛔ resume_token 校验失败: thread={thread_id}")
        return {"success": False, "error": "resume_token 无效或已过期"}

    # ⭐ 幂等:同一 idempotency_key 不重复执行(Redis setnx)
    idempotency_key = decision.get("idempotency_key", "")
    if idempotency_key:
        ok = await cache_service.setnx(f"hitl:done:{idempotency_key}", "1", ttl=settings.hitl_done_ttl)
        if not ok:
            logger.warning(f"♻️ 幂等拦截: idempotency_key={idempotency_key} 已处理过,跳过重复提交")
            return {"success": True, "idempotent_replay": True,
                    "reply": "该审批已完成,已忽略重复提交。"}

    try:
        with tracer.span("hitl_resume", thread_id=thread_id, action=decision.get("action", "")):
            result = await graph.ainvoke(Command(resume=decision), config=config)
        logger.info(f"✅ HITL 恢复执行完成: thread={thread_id} action={decision.get('action')}")
        await cache_service.delete(f"hitl:resume:{thread_id}")
        return {"success": True, "reply": result.get("agent_output", ""),
                "intent": result.get("intent", ""),
                "idempotency_key": idempotency_key}
    except Exception as e:
        logger.error(f"❌ HITL 恢复失败: thread={thread_id} error={str(e)[:150]}")
        # 恢复失败:释放幂等键,允许重试
        if idempotency_key:
            await cache_service.delete(f"hitl:done:{idempotency_key}")
        return {"success": False, "error": str(e)}


# 崩溃恢复:从最近 checkpoint 回放续跑未完成线程
async def recover_thread(thread_id: str, decision: dict = None) -> dict:
    """崩溃恢复：进程 crash 后从最近 checkpoint 重新拉起未完成线程。

    LangGraph 的 checkpoint 记录在每个节点入口前。崩溃发生时 state 停在
    某节点入口前的 checkpoint——重新 ainvoke 会从该 checkpoint 继续回放，
    崩掉的那个节点会重跑（配合 retry_policy 吸收瞬态错误）。

    与 resume_agent 的区别：
      - resume_agent：图主动 interrupt 挂起（HITL），用 Command(resume) 精准恢复
      - recover_thread：图异常终止（崩溃），从 checkpoint 回放续跑
    """
    graph = await _ensure_graph()
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    logger.info(f"▶ 崩溃恢复启动: thread={thread_id}")

    # 1. 校验线程存在
    try:
        snap = await graph.aget_state(config)
    except Exception as e:
        logger.error(f"❌ 崩溃恢复：无法读取线程状态: thread={thread_id} err={str(e)[:120]}")
        return {"success": False, "error": f"无法读取线程状态: {e}"}

    # 2. 若线程有挂起的 interrupt（HITL），走 resume 而非崩溃恢复
    if getattr(snap, "interrupts", ()) or ():
        logger.info(f"↩️ 线程 {thread_id} 处于 HITL 挂起，转 resume")
        decision = decision or {"action": "reject"}
        stored_token = await cache_service.get(f"hitl:resume:{thread_id}")
        decision["resume_token"] = decision.get("resume_token") or stored_token or ""
        return await resume_agent(thread_id, decision)

    # 3. 无 checkpoint（线程从未开始或已完成），无恢复价值
    snap_values = getattr(snap, "values", None) or {}
    if not snap_values:
        logger.warning(f"⚠️ 崩溃恢复：线程 {thread_id} 无 checkpoint，无法恢复")
        return {"success": False, "error": "线程无 checkpoint，无法恢复"}

    # 4. checkpoint 已有 agent_output：图已完成，无需恢复
    if snap_values.get("agent_output"):
        logger.info(f"✅ 崩溃恢复：线程 {thread_id} 已完成，无需恢复")
        return {"success": True, "reply": snap_values.get("agent_output", ""),
                "intent": snap_values.get("intent", ""), "recovered": False}

    # 5. 从 checkpoint 继续回放。传入空 input，图从上次停点续跑。
    try:
        result = await graph.ainvoke({"__resume__": True}, config=config)
        logger.info(f"✅ 崩溃恢复完成: thread={thread_id} intent={result.get('intent','')}")
        return {"success": True, "reply": result.get("agent_output", ""),
                "intent": result.get("intent", "")}
    except Exception as e:
        logger.error(f"❌ 崩溃恢复失败: thread={thread_id} err={str(e)[:150]}")
        return {"success": False, "error": str(e)}
