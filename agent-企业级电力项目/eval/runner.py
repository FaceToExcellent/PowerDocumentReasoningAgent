"""回归评测运行器 — /eval/run 固定 case 结构化断言

用法:
  GET/POST /eval/run            → 跑全部固定 case
  POST /eval/run  {"case_ids": ["EVAL-001"]}  → 只跑指定 case

断言维度(对齐 AgentState / SSE done 返回):
  intent / citations / model_backend / rag_hit_rate / need_human_confirm / cost_summary / forbidden
"""
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml

logger = logging.getLogger(__name__)

CASES_PATH = Path(__file__).parent / "cases.yml"


# 加载评测用例:从 cases.yml 读取并返回用例列表
def load_cases() -> List[Dict[str, Any]]:
    if not CASES_PATH.exists():
        return []
    with CASES_PATH.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("cases", [])


# 断言单字段,返回失败原因列表
def _check_expected(section: Dict[str, Any], name: str, actual: Any) -> List[str]:
    """断言单字段,返回失败原因列表。"""
    failures = []
    if name == "intent":
        if actual != section[name]:
            failures.append(f"intent 期望 {section[name]} 实际 {actual}")
    elif name == "model_backend":
        if actual != section[name]:
            failures.append(f"model_backend 期望 {section[name]} 实际 {actual}")
    elif name == "citations_required":
        if section[name] and not actual:
            failures.append("期望有 citations 但为空")
    elif name == "cost_summary_required":
        if section[name] and not actual:
            failures.append("期望有 cost_summary 但为空")
    elif name == "rag_hit_rate":
        if actual != section[name]:
            failures.append(f"rag_hit_rate 期望 {section[name]} 实际 {actual}")
    elif name == "need_human_confirm":
        if section[name] != bool(actual):
            failures.append(f"need_human_confirm 期望 {section[name]} 实际 {actual}")
    return failures


# 断言回答中不出现禁止文本
def _check_forbidden(forbidden: List[str], reply: str) -> List[str]:
    """断言回答中不出现禁止文本。"""
    failures = []
    if not reply:
        return failures
    for word in forbidden:
        if word in reply:
            failures.append(f"回答包含禁止文本: '{word}'")
    return failures


# 运行固定 case 回归评测并汇总结果
async def run_eval(case_ids: List[str] | None = None) -> Dict[str, Any]:
    """运行固定 case 回归评测。"""
    from agent.graph import run_agent

    cases = load_cases()
    if case_ids:
        cases = [c for c in cases if c["id"] in case_ids]

    results = []
    for case in cases:
        case_id = case["id"]
        expected = case.get("expected", {})
        start = time.time()
        failed_checks: List[str] = []
        passed = False

        try:
            r = await run_agent(case["input"], thread_id=f"eval-{case_id}",
                                tenant_id="eval")
            reply = r.get("reply", "")
            # 路由:优先取 state 里的 intent
            intent = r.get("intent", "")
            citations = r.get("citations") or []
            cost_summary = r.get("cost_summary")
            rag_hit_rate = r.get("rag_hit_rate", 0.0)
            need_human = r.get("need_human_confirm", False)
            # model_backend: 从 cost_summary 或 fallback
            model_backend = ""
            if isinstance(cost_summary, dict):
                model_backend = cost_summary.get("model_backend", "")

            # 逐项断言
            for key in ("intent", "model_backend", "citations_required",
                        "cost_summary_required", "rag_hit_rate", "need_human_confirm"):
                if key in expected:
                    val = {"citations_required": citations,
                           "cost_summary_required": cost_summary,
                           "intent": intent,
                           "model_backend": model_backend,
                           "rag_hit_rate": rag_hit_rate,
                           "need_human_confirm": need_human}[key]
                    failed_checks += _check_expected(expected, key, val)
            failed_checks += _check_forbidden(expected.get("forbidden", []), reply)
            passed = len(failed_checks) == 0
        except Exception as e:
            logger.error(f"Eval {case_id} 执行异常: {e}")
            failed_checks.append(f"执行异常: {str(e)[:100]}")

        results.append({
            "id": case_id,
            "name": case.get("name", ""),
            "passed": passed,
            "duration_ms": int((time.time() - start) * 1000),
            "failures": failed_checks,
        })

    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    return {
        "schema_version": "eval_report_v1",
        "summary": {"total": total, "passed": passed_count, "failed": total - passed_count},
        "results": results,
    }
