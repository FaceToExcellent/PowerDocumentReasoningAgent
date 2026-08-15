"""Agent 执行日志 — 复用 observability/audit 的 SQLite 审计，保持 write_log 接口兼容"""
from observability.audit import audit_logger


# 写 Agent 执行日志,复用审计模块落库
def write_log(thread_id, account, emp_id, intent, user_input, output, confidence,
              fc_passed, fc_errors, fc_feedback, duration_ms, iteration, cache_hit, rag_hit_rate,
              fallback_used=0, timeout=0, tenant_id="", user_id="", **kwargs):
    audit_logger.log_chat(
        tenant_id=tenant_id, user_id=user_id or account, thread_id=thread_id,
        intent=intent, user_input=user_input, agent_output=output, confidence=confidence,
        fact_check_passed=bool(fc_passed), duration_ms=duration_ms, cache_hit=bool(cache_hit),
        success=True, error=fc_feedback or "",
    )
