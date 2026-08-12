"""Harness 前置拦截器 — 高危操作检测，返回是否需人工确认"""
import hashlib
import json
import logging
import uuid
from typing import Dict, Any, Optional

from agent.harness.risk_level import RiskLevel, get_risk_level
from observability.audit import audit_logger

logger = logging.getLogger(__name__)


class InterceptResult:
    def __init__(self, need_confirm: bool, risk_level: RiskLevel = RiskLevel.LOW,
                 message: Optional[Dict] = None):
        self.need_confirm = need_confirm
        self.risk_level = risk_level
        self.message = message or {}


class HarnessInterceptor:
    """所有 Skill 执行前必经的强制拦截层，无法绕过"""

    async def before_skill_execute(self, skill_name: str, params: dict,
                                   user_id: str = "", thread_id: str = "") -> InterceptResult:
        risk = get_risk_level(skill_name)
        if risk in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            return InterceptResult(
                need_confirm=True,
                risk_level=risk,
                message=self._build_confirm(skill_name, params, risk, thread_id),
            )
        if risk == RiskLevel.MEDIUM:
            # 中风险：自动执行但记录提示
            logger.info(f"[harness] 中风险 Skill {skill_name} 自动执行（已记录）")
        return InterceptResult(need_confirm=False, risk_level=risk)

    def _build_confirm(self, skill_name: str, params: dict, risk: RiskLevel,
                       thread_id: str = "") -> dict:
        # ⭐ resume_token:防冒用凭证(仅凭 thread_id 不能恢复高危操作)
        resume_token = uuid.uuid4().hex[:16]
        # ⭐ idempotency_key:防重复提交(同一恢复命中同一 key 不重复执行)
        raw = f"{thread_id}:{skill_name}:{json.dumps(params, ensure_ascii=False)}"
        idempotency_key = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return {
            "type": "human_confirm",
            "title": f"高危操作确认：{skill_name}",
            "risk_level": risk.value,
            "params": params,
            "description": f"即将执行 {skill_name}，参数：{json.dumps(params, ensure_ascii=False)[:200]}",
            "options": ["确认执行", "驳回操作", "修改参数后执行"],
            "resume_token": resume_token,        # 恢复凭证
            "idempotency_key": idempotency_key,  # 幂等键
            "thread_id": thread_id,
        }

    async def record_human_action(self, *, thread_id, user_id, skill_name, risk_level,
                                  action, params=None, reason=""):
        """人工操作全量审计"""
        audit_logger.log_human(
            thread_id=thread_id, user_id=user_id, skill_name=skill_name,
            risk_level=risk_level, action=action, params=params, reason=reason,
        )


harness_interceptor = HarnessInterceptor()
