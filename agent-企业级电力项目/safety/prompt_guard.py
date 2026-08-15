"""Prompt 注入防护 — 外部文本分源扫描 + 中和 + 脱敏 + 数据边界

- ExternalText 分源(user/tool/rag),各自扫描
- scan_categories 三类风险:prompt_injection / secret_or_reasoning_request / privacy
- 风险分级处理:脏指令中和、隐私脱敏、系统信息请求拒绝
- 数据边界前缀 [CLEAN/TAINTED],外部文本只能作为数据不能覆盖系统规则
"""
import logging
import re
from typing import Any, Dict, List

from config.settings import settings
from config.domain import get_domain

logger = logging.getLogger(__name__)

# 注入/越权指令标记：基础集来自 settings，叠加当前领域扩展（DomainConfig.injection_markers）
_domain = get_domain(settings.domain)
_INJECTION_MARKERS = list(settings.safety_injection_markers) + list(_domain.injection_markers)
# 索要系统信息
_SECRET_MARKERS = list(settings.safety_secret_markers) + list(_domain.secret_markers)
# 隐私(手机号/地址)
_PHONE_RE = re.compile(r"1[3-9]\d{9}")
_ADDR_RE = re.compile(r"(地址|收货|住址)[:：]?\S{4,30}")


# 外部文本数据类:携带来源类型与来源 ID 供扫描
class ExternalText:
    """外部文本:带来源与来源ID(用于扫描与边界标记)。"""

    # 初始化外部文本的来源与内容
    def __init__(self, source_type: str, text: str, source_id: str = ""):
        self.source_type = source_type   # user / tool / rag
        self.source_id = source_id
        self.text = text


# 安全扫描结果:记录来源/风险分类/脱敏内容与处置
class SafetyScan:
    # 初始化扫描结果各字段
    def __init__(self, source_type: str, source_id: str, categories: List[str],
                 tainted: bool, sanitized_content: str, allowed_for_model: bool,
                 handling: str):
        self.source_type = source_type
        self.source_id = source_id
        self.categories = categories
        self.tainted = tainted
        self.sanitized_content = sanitized_content
        self.allowed_for_model = allowed_for_model
        self.handling = handling

    # 将扫描结果转为字典(供上报与日志)
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_type": self.source_type, "source_id": self.source_id,
            "categories": self.categories, "tainted": self.tainted,
            "sanitized_content": self.sanitized_content[:120],
            "allowed_for_model": self.allowed_for_model, "handling": self.handling,
        }


# 识别文本的风险分类(注入/索密/隐私)
def _scan_categories(text: str) -> List[str]:
    """识别文本风险分类。"""
    categories = []
    lowered = text.lower()
    if any(m in lowered for m in _INJECTION_MARKERS):
        categories.append("prompt_injection")
    if any(m in lowered for m in _SECRET_MARKERS):
        categories.append("secret_or_reasoning_request")
    if _PHONE_RE.search(text) or _ADDR_RE.search(text):
        categories.append("privacy")
    return categories


# 中和脏指令并脱敏隐私信息
def _sanitize(text: str) -> str:
    """中和脏指令 + 脱敏隐私。"""
    # 中和注入指令(保留证据但执行不了)
    for marker in _INJECTION_MARKERS:
        if marker in text:
            text = re.sub(re.escape(marker), "[已隔离的外部指令]", text, flags=re.IGNORECASE)
    # 脱敏
    text = _PHONE_RE.sub("[phone-redacted]", text)
    text = _ADDR_RE.sub(r"\1[已脱敏地址]", text)
    return text


# 扫描单条外部文本并决定放行/中和/脱敏
def scan_external_text(ext: ExternalText) -> SafetyScan:
    """扫描单条外部文本,决定是否放行 + 脱敏内容。"""
    categories = _scan_categories(ext.text)
    handling = "pass"
    if "secret_or_reasoning_request" in categories:
        # 索要系统信息:不进入模型上下文
        handling = "blocked"
        allowed = False
        sanitized = "[已隔离受保护信息请求]"
    elif "prompt_injection" in categories:
        # 注入:可进但降权中和
        handling = "neutralized"
        allowed = True
        sanitized = _sanitize(ext.text)
    elif "privacy" in categories:
        handling = "sanitized"
        allowed = True
        sanitized = _sanitize(ext.text)
    else:
        allowed = True
        sanitized = ext.text

    tainted = bool(categories)
    # 数据边界前缀
    prefix = "[TAINTED/{}]".format(ext.source_type) if tainted else "[CLEAN/{}]".format(ext.source_type)
    sanitized = f"{prefix}{sanitized}"
    return SafetyScan(ext.source_type, ext.source_id, categories, tainted,
                      sanitized, allowed, handling)


# 统一扫描用户消息与外部文本,输出安全决策
def build_safety_decision(user_message: str, external_texts: List[ExternalText]) -> Dict[str, Any]:
    """对用户消息 + 外部文本(工具/RAG)统一扫描,输出安全决策。"""
    source_scans = []
    blocked_topics = []

    # 用户消息本身也扫描(直接注入)
    user_scan = scan_external_text(ExternalText("user", user_message))
    source_scans.append(user_scan.to_dict())
    if not user_scan.allowed_for_model:
        blocked_topics.append("system_prompt_or_hidden_reasoning")

    for ext in external_texts:
        scan = scan_external_text(ext)
        source_scans.append(scan.to_dict())

    return {
        "blocked_user_request": len(blocked_topics) > 0,
        "refused_topics": blocked_topics,
        "source_scans": source_scans,
        "public_summary": f"扫描 {len(source_scans)} 个来源,"
                          f"{sum(1 for s in source_scans if s['tainted'])} 个带风险标记",
    }


# 重组脱敏上下文,返回安全后的用户消息与 RAG 文本
def sanitize_context_for_model(user_message: str, rag_context: str = "",
                               tool_context: str = "") -> Dict[str, Any]:
    """重组脱敏上下文:返回安全后的 user 消息 + 数据边界包裹的外部文本。"""
    decision = build_safety_decision(
        user_message,
        [ExternalText("rag", rag_context, "rag_snippet")] if rag_context else [],
    )
    # 用户消息若含注入,取中和版
    user_scan = decision["source_scans"][0]
    safe_user = user_scan["sanitized_content"] if user_scan["tainted"] else user_message
    safe_rag = ""
    for scan in decision["source_scans"][1:]:
        if scan["source_type"] == "rag" and scan["allowed_for_model"]:
            safe_rag = scan["sanitized_content"]
    return {
        "safety_decision": decision,
        "safe_user_message": safe_user,
        "safe_rag_context": safe_rag,
    }


prompt_guard = None  # 占位:核心逻辑为纯函数,无需单例
