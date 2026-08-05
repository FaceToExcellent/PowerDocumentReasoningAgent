"""上下文管理 — 简化：recent_rounds 维护 + 实体抽取占位（记忆已迁移到 memory 模块）"""
import logging
import re
from typing import Dict, List

logger = logging.getLogger(__name__)


class ContextManager:
    """保留最小职责：recent_rounds 更新与实体抽取；长对话记忆归 memory 模块管"""

    def extract_entities(self, text: str) -> List[str]:
        """抽取电力设备/电压等级实体（简化规则）"""
        entities = []
        for kw in ["主变", "变压器", "母线", "断路器", "熔断器", "互感器", "避雷器", "线路"]:
            if kw in text:
                entities.append(kw)
        for v in re.findall(r"\d+[kVKVA]+", text):
            entities.append(v)
        return list(set(entities))

    def update_recent_rounds(self, rounds: List[Dict], user_input: str, agent_output: str,
                             max_rounds: int = 8) -> List[Dict]:
        rounds = list(rounds)
        rounds.append({"role": "user", "content": user_input})
        rounds.append({"role": "assistant", "content": agent_output})
        return rounds[-max_rounds:]


context_manager = ContextManager()
