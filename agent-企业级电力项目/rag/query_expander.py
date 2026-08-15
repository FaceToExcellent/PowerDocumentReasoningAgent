"""查询扩展 — 轻量：意图关键词 + 同义词补充"""
from typing import List

_INTENT_SYNONYMS = {
    "spec_retrieval": ["规程", "标准", "规范", "DL/T", "GB", "Q/GDW", "图纸"],
    "cost_audit": ["造价", "结算", "定额", "预算", "费用", "每公里", "核价"],
    "doc_archive": ["归档", "资料", "验收", "竣工", "存档"],
    "grid_op": ["运维", "巡视", "巡检", "操作", "台账", "参数"],
    "fault_disposal": ["故障", "跳闸", "异常", "告警", "事故", "抢修"],
}


# 查询扩展器：按意图补充同义词关键词
class QueryExpander:
    # 按意图在原始查询上追加同义词，最多返回3条变体
    def expand(self, query: str, intent: str = "") -> List[str]:
        queries = [query]
        if intent and intent in _INTENT_SYNONYMS:
            for kw in _INTENT_SYNONYMS[intent]:
                if kw not in query:
                    queries.append(f"{query} {kw}")
        return queries[:3]


query_expander = QueryExpander()
