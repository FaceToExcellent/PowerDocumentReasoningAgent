"""元数据过滤构建 — 统一 filters → 向量库 where/expr"""
from typing import Dict, Any, Optional


class MetadataPreFilter:
    @staticmethod
    def build_filters(filters: Optional[Dict]) -> Dict:
        """白名单过滤：只保留支持字段，避免注入非法键"""
        if not filters:
            return {}
        allowed = {"voltage", "doc_type", "standards", "devices", "source"}
        return {k: v for k, v in filters.items() if k in allowed and v}
