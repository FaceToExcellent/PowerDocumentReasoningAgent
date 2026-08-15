"""元数据过滤构建 — 统一 filters → 向量库 where/expr"""
from typing import Dict, Any, Optional


# 元数据预过滤：构建白名单过滤条件供向量库使用
class MetadataPreFilter:
    # 只保留白名单字段的过滤条件，避免注入非法键
    @staticmethod
    def build_filters(filters: Optional[Dict]) -> Dict:
        """白名单过滤：只保留支持字段，避免注入非法键"""
        if not filters:
            return {}
        allowed = {"voltage", "doc_type", "standards", "devices", "source"}
        return {k: v for k, v in filters.items() if k in allowed and v}
