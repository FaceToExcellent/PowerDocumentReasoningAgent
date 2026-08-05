"""向量库抽象层 — 上层零改动，Milvus/Chroma 可切换"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional


class BaseVectorStore(ABC):
    """统一向量库接口：search / add_documents / delete_by_tenant"""

    @abstractmethod
    def search(self, query: str, top_k: int = 5, filters: Optional[Dict] = None,
               tenant_id: str = "") -> List[Dict[str, Any]]:
        """语义检索。filters 额外元数据过滤；tenant_id 指定租户分区（隔离）"""
        ...

    @abstractmethod
    def add_documents(self, docs: List[Dict[str, Any]], tenant_id: str = "") -> int:
        """批量入库。doc: {"content": str, "metadata": dict}"""
        ...

    @abstractmethod
    def delete_by_tenant(self, tenant_id: str) -> bool:
        """删除某租户全部文档"""
        ...

    def count(self, tenant_id: str = "") -> int:
        """统计文档数（可选实现）"""
        return 0
