"""Milvus 向量库适配 — 基于 pymilvus.MilvusClient，本机 Milvus Lite / 生产分布式双模式

关键设计：
  - Milvus Lite（进程内，几十 MB）与分布式集群用同一个 MilvusClient API，
    靠 uri 切换：lite:///./data/milvus.db ↔ tcp://milvus-host:19530
  - 多租户隔离双保险：partition_names（分区）+ expr（tenant_id 过滤）
  - 单 collection 多分区：每个租户一个 partition，隔离 A/B 租户
"""
import logging
import os
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional

from pymilvus import MilvusClient, DataType
from pymilvus.client.types import IndexType
from pymilvus.milvus_client.index import IndexParams

from config.settings import settings
from rag.embedder import embedding_provider

logger = logging.getLogger(__name__)


# Milvus向量库适配：分区+expr双重租户隔离，单collection多分区
class MilvusVectorStore:
    COLLECTION = "power_docs"
    PARTITION_PREFIX = "tenant_"      # 每个租户一个分区
    DIM = settings.milvus_dim         # BGE-M3 输出维度 1024

    # 初始化URI、确保数据目录并建立连接
    def __init__(self):
        self.client: Optional[MilvusClient] = None
        self._uri = settings.milvus_db_uri
        self._ensure_db_dir()
        self._connect()

    # 本地模式确保db文件父目录存在
    def _ensure_db_dir(self):
        # pymilvus 3.x 本地模式：.db 文件路径，确保父目录存在
        if not self._uri.startswith(("tcp://", "http://", "https://")):
            Path(self._uri).parent.mkdir(parents=True, exist_ok=True)

    # 建立MilvusClient连接并确保collection就绪
    def _connect(self):
        self.client = MilvusClient(uri=self._uri)
        self._ensure_collection()
        logger.info(f"✅ Milvus 连接成功: {self._uri} (collection={self.COLLECTION})")

    # 确保collection存在：建schema、HNSW索引并load
    def _ensure_collection(self):
        if not self.client.has_collection(self.COLLECTION):
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
            schema.add_field("id", datatype=DataType.VARCHAR, is_primary=True, max_length=128)
            schema.add_field("vector", datatype=DataType.FLOAT_VECTOR, dim=self.DIM)
            schema.add_field("tenant_id", datatype=DataType.VARCHAR, max_length=64)
            # 动态字段：content / title / source / voltage / standards / chunk_index 等自动吸收
            self.client.create_collection(self.COLLECTION, schema=schema)
            # HNSW 索引
            index_params = IndexParams()
            index_params.add_index(
                field_name="vector", index_type=IndexType.HNSW,
                metric_type="COSINE", params={"M": 16, "efConstruction": 128},
            )
            self.client.create_index(self.COLLECTION, index_params)
            logger.info(f"Collection 创建完成: {self.COLLECTION} dim={self.DIM}")
        # 每次连接后确保 load（进程重启后可能处于 released 状态）
        try:
            state = self.client.get_load_state(self.COLLECTION)
            if state.get("state") not in ("Loaded", 3):
                self.client.load_collection(self.COLLECTION)
        except Exception:
            try:
                self.client.load_collection(self.COLLECTION)
            except Exception:
                pass

    # 确保租户分区存在，返回分区名(租户隔离第一道)
    def _ensure_partition(self, tenant_id: str) -> str:
        """确保租户分区存在。partition_names 隔离第一道"""
        partition = f"{self.PARTITION_PREFIX}{tenant_id}"
        if not self.client.has_partition(self.COLLECTION, partition):
            self.client.create_partition(self.COLLECTION, partition)
        return partition

    # ── 检索 ──
    def search(self, query: str, top_k: int = 5, filters: Optional[Dict] = None,
               tenant_id: str = "") -> List[Dict[str, Any]]:
        if not query:
            return []
        emb = embedding_provider.encode(query)[0]

        partition = self._ensure_partition(tenant_id) if tenant_id else None
        # expr 过滤：租户隔离第二道 + 元数据过滤
        exprs = []
        if tenant_id:
            exprs.append(f'tenant_id == "{tenant_id}"')
        for k, v in (filters or {}).items():
            if isinstance(v, str):
                exprs.append(f'{k} == "{v}"')
            elif isinstance(v, (int, float)):
                exprs.append(f"{k} == {v}")
        expr = " and ".join(exprs) if exprs else None

        try:
            results = self.client.search(
                self.COLLECTION,
                data=[emb],
                anns_field="vector",
                limit=top_k,
                search_params={"metric_type": "COSINE", "params": {"nprobe": 16}},
                partition_names=[partition] if partition else None,
                filter=expr,               # pymilvus 3.x 参数名是 filter（expr 已废弃）
                output_fields=["id", "tenant_id", "content", "source", "title"],
            )
        except Exception as e:
            logger.warning(f"Milvus 检索失败: {e}")
            return []

        out = []
        for hit in results[0]:
            entity = hit.get("entity", {})
            doc_id = entity.get("id", "")
            out.append({
                "doc": {"content": self._load_content(doc_id, entity),
                        "metadata": {"title": doc_id.split("-")[0] if "-" in doc_id else "",
                                     "source": doc_id, "chunk_id": doc_id,
                                     "tenant_id": entity.get("tenant_id", tenant_id)}},
                "score": float(hit.get("distance", 0)),
                "chunk_id": doc_id,
            })
        return out

    # 优先从动态字段取content，否则回退到id编码的摘要
    def _load_content(self, doc_id: str, entity: dict) -> str:
        """优先从动态字段取 content，否则从 id 反查（id 里编码了内容摘要）"""
        return entity.get("content", doc_id)

    # ── 入库 ──
    def add_documents(self, docs: List[Dict[str, Any]], tenant_id: str = "") -> int:
        if not docs:
            return 0
        partition = self._ensure_partition(tenant_id) if tenant_id else None
        contents = [d["content"] for d in docs]
        # 批量向量化
        embs = embedding_provider.encode(contents)
        rows = []
        for i, (doc, emb) in enumerate(zip(docs, embs)):
            meta = doc.get("metadata", {})
            chunk_id = meta.get("chunk_id") or f"{doc.get('source', 'doc')}-{uuid.uuid4().hex[:8]}"
            row = {
                "id": chunk_id,
                "vector": emb,
                "tenant_id": tenant_id or meta.get("tenant_id", ""),
                "content": doc["content"],
                "source": meta.get("source", ""),
                "title": meta.get("title", ""),
                "voltage": meta.get("voltage", ""),
                "standards": meta.get("standards", ""),
                "chunk_index": meta.get("chunk_index", i),
            }
            rows.append(row)
        # 分批写入（每批 100，防内存/请求过大）
        inserted = 0
        for i in range(0, len(rows), 100):
            batch = rows[i:i + 100]
            try:
                self.client.insert(self.COLLECTION, batch, partition_name=partition)
                inserted += len(batch)
            except Exception as e:
                logger.error(f"Milvus 批量入库失败: {e}")
        self.client.flush(self.COLLECTION)
        return inserted

    # 按租户过滤删除全部文档
    def delete_by_tenant(self, tenant_id: str) -> bool:
        try:
            self.client.delete(self.COLLECTION, filter=f'tenant_id == "{tenant_id}"')
            return True
        except Exception as e:
            logger.error(f"删除租户数据失败: {e}")
            return False

    # 按id列表删除文档
    def delete_by_ids(self, ids: List[str]):
        try:
            self.client.delete(self.COLLECTION, ids=ids)
        except Exception as e:
            logger.error(f"按 id 删除失败: {e}")

    # 统计文档数(可选按租户过滤)
    def count(self, tenant_id: str = "") -> int:
        try:
            if tenant_id:
                return self.client.query(self.COLLECTION, filter=f'tenant_id == "{tenant_id}"', output_fields=["id"])
            return self.client.query(self.COLLECTION, output_fields=["id"])
        except Exception:
            return 0

    # 按租户查询原始数据(管理/调试用)；include_content=True 时带 content 供 BM25 建索引
    def query(self, tenant_id: str = "", limit: int = 20, include_content: bool = False) -> list:
        """按租户查原始数据（管理/调试用）；include_content=True 时带 content 供 BM25 建索引"""
        try:
            fields = ["id", "tenant_id", "title", "source"]
            if include_content:
                fields.append("content")
            if tenant_id:
                return self.client.query(self.COLLECTION, filter=f'tenant_id == "{tenant_id}"',
                                         output_fields=fields, limit=limit)
            return self.client.query(self.COLLECTION, output_fields=fields, limit=limit)
        except Exception as e:
            logger.error(f"查询失败: {e}")
            return []
