"""Chroma 向量库适配 — 开发 fallback，与 Milvus 同一接口"""
import logging
import uuid
from typing import Dict, List, Any, Optional

from rag.embedder import embedding_provider

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    COLLECTION = "power_docs"

    def __init__(self, persist_dir: str = "./data/chroma_db"):
        self._client = None
        self._collection = None
        self.persist_dir = persist_dir

    def _get_collection(self):
        if self._collection is not None:
            return self._collection
        import chromadb
        self._client = chromadb.PersistentClient(path=self.persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION, metadata={"hnsw:space": "cosine"}
        )
        return self._collection

    def search(self, query: str, top_k: int = 5, filters: Optional[Dict] = None,
               tenant_id: str = "") -> List[Dict[str, Any]]:
        if not query:
            return []
        collection = self._get_collection()
        emb = embedding_provider.encode(query)[0]

        where = {}
        if tenant_id:
            where["tenant_id"] = tenant_id
        for k, v in (filters or {}).items():
            where[k] = v

        try:
            results = collection.query(
                query_embeddings=[emb], n_results=top_k,
                where=where if where else None,
            )
        except Exception as e:
            logger.warning(f"Chroma 检索失败: {e}")
            return []

        out = []
        if results and results.get("ids"):
            for i, doc_id in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][i] if results.get("metadatas") else {}
                content = results["documents"][0][i] if results.get("documents") else ""
                distance = results["distances"][0][i] if results.get("distances") else 0
                out.append({
                    "doc": {"content": content, "metadata": meta},
                    "score": float(1.0 - distance) if distance else 1.0,
                    "chunk_id": doc_id,
                })
        return out

    def add_documents(self, docs: List[Dict[str, Any]], tenant_id: str = "") -> int:
        if not docs:
            return 0
        collection = self._get_collection()
        contents = [d["content"] for d in docs]
        embs = embedding_provider.encode(contents)
        ids, metas = [], []
        for i, (doc, emb) in enumerate(zip(docs, embs)):
            meta = dict(doc.get("metadata", {}))
            meta["tenant_id"] = tenant_id or meta.get("tenant_id", "")
            ids.append(meta.get("chunk_id") or f"{meta.get('source','doc')}-{uuid.uuid4().hex[:8]}")
            metas.append(meta)
        collection.add(ids=ids, embeddings=embs, documents=contents, metadatas=metas)
        return len(docs)

    def delete_by_tenant(self, tenant_id: str) -> bool:
        try:
            collection = self._get_collection()
            existing = collection.get(where={"tenant_id": tenant_id})
            if existing and existing.get("ids"):
                collection.delete(ids=existing["ids"])
            return True
        except Exception as e:
            logger.error(f"Chroma 删除租户数据失败: {e}")
            return False

    def count(self, tenant_id: str = "") -> int:
        try:
            collection = self._get_collection()
            return collection.count()
        except Exception:
            return 0
