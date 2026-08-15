#!/usr/bin/env python3
"""灌入演示文档 — 按当前领域从 DomainConfig.demo_docs 读取（启动 --seed 或手动执行）"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


# 按领域灌入演示文档并返回入库片数
def seed_demo_docs(tenant_id: str = "default", domain_name: str = None):
    from config.settings import settings
    from config.domain import get_domain
    from rag.doc_splitter import split_document
    from rag.retriever import rag_service

    domain = get_domain(domain_name or settings.domain)
    print(f"🌍 领域: {domain.label} ({domain.name}) — 灌入演示文档")

    total = 0
    for doc in domain.demo_docs:
        chunks = split_document(doc["content"], source=doc["source"], title=doc["title"])
        n = rag_service.add_documents(chunks, tenant_id=tenant_id)
        total += n
        print(f"  ✅ {doc['source']} ({doc['title']}) → {n} 片")
    print(f"总入库 {total} 片 (tenant={tenant_id})")
    return total


if __name__ == "__main__":
    seed_demo_docs()
