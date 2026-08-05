"""MQ 生产者/消费者 — 文档摄入异步（本地 Redis transport）"""
import json
import logging
from typing import Optional

from mq.transport import mq_transport, MQTransport

logger = logging.getLogger(__name__)

TOPIC_DOC_INGEST = "doc_ingest_topic"


class MQProducer:
    def __init__(self, transport: MQTransport = mq_transport):
        self.t = transport

    async def send_doc_ingest(self, file_info: dict, tenant_id: str) -> str:
        return await self.t.send(TOPIC_DOC_INGEST, {
            "file_id": file_info.get("file_id", ""),
            "file_name": file_info.get("file_name", ""),
            "tenant_id": tenant_id,
            "content": file_info.get("content", ""),
        }, tags=file_info.get("file_type", "text"))


mq_producer = MQProducer()


class DocIngestConsumer:
    """文档摄入消费者：解析 → 切片 → 入库 → 更新状态"""

    def __init__(self, transport: MQTransport = mq_transport):
        self.t = transport

    async def start(self):
        logger.info("📥 文档摄入消费者启动（本地 Redis transport）")
        await self.t.consume(TOPIC_DOC_INGEST, self._handle)

    async def _handle(self, msg: dict) -> bool:
        body = msg.get("body", {})
        tenant_id = body.get("tenant_id", "default")
        content = body.get("content", "")
        file_name = body.get("file_name", "doc.txt")
        try:
            from rag.doc_splitter import split_document
            from rag.retriever import rag_service
            docs = split_document(content, source=file_name, title=file_name.split(".")[0])
            n = rag_service.add_documents(docs, tenant_id=tenant_id)
            logger.info(f"✅ 文档 {file_name} 消费完成: {n} 片")
            return True
        except Exception as e:
            logger.error(f"文档消费失败: {e}")
            return False


doc_ingest_consumer = DocIngestConsumer()
