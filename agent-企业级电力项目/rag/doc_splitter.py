"""文档切片 — 企业版：标题层级感知 + chunk_index + 唯一 id + 父子元数据

本机实现（轻量）：按标题/段落切，带 section_path / chunk_index / chunk_id。
生产可无缝换 LlamaIndex HierarchicalNodeParser（M1.x 计划），接口兼容。
"""
import hashlib
import logging
from typing import List, Dict, Any

from config.settings import settings

logger = logging.getLogger(__name__)

_HEADING_RE = None  # 轻量标题匹配在 split 内做


# 按标题层级+段落切分文档，产出带元数据(含前后chunk链接)的chunks列表
def split_document(content: str, source: str = "", title: str = "",
                   chunk_size: int = None) -> List[Dict[str, Any]]:
    """按标题层级 + 段落切分，产出带元数据的 chunks

    Returns:
        [{"content": str, "metadata": {chunk_id, source, title, section_path,
                                        chunk_index, prev_id, next_id, ...}}]
    """
    chunk_size = chunk_size or settings.default_doc_chunk_size
    chunks = _split_by_heading_and_size(content, chunk_size)

    docs = []
    for i, (text, section_path) in enumerate(chunks):
        chunk_id = _make_chunk_id(source, i, text)
        meta = {
            "chunk_id": chunk_id,
            "source": source,
            "title": title,
            "section_path": section_path,
            "chunk_index": i,
            "prev_id": _make_chunk_id(source, i - 1, "") if i > 0 else "",
            "next_id": _make_chunk_id(source, i + 1, "") if i < len(chunks) - 1 else "",
        }
        docs.append({"content": text, "metadata": meta})
    return docs


# 生成稳定的chunk唯一id(来源+索引md5)
def _make_chunk_id(source: str, index: int, text: str) -> str:
    base = hashlib.md5(f"{source}:{index}".encode()).hexdigest()[:10]
    return f"{source}-{base}-{index:04d}" if source else f"doc-{base}-{index:04d}"


# 按疑似标题累积section_path，并按chunk_size聚块的轻量切分实现
def _split_by_heading_and_size(content: str, chunk_size: int) -> List[tuple]:
    """轻量实现：先按行切，遇到短行（疑似标题）累积到 section_path；按 chunk_size 聚块"""
    lines = content.splitlines()
    chunks: List[tuple] = []          # (text, section_path)
    current_lines: List[str] = []
    current_size = 0
    current_path: List[str] = []

    # 将当前累积的行写入chunk并重置缓冲
    def flush():
        nonlocal current_lines, current_size
        if current_lines:
            chunks.append(("\n".join(current_lines).strip(), list(current_path)))
            current_lines, current_size = [], 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # 疑似标题：较短 + 无句号结尾 + 以数字/章/第开头 → 更新 section_path
        is_heading = (
            len(stripped) <= 40
            and not stripped.endswith(("。", "；", "："))
            and any(k in stripped for k in ("第", "章", "节", "规程", "规定", "标准"))
        )
        if is_heading and current_lines:
            flush()
            current_path = _merge_path(current_path, stripped)

        current_lines.append(line)
        current_size += len(line)
        if current_size >= chunk_size:
            flush()

    flush()
    if not chunks:
        # 兜底：按 chunk_size 硬切
        for i in range(0, len(content), chunk_size):
            chunks.append((content[i:i + chunk_size], ["正文"]))
    return chunks


# 合并标题路径：新标题替换/追加，保持层级
def _merge_path(old: List[str], heading: str) -> List[str]:
    """简化：新标题替换最后一个元素（单层），够本机演示"""
    if old and len(old) >= 1:
        return [old[0]] if heading.startswith("第") and "章" in heading[:3] else old[:-1] + [heading]
    return [heading]


# ── LlamaIndex 语义切片(替换自研标题切片,接口兼容)──
def split_document_hierarchical(content: str, source: str = "", title: str = "",
                                parent_size: int = 1024, child_size: int = None) -> List[Dict[str, Any]]:
    """LlamaIndex HierarchicalNodeParser 语义切片:句子感知 + 父子分层。

    产出叶子(小 chunk)作为检索单元,带 parent_id 指向父 chunk(大上下文,供后续检索扩上下文)。
    输出形状与 split_document 兼容,可直接喂 rag_service.add_documents。
    """
    from llama_index.core import Document
    from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
    from llama_index.core.schema import NodeRelationship
    child_size = child_size or settings.default_doc_chunk_size or 512
    parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[parent_size, child_size])
    nodes = parser.get_nodes_from_documents([Document(text=content)])
    leaves = get_leaf_nodes(nodes)

    docs = []
    for i, node in enumerate(leaves):
        text = (node.text or "").strip()
        if not text:
            continue
        parent_id = ""
        try:
            rels = getattr(node, "relationships", {}) or {}
            parent = rels.get(NodeRelationship.PARENT)
            parent_id = getattr(parent, "node_id", "") or ""
        except Exception:
            pass
        meta = {
            "chunk_id": _make_chunk_id(source, i, text),
            "source": source,
            "title": title,
            "section_path": [],
            "chunk_index": i,
            "prev_id": _make_chunk_id(source, i - 1, "") if i > 0 else "",
            "next_id": _make_chunk_id(source, i + 1, "") if i < len(leaves) - 1 else "",
            "parent_id": parent_id,
            "type": "text",
            "is_table": False,
        }
        docs.append({"content": text, "metadata": meta})
    return docs if docs else split_document(content, source, title)   # 兜底走标题切片
