"""文档解析 — 基于 unstructured,按格式分流解析为结构化 chunk(P1)

设计原则:
  - 结构化优先:表格(text_as_html)/正文/标题分离,表格永不截断
  - 输出形状与 split_document 兼容(chunk_id/section_path/prev_id/next_id),
    可直接喂 rag_service.add_documents
  - 扫描件 PDF 走 hi_res OCR 策略;PDF 内图片/图注为后续阶段(多模态模型),
    本模块对 Image 元素仅留占位,不影响主链路
"""
import hashlib
from io import BytesIO
from typing import Any, Dict, List, Optional

from config.settings import settings

_HEADING_CATS = {"Title", "Header", "SectionHeader", "PageBreak"}


def _make_chunk_id(source: str, index: int, text: str) -> str:
    base = hashlib.md5(f"{source}:{index}".encode()).hexdigest()[:10]
    return f"{source}-{base}-{index:04d}" if source else f"doc-{base}-{index:04d}"


def _partition(content: bytes, filename: str, ext: str) -> List[Any]:
    """按格式分流调用 unstructured partition(懒加载,只导需要的解析器)。"""
    from unstructured.partition.text import partition_text
    file = BytesIO(content)
    kwargs = {"file": file, "file_filename": filename}
    if ext in ("txt", "md", "markdown"):
        return partition_text(**kwargs)
    if ext == "pdf":
        # 文本型 PDF 走 auto;扫描件可切 strategy="hi_res"(更慢,需模型)
        from unstructured.partition.pdf import partition_pdf
        return partition_pdf(**kwargs, strategy="auto")
    if ext == "docx":
        from unstructured.partition.docx import partition_docx
        return partition_docx(**kwargs)
    if ext in ("xlsx", "xls"):
        from unstructured.partition.xlsx import partition_xlsx
        return partition_xlsx(**kwargs)
    if ext in ("png", "jpg", "jpeg", "bmp", "tiff"):
        from unstructured.partition.image import partition_image
        return partition_image(**kwargs, strategy="hi_res")
    if ext == "pptx":
        from unstructured.partition.pptx import partition_pptx
        return partition_pptx(**kwargs)
    from unstructured.partition.auto import partition
    return partition(**kwargs)


def _image_caption(source: str, page, idx: int) -> str:
    """图注接口:配置本地视觉模型时生成图注,否则占位(当前默认不启用,不拉模型)。"""
    if getattr(settings, "vision_model_enabled", False) and settings.vision_model_enabled:
        try:
            # 预留:调本地视觉模型(qwen2.5-vl / llava)描述图片,产出可检索图注
            model = settings.vision_model_name or "qwen2.5-vl"
            # from ollama import Client; ... (按需接入)
            pass
        except Exception:
            pass
    return f"[图片 {source} 第{page or '?'}页:图注待生成]"


def _elements_to_chunks(elements: List[Any], source: str, title: str) -> List[Dict[str, Any]]:
    """把 unstructured 元素流组装成切片列表(表格独立成块,不截断)。"""
    raw: List[Dict[str, Any]] = []   # {content, section, is_table, page}
    section_path: List[str] = []
    buf: List[str] = []
    buf_size = 0
    chunk_size = settings.default_doc_chunk_size or 512

    def flush(section: List[str]):
        nonlocal buf, buf_size
        if buf:
            raw.append({"content": "\n".join(buf).strip(), "section": list(section),
                        "is_table": False, "page": None})
            buf, buf_size = [], 0

    for el in elements:
        cat = getattr(el, "category", "")
        text = (getattr(el, "text", "") or "").strip()
        text_as_html = ""
        try:
            text_as_html = (el.metadata.text_as_html or "") if el.metadata else ""
        except Exception:
            pass
        page = getattr(getattr(el, "metadata", None), "page_number", None)

        if cat in _HEADING_CATS:
            if text:
                section_path.append(text[:60])
            continue  # 标题只记 section,不进正文(避免检索到重复标题)
        if cat == "Table":
            flush(section_path)
            raw.append({"content": text_as_html or text, "section": list(section_path),
                        "is_table": True, "page": page})
            continue
        if cat in ("Image", "Figure", "FigureCaption"):
            if text:
                buf.append(text)   # 已有图注文字,进正文
            else:
                raw.append({"content": _image_caption(source, page, len(raw)),
                            "section": list(section_path), "is_table": False, "page": page})
            continue
        if text:
            buf.append(text)
            buf_size += len(text)
            if buf_size >= chunk_size:
                flush(section_path)

    flush(section_path)

    docs = []
    for i, c in enumerate(raw):
        content = c["content"]
        if not content:
            continue
        meta = {
            "chunk_id": _make_chunk_id(source, i, content),
            "source": source,
            "title": title,
            "section_path": c["section"],
            "chunk_index": i,
            "prev_id": _make_chunk_id(source, i - 1, "") if i > 0 else "",
            "next_id": _make_chunk_id(source, i + 1, "") if i < len(raw) - 1 else "",
            "type": "table" if c["is_table"] else "text",
            "is_table": c["is_table"],
            "page_number": c.get("page"),
        }
        docs.append({"content": content, "metadata": meta})
    return docs


def parse_document(content: bytes, filename: str, source: str = "", title: str = "") -> List[Dict[str, Any]]:
    """解析上传文档 → 结构化 chunks(与 split_document 输出形状兼容)。"""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    elements = _partition(content, filename, ext)
    return _elements_to_chunks(elements, source or filename, title or filename.split(".")[0])
