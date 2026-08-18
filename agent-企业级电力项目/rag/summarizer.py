"""文档汇总(map-reduce)— planner 模式:看 N 篇文档写总结

选文档(按主题)→ 逐篇摘要(map)→ 整体总结(reduce)。
适用"总结/综述/归纳"类任务;特定问题检索仍走 RAG(总结需要全量内容,不能用 RAG 片段)。
"""
import logging
import re
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# 汇总触发动词(从用户输入剥离)
_SUMMARY_VERBS = ["总结", "汇总", "综述", "概述", "归纳", "提炼", "摘要", "概括"]


def extract_topic(user_input: str) -> str:
    """从用户输入剥离"总结/汇总"等动词,提取真正的主题。"""
    topic = user_input
    for v in _SUMMARY_VERBS:
        topic = topic.replace(v, "")
    return topic.strip(" ，。,:：、的给我写份把给我一下请帮我").strip()


def _pick_docs(corpus: List[Dict], topic: str, max_docs: int) -> List[Tuple[str, str]]:
    """按来源归组语料,按主题关键词排序,取前 max_docs 篇,返回 [(source, content)]。"""
    docs: Dict[str, str] = {}
    for rec in corpus:
        source = str(rec.get("source") or rec.get("title") or "unknown")
        content = str(rec.get("content") or "")
        if content:
            docs[source] = docs.get(source, "") + content + "\n"

    kws = [k for k in re.split(r"[\s,，、/]+", topic) if len(k) >= 2]
    scored = [(sum(1 for kw in kws if kw in content or kw in src), src, content)
              for src, content in docs.items()]
    scored.sort(key=lambda x: (-x[0], len(x[2])))   # 主题命中多 + 内容全的排前
    if kws:
        hit = [(src, c) for s, src, c in scored if s > 0][:max_docs]
        if hit:
            return hit
    return [(src, c) for _, src, c in scored[:max_docs]]


async def summarize_docs(topic: str, tenant_id: str = "default",
                         max_docs: int = 5, per_doc_budget: int = 1200) -> str:
    """map-reduce 汇总:选文档 → 逐篇摘要(map)→ 整体总结(reduce)。"""
    from rag.retriever import rag_service
    from llm.adapter import unified_llm

    corpus = rag_service.query(tenant_id=tenant_id, limit=1000, include_content=True)
    if not corpus:
        return "知识库为空,没有可汇总的文档。"

    selected = _pick_docs(corpus, topic, max_docs)
    if not selected:
        return "未找到可汇总的文档。"

    # ── map:逐篇摘要 ──
    summaries = []
    for source, content in selected:
        snippet = content[:per_doc_budget]
        r = await unified_llm.ainvoke("chat", [
            {"role": "system", "content": "你是文档摘要员。用 3-5 句话概括这份文档的核心内容,中文,只依据给出的内容,不编造。"},
            {"role": "user", "content": f"文档《{source}》:\n{snippet}"},
        ])
        summaries.append(f"【{source}】{r.content.strip()}")

    # ── reduce:整体总结(主题含"分档/等级/评分"时附带分级输出)──
    combined = "\n\n".join(summaries)
    grading_instr = ("并给出分档结果:按风险/重要性/健康度把相关项分成若干等级,每档列出涉及文档与依据。"
                     if any(kw in topic for kw in ["分档", "等级", "评分", "归类"]) else "")
    r = await unified_llm.ainvoke("chat", [
        {"role": "system", "content": "你是综述员。把下面多篇文档摘要整合成一篇结构化总结:先给总体结论,再分主题展开,最后列出涉及的文档。"
                                      + grading_instr},
        {"role": "user", "content": f"汇总主题:{topic}\n\n{combined}"},
    ])
    return r.content.strip()
