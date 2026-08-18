"""电力 Agent 的 MCP 服务端 — 把能力以 MCP 工具暴露(P4)

供本 agent 与外部 agent 复用:
  - power_retrieve: 混合检索(向量+BM25+重排),返回片段与引用
  - power_kg_query:  知识图谱一跳关系查询
  - power_chat:      完整跑一次电力问答(调 graph.run_agent)

运行: `python -m mcp_servers.power_mcp` 或由 main 挂载。
"""
import json
import logging

from fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP("power-agent")


@mcp.tool()
def power_retrieve(query: str, top_k: int = 5, tenant_id: str = "default") -> str:
    """电力知识检索:混合检索(向量 + BM25 + Reranker),返回文档片段与引用(JSON)。"""
    from rag.retriever import rag_service
    result = rag_service.search(query, top_k=top_k, tenant_id=tenant_id)
    return json.dumps(result.get("results", [])[:5], ensure_ascii=False)


@mcp.tool()
def power_kg_query(entity: str) -> str:
    """知识图谱关系查询:输入设备/规程实体,返回其一跳关系(如 主变压器--[连接到]-->母线)。"""
    from rag.kg.entity_index import entity_index
    ents = entity_index.extract_entities(entity)
    lines = []
    for e in ents:
        for r in entity_index.query(e):
            lines.append(f"{r['subject']} --[{r['relation']}]--> {r['object']}")
    return "\n".join(lines) if lines else "无命中"


@mcp.tool()
async def power_chat(message: str, tenant_id: str = "default") -> str:
    """电力智能问答:让电力 Agent 完整处理一次问答,返回回答。"""
    from agent.graph import run_agent
    result = await run_agent(message, tenant_id=tenant_id, account="mcp")
    if result.get("success"):
        return result.get("reply", "")
    return f"处理失败: {result.get('error', '')}"


@mcp.tool()
async def summarize_docs(topic: str, tenant_id: str = "default") -> str:
    """文档总结(planner 模式):按主题对知识库文档做 map-reduce 汇总,返回结构化总结。"""
    from rag.summarizer import summarize_docs as _summarize
    return await _summarize(topic, tenant_id=tenant_id)


async def list_tool_names() -> list:
    """列出已注册的 MCP 工具名(供调试/文档)。"""
    tools = await mcp.list_tools()
    return sorted(getattr(t, "name", str(t)) for t in tools)


if __name__ == "__main__":
    mcp.run()
