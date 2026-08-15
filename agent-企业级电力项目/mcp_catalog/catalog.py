"""MCP 工具目录 — 把 Skill 注册中心的能力标准化为 MCP Tool 定义

MCP 解决"能力从哪里来",Skill 解决"本轮怎么执行"。
MCPCatalog 统一列出工具/Resource/Prompt 三类能力,带 read_only/risk_level/resource_uris/prompt_ids。
不替代 Hooks 治理、不替代 Workflow/HITL。
"""
import logging
from typing import Any, Dict, List

# 确保 Skill 已按领域注册(mcp_catalog 单独 import 时也要注册)
from agent.skills import bootstrap  # noqa: F401
from agent.skills.registry import skill_registry

logger = logging.getLogger(__name__)


# MCP Resource:工具边界/高风险说明的资源定义
class MCPResource:
    """MCP Resource:给工具绑定边界说明/高风险说明。"""

    # 初始化资源的 URI/名称/描述
    def __init__(self, uri: str, name: str, description: str):
        self.uri = uri
        self.name = name
        self.description = description

    # 将资源转为字典
    def to_dict(self) -> Dict[str, Any]:
        return {"uri": self.uri, "name": self.name, "description": self.description}


# MCP Prompt:可复用口径的提示定义
class MCPPrompt:
    """MCP Prompt:给工具绑定可复用口径(Observation/转人工)。"""

    # 初始化提示的 URI/名称/内容
    def __init__(self, uri: str, name: str, content: str):
        self.uri = uri
        self.name = name
        self.content = content

    # 将提示转为字典
    def to_dict(self) -> Dict[str, Any]:
        return {"uri": self.uri, "name": self.name, "content": self.content}


# MCP 工具定义:名称/参数/只读/风险及绑定资源
class MCPToolDefinition:
    """MCP 工具定义:名称/描述/参数/只读/风险/绑定资源与 Prompt。"""

    # 初始化工具定义各字段
    def __init__(self, name: str, description: str, required: List[str] = None,
                 parameters_schema: Dict[str, str] = None, read_only: bool = True,
                 risk_level: str = "low", resource_uris: List[str] = None,
                 prompt_ids: List[str] = None):
        self.name = name
        self.description = description
        self.required = required or []
        self.parameters_schema = parameters_schema or {}
        self.read_only = read_only
        self.risk_level = risk_level
        self.resource_uris = resource_uris or []
        self.prompt_ids = prompt_ids or []

    # 将工具定义转为字典
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "description": self.description,
            "required": self.required, "parameters_schema": self.parameters_schema,
            "read_only": self.read_only, "risk_level": self.risk_level,
            "resource_uris": self.resource_uris, "prompt_ids": self.prompt_ids,
        }


# 共享 Resource / Prompt(领域无关)
_OBSERVATION_PROMPT = MCPPrompt(
    uri="prompt://power/tool-observation", name="tool_observation",
    content="把工具返回压缩成事实摘要,保留必要字段,不把内部调试字段暴露给用户。")
_TRANSFER_PROMPT = MCPPrompt(
    uri="prompt://power/high-risk-transfer", name="high_risk_transfer",
    content="高风险操作需人工确认,不能由模型或自由 Agent 直接执行。")


# 统一工具目录:由 Skill 注册中心生成 MCP 定义
class MCPCatalog:
    """统一工具目录:从 Skill 注册中心生成 MCP 工具定义。"""

    # 初始化目录并注册共享资源与 Skill 工具
    def __init__(self):
        self.resources: Dict[str, MCPResource] = {}
        self.prompts: Dict[str, MCPPrompt] = {}
        self.tools: Dict[str, MCPToolDefinition] = {}
        self._register_shared()
        self._build_from_skills()

    # 注册领域无关的共享 Resource 与 Prompt
    def _register_shared(self):
        self.resources["resource://power/tool-boundary"] = MCPResource(
            uri="resource://power/tool-boundary", name="tool_boundary",
            description="工具只返回当前租户/用户权限内的数据,不能编造,不能越权查询。")
        self.prompts[_OBSERVATION_PROMPT.uri] = _OBSERVATION_PROMPT
        self.prompts[_TRANSFER_PROMPT.uri] = _TRANSFER_PROMPT

    # 从 Skill 元数据构建 MCP 工具定义
    def _build_from_skills(self):
        """从 Skill 元数据生成 MCPToolDefinition(带 read_only/risk_level)。"""
        for meta in skill_registry.all_metadata():
            tool = MCPToolDefinition(
                name=meta.name,
                description=meta.description,
                required=getattr(meta, "required_params", []) or [],
                read_only=getattr(meta, "read_only", True),
                risk_level=getattr(meta, "risk_level", None).value if getattr(meta, "risk_level", None) else "low",
                resource_uris=["resource://power/tool-boundary"],
                prompt_ids=[_OBSERVATION_PROMPT.uri],
            )
            self.tools[meta.name] = tool

    # 列出所有工具定义(字典列表)
    def list_tools(self) -> List[Dict[str, Any]]:
        return [t.to_dict() for t in self.tools.values()]

    # 按名称获取单个工具定义
    def get_tool(self, name: str) -> MCPToolDefinition | None:
        return self.tools.get(name)

    # 转为模型可见的工具 JSON 规格
    def to_tool_specs(self) -> List[Dict[str, Any]]:
        """转模型可见的工具 JSON(供 tool use / Skill 选择)。"""
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": {"type": "object",
                                 "properties": {p: {"type": "string"} for p in t.parameters_schema},
                                 "required": t.required},
                "read_only": t.read_only,
                "risk_level": t.risk_level,
            }
            for t in self.tools.values()
        ]

    # 列出所有资源定义
    def list_resources(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self.resources.values()]

    # 列出所有提示定义
    def list_prompts(self) -> List[Dict[str, Any]]:
        return [p.to_dict() for p in self.prompts.values()]

    # 汇总目录中的工具/资源/提示数量
    def binding_summary(self) -> Dict[str, Any]:
        return {
            "tool_count": len(self.tools),
            "resource_count": len(self.resources),
            "prompt_count": len(self.prompts),
            "tool_names": sorted(self.tools.keys()),
        }


mcp_catalog = MCPCatalog()
