"""电力领域配置 — 第一个落地的领域"""
from config.domain import DomainConfig


# 电力领域配置：规程检索、造价核算、故障处置、设备对比
class PowerDomainConfig(DomainConfig):
    name = "power"
    label = "电力智能运维"
    description = "电力规程检索、造价核算、故障处置、设备对比分析"

    intent_keywords = {
        "spec_retrieval": ["规程", "规范", "图纸", "标准", "安规", "DL/T", "GB", "设计", "检修", "主变", "变压器"],
        "cost_audit": ["造价", "结算", "审计", "核价", "预算", "定额", "费用", "每公里", "成本"],
        "doc_archive": ["归档", "资料", "验收", "竣工", "存档"],
        "grid_op": ["运维", "巡视", "巡检", "操作", "运行", "台账", "检查"],
        "fault_disposal": ["故障", "跳闸", "异常", "告警", "事故", "抢修", "处理", "雷击"],
        "comparison_analysis": ["对比", "比较", "区别", "关系", "影响", "换成", "假设", "换成会"],
    }

    intent_prompts = {
        "spec_retrieval": "你是电力规程检索助理。基于检索到的规程文档回答，必须引用来源（规程名/章节）。若检索不到依据，明确说'未检索到相关规程'，不要编造。",
        "cost_audit": "你是电力造价核算助理。基于造价数据和定额规则核算，给出费用明细。注意区分新建/改造/大修的定额系数。",
        "grid_op": "你是电网运维助理。基于设备台账和运维规程回答巡检、操作、参数问题。涉及操作步骤时按规程分条列出。",
        "fault_disposal": "你是电力故障处置助理。基于故障记录和处置规程分析原因、给出处置建议。若涉及停电/跳闸等高风险操作，提示需人工确认。",
        "doc_archive": "你是电力资料归档助理。回答验收、竣工、存档相关问题。",
        "comparison_analysis": "你是电力分析助理。基于检索证据做结构化对比/影响/反事实分析，逐维度输出，并明确标注【事实】与【推演】。",
        "chat": "你是友好的电力智能运维助手。闲聊时简短回复，电力问题引导到专业渠道。",
    }

    skill_classes = [
        "agent.skills.power_rag_skill:PowerRAGSkill",
        "agent.skills.quota_match_skill:QuotaMatchSkill",
        "agent.skills.comparison_skill:ComparisonAnalysisSkill",
        "agent.skills.fact_check_skill:FactCheckSkill",
    ]

    demo_docs = [
        {"source": "DL-T572-2026", "title": "变压器检修规程",
         "content": """第一章 总则
1.1 本规程适用于 220kV 及以下变电站变压器检修。
1.2 检修应遵循国家电网 DL/T 572 标准，确保设备安全稳定运行。
第二章 主变检修
2.1 主变压器大修周期为每 5 年一次，小修周期为每年一次。
2.2 检修项目包括绕组检查、绝缘测试、冷却系统维护。
第三章 母线保护
3.1 母线保护装置应定期校验，防止保护拒动。
3.2 母线保护与主变保护应配合整定，确保选择性。"""},
        {"source": "DL-T574-2026", "title": "母线保护检验规程",
         "content": """第一章 适用范围
1.1 本规程适用于 110kV 及以上母线保护装置的定期检验。
第二章 检验项目
2.1 母线保护每年校验一次，大修后必须校验。
2.2 校验内容包括差动保护动作值、母联闭锁逻辑。
第三章 与主变保护配合
3.1 母线保护与主变保护应配合整定，防止越级跳闸。"""},
        {"source": "COST-110kV-2026", "title": "110kV 线路造价标准",
         "content": """第一章 造价标准
1.1 110kV 架空线路造价标准为每公里 85 万元。
1.2 电缆线路造价为每公里 180 万元。
第二章 定额规则
2.1 新建工程使用标准定额，改造工程增加 15% 拆除及措施费。
2.2 大修工程按新建工程 70% 定额执行。"""},
        {"source": "FAULT-LIGHTNING-2026", "title": "雷击跳闸故障处置规程",
         "content": """第一章 故障特征
1.1 雷击跳闸多发生于 6-8 月雷雨季节，线路绝缘薄弱点易受冲击。
第二章 处置流程
2.1 故障后应立即巡线，重点检查绝缘子、避雷器、接地装置。
2.2 检查避雷器动作次数，超过标准应更换。
2.3 恢复送电前须确认故障点已隔离并具备送电条件。"""},
    ]

    # 意图映射到电力 Skill 名
    def intent_to_skill(self, intent: str) -> str:
        mapping = {
            "spec_retrieval": "power_rag",
            "cost_audit": "power_rag",
            "grid_op": "power_rag",
            "fault_disposal": "power_rag",
            "doc_archive": "power_rag",
            "comparison_analysis": "comparison_analysis",
        }
        return mapping.get(intent, "")
