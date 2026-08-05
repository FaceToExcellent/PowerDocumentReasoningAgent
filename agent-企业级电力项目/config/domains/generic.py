"""通用领域配置 — 第二领域演示：文档问答/对比/总结，证明底座领域无关"""
from config.domain import DomainConfig


class GenericDomainConfig(DomainConfig):
    name = "generic"
    label = "通用文档分析"
    description = "任意文档的问答、对比、总结分析"

    intent_keywords = {
        "doc_qa": ["文档", "规定", "说明", "流程", "制度", "手册", "要求", "内容", "关于", "是什么"],
        "doc_compare": ["对比", "比较", "区别", "差异", "关系", "哪个好", "优劣"],
        "doc_summary": ["总结", "概括", "要点", "摘要", "归纳", "提炼"],
        "chat": ["你好", "hi", "hello", "谢谢", "在吗"],
    }

    intent_prompts = {
        "doc_qa": "你是通用文档问答助理。基于检索到的文档内容回答，引用文档出处。检索不到就说'未检索到相关内容'，不要编造。",
        "doc_compare": "你是通用分析助理。基于检索证据做结构化对比，逐维度输出，并标注【事实】与【推演】。",
        "doc_summary": "你是通用文档总结助理。基于检索内容提炼要点，分条列出，标注依据章节。",
        "chat": "你是友好的通用文档分析助手。闲聊时简短回复，文档问题引导到检索。",
    }

    skill_classes = [
        "agent.skills.generic_skills:DocQASkill",
        "agent.skills.generic_skills:DocCompareSkill",
        "agent.skills.generic_skills:DocSummarySkill",
    ]

    demo_docs = [
        {"source": "COMPANY-POLICY-2026", "title": "公司差旅报销制度",
         "content": """第一章 差旅标准
1.1 员工出差住宿标准：一线城市每天 500 元，二三线城市每天 350 元。
1.2 交通标准：高铁二等座，飞机经济舱，特殊情况需审批。
第二章 报销流程
2.1 出差结束后 5 个工作日内提交报销申请。
2.2 报销需附发票、行程单，部门负责人审批后财务审核。
2.3 虚报费用按公司制度严肃处理。"""},
        {"source": "HR-LEAVE-2026", "title": "员工请假制度",
         "content": """第一章 请假类型
1.1 年假：工作满一年享 5 天，满五年享 10 天。
1.2 病假：需提供医院证明，按国家规定发放工资。
1.3 事假：全年累计不超过 15 天。
第二章 审批流程
2.1 请假 3 天内部门负责人审批。
2.2 请假 3 天以上需部门负责人 + 人事部审批。
2.3 紧急情况可事后补办，需说明原因。"""},
        {"source": "SEC-ACCESS-2026", "title": "机房出入管理规定",
         "content": """第一章 访问权限
1.1 机房出入需提前申请，凭工牌 + 审批单进入。
1.2 核心机柜区域仅限运维人员进入，其他人员需陪同。
第二章 安全管理
2.1 进入机房禁止携带电子设备（手机、相机）。
2.2 操作服务器需双人复核，避免误操作。
2.3 发现异常立即上报值班工程师。"""},
    ]

    def intent_to_skill(self, intent: str) -> str:
        mapping = {
            "doc_qa": "doc_qa",
            "doc_compare": "doc_compare",
            "doc_summary": "doc_summary",
        }
        return mapping.get(intent, "")
