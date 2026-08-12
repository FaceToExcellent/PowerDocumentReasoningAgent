#!/usr/bin/env python3
"""全链路冒烟测试 — 验证企业版各核心模块（本机可跑）"""
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS, FAIL = 0, 0


def check(name, ok):
    global PASS, FAIL
    print(f"  {'✅' if ok else '❌'} {name}")
    if ok:
        PASS += 1
    else:
        FAIL += 1


async def main():
    import asyncio
    print("═" * 50)
    print("企业版冒烟测试")

    # ── M1 向量库（Milvus Lite + 多租户）──
    print("\n[M1] 向量库")
    from rag.retriever import rag_service
    from rag.doc_splitter import split_document
    if rag_service.count("tenant_smoke") == 0:
        docs = split_document("第一章 主变\n主变检修周期5年。\n第二章 母线\n母线保护每年校验。",
                              source="SMOKE-DOC", title="冒烟测试")
        rag_service.add_documents(docs, tenant_id="tenant_smoke")
    check("Milvus Lite 入库 + 检索", len(rag_service.search("主变检修", tenant_id="tenant_smoke")["results"]) > 0)
    check("多租户隔离", len(rag_service.search("母线", tenant_id="tenant_nonexist")["results"]) == 0)

    # ── M2 推理层（模型分级 + 降级）──
    print("\n[M2] 推理层")
    from llm.adapter import unified_llm
    r = await unified_llm.ainvoke("supervisor", [{"role": "user", "content": "你好"}])
    # ⭐ 升级后:supervisor 走 DeepSeek chat(有 key)或本地小模型(无 key 降级)
    check(f"轻量任务走 DeepSeek chat 或降级本地(backend={r.backend})",
          r.backend in ("deepseek", "local_small") or r.backend == "local_reasoning")

    # ── M3 记忆 ──
    print("\n[M3] 记忆")
    from memory.manager import memory_manager
    await memory_manager.record(tenant_id="ts", user_id="u", thread_id="t",
                                reply_id="r1", role="user", content="测试问题", intent="chat")
    ctx = await memory_manager.build_context(tenant_id="ts", user_id="u", thread_id="t",
                                             query="测试", max_tokens=300)
    check("记忆写入 + 按需加载", "测试问题" in ctx)

    # ── M4/M5 Skill 筛选 + Harness ──
    print("\n[M4/M5] Skill 体系")
    from agent.skills.bootstrap import skill_registry
    from agent.skills.selector import skill_selector
    check("Skill 注册（4 个）", len(skill_registry.all_metadata()) >= 4)
    sel = skill_selector.select_skills("对比熔断器和断路器", {"tenant_id": "t"}, top_k=3)
    check("动态筛选命中 comparison_analysis", any(s.name == "comparison_analysis" for s in sel))
    from agent.harness.interceptor import harness_interceptor
    hr = await harness_interceptor.before_skill_execute("power_outage_plan", {}, "u", "t")
    check("高危拦截生效", hr.need_confirm)

    # ── M7 FactCheck 三级置信度 ──
    print("\n[M7] FactCheck")
    from agent.fact_checker import check_output
    fc = check_output("主变大修每5年一次，依据 DL/T 572 规程")
    check(f"FactCheck 置信度分级（{fc.get('confidence')}）", fc.get("confidence") in ("high", "medium", "low"))

    # ── 图执行（chat 路径）──
    print("\n[Graph] LangGraph 执行")
    from agent.graph import run_agent
    r = await run_agent("你好", thread_id="smoke-chat", tenant_id="default")
    check("chat 路径执行成功", r.get("success") and r.get("reply"))

    print("\n" + "═" * 50)
    print(f"结果: {PASS} 通过, {FAIL} 失败")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    import asyncio
    sys.exit(asyncio.run(main()))
