from __future__ import annotations

from datetime import datetime, timezone

from app.services.context_builder import ContextBuilder, ContextConfig


def test_context_builder_outputs_gssc_sections_and_budget():
    builder = ContextBuilder(
        ContextConfig(
            max_tokens=220,
            reserve_ratio=0.2,
            min_relevance=0.0,
            enable_compression=True,
        )
    )
    built = builder.build(
        user_query="这个论文用了什么开源数据集？",
        system_instructions="Use citations and do not invent evidence.",
        state_summary="route=rag; tools=paper_rag",
        conversation_history=[],
        tool_results=[
            {
                "tool": "paper_rag",
                "status": "ok",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "content": "[S1] Data availability\nThe paper uses the MATR open-source dataset.",
            },
            {
                "tool": "tavily_research",
                "status": "ok",
                "content": "[R1] Irrelevant research result\nA long unrelated result.",
            },
        ],
        output_instructions="Answer with citations.",
    )

    assert "[Role & Policies]" in built.context
    assert "[Task]" in built.context
    assert "[State]" in built.context
    assert "[Evidence]" in built.context
    assert "[Output]" in built.context
    assert "MATR" in built.context
    assert built.stats["token_count"] <= built.stats["max_tokens"]


def test_context_builder_filters_low_relevance_packets():
    builder = ContextBuilder(ContextConfig(max_tokens=500, min_relevance=0.6))
    built = builder.build(
        user_query="graph neural network",
        system_instructions="Policy",
        tool_results=[
            {"tool": "paper_rag", "status": "ok", "content": "graph neural network encoder"},
            {"tool": "unknown", "status": "ok", "content": "unrelated battery note"},
        ],
    )

    assert "graph neural network" in built.context
    assert "unrelated battery note" not in built.context
