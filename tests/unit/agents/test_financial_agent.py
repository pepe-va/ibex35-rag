"""Unit tests for the financial agent's streaming events."""

from unittest.mock import MagicMock

from langchain_core.messages import AIMessageChunk, ToolMessage

from src.agents.financial_agent import MODEL_NODE, FinancialAgent


def _agent_with_stream(messages) -> FinancialAgent:
    agent = object.__new__(FinancialAgent)
    agent._agent = MagicMock()
    agent._agent.stream.return_value = iter(messages)
    return agent


def test_stream_run_emits_tool_and_token_events():
    tool_call = {"name": "get_stock_price", "id": "c1", "args": "", "index": 0}
    agent = _agent_with_stream(
        [
            (
                AIMessageChunk(content="", tool_call_chunks=[tool_call]),
                {"langgraph_node": MODEL_NODE},
            ),
            (ToolMessage(content="12", tool_call_id="c1"), {"langgraph_node": "tools"}),
            (AIMessageChunk(content="Hola"), {"langgraph_node": MODEL_NODE}),
        ]
    )
    events = list(agent.stream_run("¿Cómo cotiza BBVA?", "thread"))
    assert [e["type"] for e in events] == ["tool_start", "tool_end", "token", "done"]
    assert events[2]["content"] == "Hola"
    assert events[-1]["tools_used"] == ["get_stock_price"]


def test_model_node_is_the_node_create_agent_streams_from():
    from langchain.agents import create_agent
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

    graph = create_agent(model=GenericFakeChatModel(messages=iter([])), tools=[])
    assert MODEL_NODE in graph.get_graph().nodes
