import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm_mock(content: str):
    mock_result = MagicMock()
    mock_result.content = content
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_result
    return mock_llm


# ---------------------------------------------------------------------------
# Intent classification routing
# ---------------------------------------------------------------------------

@patch("agents.orchestrator.ChatOpenAI")
@patch("agents.rag_agent.ChatOpenAI")
@patch("agents.rag_agent._get_retriever")
def test_routes_to_rag_agent(mock_retriever, mock_rag_llm_cls, mock_orch_llm_cls):
    mock_retriever.return_value.invoke.return_value = []
    mock_orch_llm_cls.return_value = _make_llm_mock("rag")
    mock_rag_llm_cls.return_value = _make_llm_mock("A bond is a fixed-income instrument.")

    from agents.orchestrator import chat
    response = chat("What is a bond?", session_id="test-rag-1")
    assert "bond" in response.lower()


@patch("agents.orchestrator.ChatOpenAI")
@patch("agents.portfolio_agent.ChatOpenAI")
@patch("agents.portfolio_agent._fetch_prices")
def test_routes_to_portfolio_agent(mock_prices, mock_port_llm_cls, mock_orch_llm_cls):
    mock_prices.return_value = {"AAPL": 195.0}
    mock_orch_llm_cls.return_value = _make_llm_mock("portfolio")
    mock_port_llm_cls.return_value = _make_llm_mock("Your portfolio is worth $10,000.")

    portfolio = {
        "cash": 1000.0,
        "holdings": [{"symbol": "AAPL", "shares": 1, "avg_cost": 150.0, "asset_class": "stock"}],
        "liabilities": [],
        "goals": [],
    }

    from agents.orchestrator import chat
    response = chat("Summarise my portfolio", session_id="test-port-1", portfolio=portfolio)
    assert len(response) > 0


@patch("agents.orchestrator.ChatOpenAI")
@patch("agents.market_data_agent.ChatOpenAI")
@patch("agents.market_data_agent.get_quote")
@patch("agents.market_data_agent.get_news")
def test_routes_to_market_agent(mock_news, mock_quote, mock_mkt_llm_cls, mock_orch_llm_cls):
    mock_quote.return_value = {"symbol": "TSLA", "price": 250.0, "change": 1.5,
                               "change_pct": "0.6%", "volume": 10000000, "latest_trading_day": "2026-06-03"}
    mock_news.return_value = []
    mock_orch_llm_cls.return_value = _make_llm_mock("market")
    mock_mkt_llm_cls.return_value = _make_llm_mock("TSLA is trading at $250.")

    from agents.orchestrator import chat
    response = chat("What is TSLA trading at?", session_id="test-mkt-1")
    assert len(response) > 0


@patch("agents.orchestrator.ChatOpenAI")
@patch("agents.goal_planning_agent.ChatOpenAI")
def test_routes_to_goal_agent(mock_goal_llm_cls, mock_orch_llm_cls):
    mock_orch_llm_cls.return_value = _make_llm_mock("goal")
    mock_goal_llm_cls.return_value = _make_llm_mock("It will take about 48 months to reach $50,000.")

    from agents.orchestrator import chat
    response = chat("How long to save $50,000 at $500/month?", session_id="test-goal-1")
    assert len(response) > 0


@patch("agents.orchestrator.ChatOpenAI")
@patch("agents.rag_agent.ChatOpenAI")
@patch("agents.rag_agent._get_retriever")
def test_unknown_intent_falls_back_to_rag(mock_retriever, mock_rag_llm_cls, mock_orch_llm_cls):
    mock_retriever.return_value.invoke.return_value = []
    # Classifier returns something not in the allowed set
    mock_orch_llm_cls.return_value = _make_llm_mock("unknown_category")
    mock_rag_llm_cls.return_value = _make_llm_mock("Here is some general financial guidance.")

    from agents.orchestrator import chat
    response = chat("asdfghjkl", session_id="test-fallback-1")
    assert len(response) > 0


# ---------------------------------------------------------------------------
# Multi-turn state accumulation
# ---------------------------------------------------------------------------

@patch("agents.orchestrator.ChatOpenAI")
@patch("agents.rag_agent.ChatOpenAI")
@patch("agents.rag_agent._get_retriever")
def test_messages_accumulate_across_turns(mock_retriever, mock_rag_llm_cls, mock_orch_llm_cls):
    mock_retriever.return_value.invoke.return_value = []
    mock_orch_llm_cls.return_value = _make_llm_mock("rag")
    mock_rag_llm_cls.return_value = _make_llm_mock("A stock is an ownership share.")

    import uuid
    session = str(uuid.uuid4())

    from agents.orchestrator import get_graph
    graph = get_graph()
    config = {"configurable": {"thread_id": session}}

    # Turn 1
    graph.invoke({
        "messages": [HumanMessage(content="What is a stock?")],
        "query": "What is a stock?", "intent": "", "response": "", "portfolio": None, "goals": None,
    }, config=config)

    # Turn 2 — inspect state
    mock_rag_llm_cls.return_value = _make_llm_mock("Stocks carry market risk.")
    graph.invoke({
        "messages": [HumanMessage(content="What are the risks?")],
        "query": "What are the risks?", "intent": "", "response": "", "portfolio": None, "goals": None,
    }, config=config)

    state = graph.get_state(config)
    messages = state.values.get("messages", [])

    human_messages = [m for m in messages if isinstance(m, HumanMessage)]
    ai_messages = [m for m in messages if isinstance(m, AIMessage)]

    assert len(human_messages) >= 2
    assert len(ai_messages) >= 2


@patch("agents.orchestrator.ChatOpenAI")
@patch("agents.rag_agent.ChatOpenAI")
@patch("agents.rag_agent._get_retriever")
def test_history_passed_to_rag_agent(mock_retriever, mock_rag_llm_cls, mock_orch_llm_cls):
    mock_retriever.return_value.invoke.return_value = []
    mock_orch_llm_cls.return_value = _make_llm_mock("rag")

    captured_messages = []

    def capture_invoke(messages):
        captured_messages.extend(messages)
        result = MagicMock()
        result.content = "Follow-up answer."
        return result

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = capture_invoke
    mock_rag_llm_cls.return_value = mock_llm

    import uuid
    session = str(uuid.uuid4())

    from agents.orchestrator import chat
    chat("What is a bond?", session_id=session)
    chat("What are the risks of one?", session_id=session)

    # Second call should have passed at least a system message + prior context
    assert len(captured_messages) > 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

@patch("agents.orchestrator.ChatOpenAI")
@patch("agents.rag_agent.ChatOpenAI")
@patch("agents.rag_agent._get_retriever")
def test_empty_query_does_not_crash(mock_retriever, mock_rag_llm_cls, mock_orch_llm_cls):
    mock_retriever.return_value.invoke.return_value = []
    mock_orch_llm_cls.return_value = _make_llm_mock("rag")
    mock_rag_llm_cls.return_value = _make_llm_mock("Please provide a question.")

    from agents.orchestrator import chat
    response = chat("", session_id="test-empty-1")
    assert isinstance(response, str)


@patch("agents.orchestrator.ChatOpenAI")
@patch("agents.rag_agent.ChatOpenAI")
@patch("agents.rag_agent._get_retriever")
def test_malformed_query_does_not_crash(mock_retriever, mock_rag_llm_cls, mock_orch_llm_cls):
    mock_retriever.return_value.invoke.return_value = []
    mock_orch_llm_cls.return_value = _make_llm_mock("rag")
    mock_rag_llm_cls.return_value = _make_llm_mock("I could not understand that query.")

    from agents.orchestrator import chat
    response = chat("!@#$%^&*()", session_id="test-malformed-1")
    assert isinstance(response, str)


@patch("agents.orchestrator.ChatOpenAI")
@patch("agents.rag_agent.ChatOpenAI")
@patch("agents.rag_agent._get_retriever")
def test_different_sessions_are_independent(mock_retriever, mock_rag_llm_cls, mock_orch_llm_cls):
    mock_retriever.return_value.invoke.return_value = []
    mock_orch_llm_cls.return_value = _make_llm_mock("rag")
    mock_rag_llm_cls.return_value = _make_llm_mock("Answer for session A.")

    from agents.orchestrator import get_graph
    graph = get_graph()

    config_a = {"configurable": {"thread_id": "session-A"}}
    config_b = {"configurable": {"thread_id": "session-B"}}

    graph.invoke({
        "messages": [HumanMessage(content="Question from A")],
        "query": "Question from A", "intent": "", "response": "", "portfolio": None, "goals": None,
    }, config=config_a)

    state_b = graph.get_state(config_b)
    messages_b = state_b.values.get("messages", [])
    # Session B should have no messages from session A
    assert not any("Question from A" in getattr(m, "content", "") for m in messages_b)