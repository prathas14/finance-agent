import json
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# ask_finance
# ---------------------------------------------------------------------------

@patch("agents.orchestrator.chat")
def test_ask_finance_returns_response(mock_chat):
    mock_chat.return_value = "A bond is a fixed-income instrument."
    from mcp_server import ask_finance
    result = ask_finance("What is a bond?", session_id="test-session")
    assert result == "A bond is a fixed-income instrument."
    mock_chat.assert_called_once_with("What is a bond?", session_id="test-session")


@patch("agents.orchestrator.chat")
def test_ask_finance_default_session_id(mock_chat):
    mock_chat.return_value = "Some answer."
    from mcp_server import ask_finance
    ask_finance("Tell me about stocks.")
    mock_chat.assert_called_once_with("Tell me about stocks.", session_id="mcp-default")


# ---------------------------------------------------------------------------
# get_market_data
# ---------------------------------------------------------------------------

@patch("agents.market_data_agent.answer")
def test_get_market_data_returns_response(mock_answer):
    mock_answer.return_value = "AAPL is trading at $195."
    from mcp_server import get_market_data
    result = get_market_data("What is the price of AAPL?")
    assert result == "AAPL is trading at $195."
    mock_answer.assert_called_once_with("What is the price of AAPL?")


@patch("agents.market_data_agent.answer")
def test_get_market_data_passes_query_through(mock_answer):
    mock_answer.return_value = "TSLA and NVDA comparison."
    from mcp_server import get_market_data
    get_market_data("Compare TSLA and NVDA")
    mock_answer.assert_called_once_with("Compare TSLA and NVDA")


# ---------------------------------------------------------------------------
# plan_goal
# ---------------------------------------------------------------------------

@patch("agents.goal_planning_agent.answer")
def test_plan_goal_with_valid_goals_json(mock_answer):
    mock_answer.return_value = "It will take 48 months."
    goals = json.dumps([{"name": "Emergency Fund", "target": 10000, "current": 2000, "monthly_contribution": 500}])
    from mcp_server import plan_goal
    result = plan_goal("How long to reach my goal?", goals=goals)
    assert result == "It will take 48 months."
    args, kwargs = mock_answer.call_args
    assert isinstance(kwargs.get("goals") or args[1], list)


@patch("agents.goal_planning_agent.answer")
def test_plan_goal_with_empty_goals(mock_answer):
    mock_answer.return_value = "Provide more details."
    from mcp_server import plan_goal
    plan_goal("How long to save $50,000?", goals="[]")
    args, kwargs = mock_answer.call_args
    passed_goals = kwargs.get("goals") if "goals" in kwargs else args[1]
    assert passed_goals == []


@patch("agents.goal_planning_agent.answer")
def test_plan_goal_with_malformed_json_falls_back_to_empty(mock_answer):
    mock_answer.return_value = "Some answer."
    from mcp_server import plan_goal
    plan_goal("Save $50,000", goals="not-valid-json")
    args, kwargs = mock_answer.call_args
    passed_goals = kwargs.get("goals") if "goals" in kwargs else args[1]
    assert passed_goals == []


@patch("agents.goal_planning_agent.answer")
def test_plan_goal_default_goals_is_empty(mock_answer):
    mock_answer.return_value = "Answer."
    from mcp_server import plan_goal
    plan_goal("What will $10,000 grow to in 20 years?")
    args, kwargs = mock_answer.call_args
    passed_goals = kwargs.get("goals") if "goals" in kwargs else args[1]
    assert passed_goals == []


# ---------------------------------------------------------------------------
# analyze_portfolio
# ---------------------------------------------------------------------------

@patch("agents.portfolio_agent.summarize")
def test_analyze_portfolio_with_valid_json(mock_summarize):
    mock_summarize.return_value = "Your portfolio is worth $10,000."
    portfolio = json.dumps({
        "holdings": [{"symbol": "AAPL", "shares": 10, "avg_cost": 150.0, "asset_class": "stock"}],
        "cash": 1000.0,
        "liabilities": [],
    })
    from mcp_server import analyze_portfolio
    result = analyze_portfolio(portfolio=portfolio)
    assert result == "Your portfolio is worth $10,000."
    mock_summarize.assert_called_once()
    passed = mock_summarize.call_args[0][0]
    assert isinstance(passed, dict)
    assert "holdings" in passed


@patch("agents.portfolio_agent.summarize")
def test_analyze_portfolio_default_passes_none(mock_summarize):
    mock_summarize.return_value = "Sample portfolio summary."
    from mcp_server import analyze_portfolio
    analyze_portfolio()
    mock_summarize.assert_called_once_with(None)


@patch("agents.portfolio_agent.summarize")
def test_analyze_portfolio_explicit_empty_braces_passes_none(mock_summarize):
    mock_summarize.return_value = "Sample portfolio summary."
    from mcp_server import analyze_portfolio
    analyze_portfolio(portfolio="{}")
    mock_summarize.assert_called_once_with(None)


@patch("agents.portfolio_agent.summarize")
def test_analyze_portfolio_malformed_json_passes_none(mock_summarize):
    mock_summarize.return_value = "Sample portfolio summary."
    from mcp_server import analyze_portfolio
    analyze_portfolio(portfolio="not-valid-json")
    mock_summarize.assert_called_once_with(None)