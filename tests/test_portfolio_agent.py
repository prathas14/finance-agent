import pytest
from agents.portfolio_agent import analyze
from agents.goal_planning_agent import months_to_goal, compound_growth, analyze_goals


SAMPLE_PORTFOLIO = {
    "cash": 1000.0,
    "holdings": [
        {"symbol": "AAPL", "shares": 2, "avg_cost": 150.0, "asset_class": "stock"},
    ],
    "liabilities": [{"name": "Loan", "balance": 500.0, "interest_rate": 0.05}],
    "goals": [{"name": "Emergency Fund", "target": 10000, "current": 2000, "monthly_contribution": 500}],
}


def test_analyze_returns_required_keys():
    data = analyze(SAMPLE_PORTFOLIO)
    assert "total_portfolio_value" in data
    assert "net_worth" in data
    assert "allocation_pct" in data
    assert "positions" in data
    assert len(data["positions"]) == 1


def test_net_worth_calculation():
    data = analyze(SAMPLE_PORTFOLIO)
    assert data["net_worth"] == data["total_portfolio_value"] - 500.0


def test_months_to_goal_basic():
    months = months_to_goal(target=12000, current=2000, monthly=500, annual_return=0.07)
    assert 18 <= months <= 24


def test_months_to_goal_already_reached():
    assert months_to_goal(target=1000, current=2000, monthly=500) == 0


def test_months_to_goal_zero_contribution():
    assert months_to_goal(target=10000, current=0, monthly=0) == -1


def test_compound_growth():
    fv = compound_growth(principal=10000, annual_rate=0.07, years=10, monthly_contribution=0)
    assert 19000 < fv < 20500


def test_analyze_goals():
    goals = [{"name": "House", "target": 50000, "current": 10000, "monthly_contribution": 1000}]
    results = analyze_goals(goals)
    assert len(results) == 1
    assert results[0]["percent_complete"] == 20.0
    assert results[0]["months_to_goal"] > 0
