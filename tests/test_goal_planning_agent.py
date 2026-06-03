import pytest
from agents.goal_planning_agent import months_to_goal, compound_growth, analyze_goals


# ---------------------------------------------------------------------------
# months_to_goal
# ---------------------------------------------------------------------------

def test_months_to_goal_typical():
    months = months_to_goal(target=50000, current=10000, monthly=800, annual_return=0.07)
    assert 45 <= months <= 55


def test_months_to_goal_zero_return():
    # No interest: (50000 - 10000) / 800 = 50 months exactly
    months = months_to_goal(target=50000, current=10000, monthly=800, annual_return=0.0)
    assert months == 50


def test_months_to_goal_already_reached():
    assert months_to_goal(target=1000, current=5000, monthly=200) == 0


def test_months_to_goal_exactly_at_target():
    assert months_to_goal(target=10000, current=10000, monthly=500) == 0


def test_months_to_goal_zero_contribution():
    assert months_to_goal(target=10000, current=0, monthly=0) == -1


def test_months_to_goal_negative_contribution():
    assert months_to_goal(target=10000, current=0, monthly=-100) == -1


def test_months_to_goal_high_return_rate():
    # High return should reduce months vs low return
    months_low = months_to_goal(target=50000, current=0, monthly=500, annual_return=0.03)
    months_high = months_to_goal(target=50000, current=0, monthly=500, annual_return=0.12)
    assert months_high < months_low


def test_months_to_goal_returns_ceiling():
    # Result must be a whole number (math.ceil applied)
    months = months_to_goal(target=10000, current=0, monthly=333, annual_return=0.05)
    assert isinstance(months, int)


# ---------------------------------------------------------------------------
# compound_growth
# ---------------------------------------------------------------------------

def test_compound_growth_principal_only():
    # $10k at 7% for 10 years, no contributions ≈ $19,672
    fv = compound_growth(principal=10000, annual_rate=0.07, years=10, monthly_contribution=0)
    assert 19000 < fv < 20500


def test_compound_growth_with_contributions():
    # Should be greater than principal-only
    fv_no_contrib = compound_growth(10000, 0.07, 10, monthly_contribution=0)
    fv_with_contrib = compound_growth(10000, 0.07, 10, monthly_contribution=200)
    assert fv_with_contrib > fv_no_contrib


def test_compound_growth_zero_rate():
    # No interest: principal + (monthly * years * 12)
    fv = compound_growth(principal=1000, annual_rate=0.0, years=5, monthly_contribution=100)
    assert fv == pytest.approx(1000 + 100 * 60, abs=1.0)


def test_compound_growth_zero_years():
    # No time: future value equals principal
    fv = compound_growth(principal=5000, annual_rate=0.07, years=0, monthly_contribution=200)
    assert fv == pytest.approx(5000, abs=1.0)


def test_compound_growth_returns_rounded():
    fv = compound_growth(10000, 0.07, 10, 100)
    # Should be rounded to 2 decimal places
    assert fv == round(fv, 2)


# ---------------------------------------------------------------------------
# analyze_goals
# ---------------------------------------------------------------------------

def test_analyze_goals_percent_complete():
    goals = [{"name": "House", "target": 50000, "current": 10000, "monthly_contribution": 1000}]
    results = analyze_goals(goals)
    assert results[0]["percent_complete"] == 20.0


def test_analyze_goals_goal_reached():
    goals = [{"name": "Car", "target": 20000, "current": 25000, "monthly_contribution": 500}]
    results = analyze_goals(goals)
    assert results[0]["months_to_goal"] == 0
    assert results[0]["time_estimate"] == "Goal reached!"


def test_analyze_goals_multiple_goals():
    goals = [
        {"name": "Emergency Fund", "target": 10000, "current": 2000, "monthly_contribution": 500},
        {"name": "Retirement",     "target": 1000000, "current": 50000, "monthly_contribution": 1000},
    ]
    results = analyze_goals(goals)
    assert len(results) == 2
    assert results[0]["name"] == "Emergency Fund"
    assert results[1]["name"] == "Retirement"
    assert results[1]["months_to_goal"] > results[0]["months_to_goal"]


def test_analyze_goals_time_estimate_format():
    goals = [{"name": "Vacation", "target": 5000, "current": 0, "monthly_contribution": 200}]
    results = analyze_goals(goals)
    estimate = results[0]["time_estimate"]
    # Format should be "Xy Zm" or "Goal reached!"
    assert "y" in estimate or estimate == "Goal reached!"


def test_analyze_goals_zero_target():
    goals = [{"name": "Bad Goal", "target": 0, "current": 0, "monthly_contribution": 100}]
    results = analyze_goals(goals)
    assert results[0]["percent_complete"] == 0


def test_analyze_goals_preserves_all_fields():
    goals = [{"name": "Test", "target": 10000, "current": 1000, "monthly_contribution": 300}]
    results = analyze_goals(goals)
    r = results[0]
    for key in ("name", "target", "current", "monthly_contribution",
                "percent_complete", "months_to_goal", "time_estimate"):
        assert key in r