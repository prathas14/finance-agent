import math
from langchain_openai import ChatOpenAI

DISCLAIMER = "\n\n⚠️ *This is for educational purposes only and not financial advice.*"


def months_to_goal(target: float, current: float, monthly: float, annual_return: float = 0.07) -> int:
    """Calculate months needed to reach a savings goal with compound growth."""
    if monthly <= 0:
        return -1
    r = annual_return / 12
    remaining = target - current
    if remaining <= 0:
        return 0
    if r == 0:
        return math.ceil(remaining / monthly)
    months = math.log(1 + (remaining * r) / monthly) / math.log(1 + r)
    return math.ceil(months)


def compound_growth(principal: float, annual_rate: float, years: int, monthly_contribution: float = 0) -> float:
    """Future value with optional monthly contributions."""
    r = annual_rate / 12
    n = years * 12
    fv_principal = principal * (1 + r) ** n
    fv_contributions = monthly_contribution * (((1 + r) ** n - 1) / r) if r > 0 else monthly_contribution * n
    return round(fv_principal + fv_contributions, 2)


def analyze_goals(goals: list[dict]) -> list[dict]:
    results = []
    for g in goals:
        target = g.get("target", 0)
        current = g.get("current", 0)
        monthly = g.get("monthly_contribution", 0)
        months = months_to_goal(target, current, monthly)
        years = months // 12 if months > 0 else 0
        rem_months = months % 12 if months > 0 else 0
        pct_complete = round(current / target * 100, 1) if target > 0 else 0

        results.append({
            "name": g.get("name"),
            "target": target,
            "current": current,
            "monthly_contribution": monthly,
            "percent_complete": pct_complete,
            "months_to_goal": months,
            "time_estimate": f"{years}y {rem_months}m" if months > 0 else "Goal reached!",
        })
    return results


def answer(query: str, goals: list[dict] | None = None, history: list = []) -> str:
    try:
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

        goals_context = ""
        if goals:
            analyzed = analyze_goals(goals)
            goals_context = f"\nUser's current goals analysis:\n{analyzed}\n"

        system = SystemMessage(content=(
            "You are a financial goal planning assistant. Help the user with savings goals, "
            "compound interest calculations, and financial planning.\n"
            f"{goals_context}"
            "Use these formulas when relevant:\n"
            "- Future Value = P(1+r)^n + PMT*((1+r)^n - 1)/r\n"
            "- Months to goal = log(1 + remaining*r/PMT) / log(1+r)  where r = annual_rate/12\n\n"
            "Provide a clear, actionable answer with specific numbers where possible."
        ))

        prior = history[:-1] if history else []
        messages = [system] + prior + [HumanMessage(content=query)]

        result = llm.invoke(messages)
        return result.content + DISCLAIMER
    except Exception as e:
        return f"Goal planning agent error: {e}{DISCLAIMER}"
