"""
LangGraph orchestrator — routes user queries to the right agent.

Graph:
  START → classify → [rag | portfolio | market | goal] → END
"""
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from memory.conversation_memory import checkpointer


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    query: str
    intent: str
    response: str
    portfolio: dict | None
    goals: list | None


def _classify(state: AgentState) -> AgentState:
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0)
    prompt = f"""Classify this financial query into exactly one category.
Reply with only the category name, nothing else.

Categories:
- rag        (financial education, definitions, "what is", "explain", "how does")
- portfolio  (portfolio analysis, net worth, allocation, holdings, "my portfolio")
- market     (stock price, market data, news, specific ticker symbols)
- goal       (savings goals, compound interest, "how long to save", retirement planning)

Query: {state['query']}
Category:"""

    intent = llm.invoke(prompt).content.strip().lower()
    if intent not in ("rag", "portfolio", "market", "goal"):
        intent = "rag"
    return {**state, "intent": intent}


def _route(state: AgentState) -> str:
    return state["intent"]


def _rag_node(state: AgentState) -> AgentState:
    from agents.rag_agent import answer
    response = answer(state["query"], state.get("messages", []))
    return {**state, "response": response, "messages": [AIMessage(content=response)]}


def _portfolio_node(state: AgentState) -> AgentState:
    from agents.portfolio_agent import summarize
    response = summarize(state.get("portfolio"))
    return {**state, "response": response, "messages": [AIMessage(content=response)]}


def _market_node(state: AgentState) -> AgentState:
    from agents.market_data_agent import answer
    response = answer(state["query"], state.get("messages", []))
    return {**state, "response": response, "messages": [AIMessage(content=response)]}


def _goal_node(state: AgentState) -> AgentState:
    from agents.goal_planning_agent import answer
    response = answer(state["query"], state.get("goals"), state.get("messages", []))
    return {**state, "response": response, "messages": [AIMessage(content=response)]}


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("classify", _classify)
    builder.add_node("rag", _rag_node)
    builder.add_node("portfolio", _portfolio_node)
    builder.add_node("market", _market_node)
    builder.add_node("goal", _goal_node)

    builder.set_entry_point("classify")
    builder.add_conditional_edges("classify", _route, {
        "rag": "rag",
        "portfolio": "portfolio",
        "market": "market",
        "goal": "goal",
    })

    for node in ("rag", "portfolio", "market", "goal"):
        builder.add_edge(node, END)

    return builder.compile(checkpointer=checkpointer)


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def chat(query: str, session_id: str = "default", portfolio: dict | None = None, goals: list | None = None) -> str:
    graph = get_graph()
    config = {"configurable": {"thread_id": session_id}}
    state = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "intent": "",
        "response": "",
        "portfolio": portfolio,
        "goals": goals,
    }
    result = graph.invoke(state, config=config)
    return result["response"]
