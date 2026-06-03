# AI Finance Assistant

A production-ready multi-agent AI system that democratises financial literacy through conversational AI. Built with LangGraph, OpenAI GPT-4o-mini, FAISS, and Streamlit.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Agent Descriptions](#agent-descriptions)
3. [Setup Instructions](#setup-instructions)
4. [Environment Variables](#environment-variables)
5. [Running the Application](#running-the-application)
6. [API Documentation](#api-documentation)
7. [Usage Examples](#usage-examples)
8. [Testing](#testing)
9. [Troubleshooting](#troubleshooting)
10. [Project Structure](#project-structure)

---

## Architecture Overview

```
User Input (Streamlit UI)
        │
        ▼
┌─────────────────────┐
│    Orchestrator     │  LangGraph StateGraph + MemorySaver
│  (orchestrator.py)  │  Thread-safe, session-persisted
└──────────┬──────────┘
           │
      [classify]  ◄── GPT-4o-mini (temperature=0)
           │
      _route(intent)
           │
   ┌───────┼──────────────┬──────────────┐
   │       │              │              │
 [rag] [portfolio]    [market]        [goal]
   │       │              │              │
   ▼       ▼              ▼              ▼
RAG     Portfolio    Market Data    Goal Planning
Agent    Agent         Agent           Agent
FAISS   Live Prices  Alpha Vantage  Compound Interest
+OpenAI  +yfinance    +yfinance      Math
   │       │              │              │
   └───────┴──────────────┴──────────────┘
                    │
                    ▼
          AIMessage appended to
          LangGraph messages state
                    │
                    ▼
          MemorySaver (per session_id)
          enables multi-turn memory
                    │
                    ▼
          Streamlit Chat UI
```

### Data Flow

1. User sends a query via the Streamlit Chat tab
2. The orchestrator classifies intent: `rag`, `portfolio`, `market`, or `goal`
3. The query routes to the appropriate agent node
4. The agent fetches data, calls GPT-4o-mini with full conversation history, and returns a response
5. The response is appended to LangGraph `messages` state as an `AIMessage`
6. The Streamlit UI displays the response with an educational disclaimer

### Multi-Turn Memory

Each browser session gets a unique `session_id` (UUID). This maps to a LangGraph `thread_id`, enabling `MemorySaver` to accumulate the full message history across turns. Every agent node receives the prior `messages` list, so follow-up queries like *"What about at 8% return?"* resolve correctly without the user repeating context. Memory is in-process only and resets on server restart or when the user clicks **Clear Chat**.

---

## Agent Descriptions

### Orchestrator (`agents/orchestrator.py`)

Routes user queries to the correct specialist agent using a zero-temperature LLM classifier.

**Intent categories:**

| Intent | Trigger phrases | Routes to |
|---|---|---|
| `rag` | "what is", "explain", "how does", definitions | RAG Agent |
| `portfolio` | "my portfolio", "net worth", "allocation", "holdings" | Portfolio Agent |
| `market` | Stock ticker symbols, "price of", "news on" | Market Data Agent |
| `goal` | "how long to save", "compound interest", retirement | Goal Planning Agent |

**Fallback:** Any unrecognised intent defaults to `rag`.

---

### RAG Agent (`agents/rag_agent.py`)

Answers financial education questions using Retrieval-Augmented Generation over a local FAISS knowledge base.

- **Knowledge base:** 3 markdown articles (investing basics, bonds, stocks/ETFs) + 20-term glossary
- **Embeddings:** `text-embedding-3-small` (OpenAI)
- **Retrieval:** Top-4 most semantically similar chunks (cosine similarity via FAISS)
- **LLM:** `gpt-4o-mini` (temperature 0.2)
- **History:** Prior conversation turns injected for follow-up resolution

---

### Portfolio Agent (`agents/portfolio_agent.py`)

Analyses a user-uploaded portfolio JSON, fetches live prices, and produces a plain-English summary.

- **Price fetching:** Alpha Vantage → yfinance fallback → `avg_cost` last resort
- **Calculations:** Net worth, gain/loss per position, asset allocation percentages, goals progress
- **LLM:** `gpt-4o-mini` (temperature 0.2) for narrative summary

**Portfolio JSON schema:**
```json
{
  "owner": "string (optional)",
  "cash": 5000.00,
  "holdings": [
    {"symbol": "AAPL", "shares": 10, "avg_cost": 150.00, "asset_class": "stock"}
  ],
  "liabilities": [
    {"name": "Student Loan", "balance": 15000.00, "interest_rate": 0.045}
  ],
  "goals": [
    {"name": "Emergency Fund", "target": 20000, "current": 5000, "monthly_contribution": 500}
  ]
}
```

**Supported `asset_class` values:** `stock`, `bond_etf`, `index_etf`, `commodity_etf`

---

### Market Data Agent (`agents/market_data_agent.py`)

Fetches live stock quotes, price history, and news. Resolves ticker symbols from natural language, including follow-up references like *"what about its news?"*.

- **Quote source:** Alpha Vantage (30-min TTL cache) → yfinance fallback
- **History source:** yfinance (periods: `1mo`, `3mo`, `6mo`, `1y`)
- **News source:** Alpha Vantage News Sentiment API (up to 5 headlines per ticker)
- **LLM:** `gpt-4o-mini` (temperature 0.1)
- **Limit:** Processes up to 3 tickers per query

---

### Goal Planning Agent (`agents/goal_planning_agent.py`)

Calculates savings timelines and future values using compound interest mathematics.

**Formulas used:**

**Future Value:**
```
FV = P × (1 + r)^n  +  PMT × ((1 + r)^n - 1) / r
where r = annual_rate / 12,  n = years × 12
```

**Months to Goal:**
```
months = ceil( log(1 + (remaining × r) / PMT) / log(1 + r) )
where r = annual_rate / 12,  remaining = target - current
```

**Special return values:** `-1` = impossible (no contribution), `0` = already reached

---

## Setup Instructions

### Prerequisites

- Python 3.11+
- OpenAI API key with active billing
- Alpha Vantage API key (free tier: 25 requests/day, 5/min)

### 1. Clone the repository

```bash
git clone https://github.com/prathas14/finance-agent.git
cd finance-agent
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file at the project root:

```
OPENAI_API_KEY=sk-...
ALPHA_VANTAGE_API_KEY=your_key_here
```

Get your keys from:
- OpenAI: https://platform.openai.com/api-keys
- Alpha Vantage: https://www.alphavantage.co/support/#api-key

### 5. Build the FAISS knowledge base index

One-time setup. Re-run only if you add new articles to `knowledge_base/articles/`.

```bash
python knowledge_base/ingest.py
```

Expected output:
```
Loading documents...
Created 14 chunks. Building FAISS index...
Index saved to knowledge_base/.faiss
```

### 6. Launch the app

```bash
streamlit run ui/app.py
```

Opens at **http://localhost:8501**.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | Used for GPT-4o-mini (all agents) and text-embedding-3-small (RAG) |
| `ALPHA_VANTAGE_API_KEY` | Yes | Live stock quotes and news. Free tier: 25 req/day, 5 req/min |

Both variables must be present in `.env` at project root. `python-dotenv` loads them automatically at startup.

---

## Running the Application

```bash
# Activate environment
source venv/bin/activate

# Start app (opens browser automatically)
streamlit run ui/app.py

# Headless mode (no browser prompt, useful for servers)
streamlit run ui/app.py --server.headless true

# Run tests
pytest tests/ -v

# Rebuild knowledge base index
python knowledge_base/ingest.py
```

---

## API Documentation

### `orchestrator.chat()`

Primary entry point for all conversational queries.

```python
from agents.orchestrator import chat

response: str = chat(
    query="What is a bond?",
    session_id="user-session-123",  # Unique ID for conversation memory
    portfolio=None,                  # Optional portfolio dict
    goals=None                       # Optional list of goal dicts
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | `str` | required | Natural language user query |
| `session_id` | `str` | `"default"` | Thread ID for MemorySaver persistence |
| `portfolio` | `dict \| None` | `None` | Portfolio data. Falls back to sample portfolio if None |
| `goals` | `list \| None` | `None` | List of savings goal dicts |

**Returns:** `str` — Agent response with educational disclaimer appended.

---

### `portfolio_agent.analyze()`

```python
from agents.portfolio_agent import analyze

result: dict = analyze(portfolio=None)
```

**Returns:**
```python
{
    "positions": [
        {
            "symbol": "AAPL",
            "shares": 10,
            "avg_cost": 150.0,
            "current_price": 195.5,
            "market_value": 1955.0,
            "gain_loss": 455.0,
            "gain_loss_pct": 30.33,
            "asset_class": "stock"
        }
    ],
    "total_portfolio_value": 12500.0,
    "total_cost_basis": 10000.0,
    "total_gain_loss": 2500.0,
    "total_liabilities": 17200.0,
    "net_worth": -4700.0,
    "allocation": {"stock": 9500.0, "bond_etf": 1480.0, "cash": 5000.0},
    "allocation_pct": {"stock": 76.0, "bond_etf": 11.84, "cash": 40.0},
    "cash": 5000.0,
    "goals": [...]
}
```

---

### `goal_planning_agent.months_to_goal()`

```python
from agents.goal_planning_agent import months_to_goal

months: int = months_to_goal(
    target=50000,
    current=10000,
    monthly=800,
    annual_return=0.07   # Default 7%
)
# Returns: -1 (no contribution), 0 (reached), or N months
```

---

### `goal_planning_agent.compound_growth()`

```python
from agents.goal_planning_agent import compound_growth

future_value: float = compound_growth(
    principal=10000,
    annual_rate=0.07,
    years=10,
    monthly_contribution=200
)
```

---

### `market_data_agent.get_quote()`

```python
from agents.market_data_agent import get_quote

quote: dict = get_quote("AAPL")
# Returns:
# {
#     "symbol": "AAPL",
#     "price": 195.50,
#     "change": 1.20,
#     "change_pct": "0.62%",
#     "volume": 52000000,
#     "latest_trading_day": "2026-06-03"
# }
```

---

## Usage Examples

### Chat Tab — Test Sequences

**Financial Education (RAG)**
```
What is dollar-cost averaging?
How is that different from lump-sum investing?    ← follow-up uses conversation history
What are the risks of bonds?
```

**Portfolio Analysis**
1. Upload your portfolio JSON or click "Load Sample Portfolio"
2. Ask:
```
Summarise my portfolio
Which holding has the worst performance?          ← no need to re-upload
Should I be concerned about my debt?
What is my asset allocation?
```

**Market Data**
```
What is NVDA trading at?
What's the latest news on it?                     ← "it" resolves to NVDA from context
How has MSFT performed over the last 3 months?
Compare AAPL and TSLA prices
```

**Goal Planning**
```
If I save $600/month at 7% return, how long to reach $50,000?
What if I increase that to $900/month?            ← uses prior goal context
What will $20,000 grow to in 15 years at 8%?
```

---

## Testing

### Running the test suite

```bash
source venv/bin/activate
pytest tests/ -v
```

### Test coverage by module

| Test File | Module | What is Covered |
|---|---|---|
| `test_rag_agent.py` | `agents/rag_agent.py` | Disclaimer injection, retriever mock, exception handling, history parameter |
| `test_portfolio_agent.py` | `agents/portfolio_agent.py`, `agents/goal_planning_agent.py` | Net worth math, gain/loss, goals progress, compound interest |
| `test_market_data_agent.py` | `agents/market_data_agent.py`, `tools/` | AV quote, yfinance fallback, news retrieval |
| `test_goal_planning_agent.py` | `agents/goal_planning_agent.py` | Edge cases: zero contribution, goal already met, high return, negative remaining |
| `test_orchestrator.py` | `agents/orchestrator.py` | All four routing paths, unknown intent fallback, multi-turn state accumulation |

### Manual multi-turn memory test

In the Chat tab, run this sequence:

1. `"What is a bond?"` — RAG agent answers
2. `"What are the risks of investing in one?"` — should reference bonds without re-explaining
3. `"What is AAPL trading at?"` — market agent answers
4. `"What's the recent news on it?"` — "it" should resolve to AAPL

If step 2 or 4 loses context, memory is not working correctly.

---

## Troubleshooting

### `openai.RateLimitError: insufficient_quota`
**Cause:** OpenAI free credits exhausted or billing not enabled.  
**Fix:** Add a payment method at https://platform.openai.com/settings/billing

### `openai.AuthenticationError`
**Cause:** `OPENAI_API_KEY` missing or invalid.  
**Fix:** Check `.env` has `OPENAI_API_KEY=sk-...` with no extra spaces or quotes.

### `RuntimeError: Alpha Vantage rate limit hit`
**Cause:** Free tier limit of 5 req/min or 25 req/day exceeded.  
**Fix:** The app automatically falls back to yfinance. Wait 1 minute for the rate limit to reset. The 30-minute TTL cache reduces repeat requests.

### `No such file or directory: 'knowledge_base/.faiss'`
**Cause:** FAISS index not built yet, or built with a different embeddings model (e.g. after switching from Google to OpenAI).  
**Fix:** Rebuild the index:
```bash
python knowledge_base/ingest.py
```

### Portfolio Dashboard shows no data
**Cause:** Uploaded JSON does not match expected schema.  
**Fix:** Ensure the JSON has `holdings` (list), `cash` (float), and `liabilities` (list). Each holding must have `symbol`, `shares`, `avg_cost`, and `asset_class`.

### Conversation follow-ups lose context
**Cause:** Session was reset (page refresh, server restart, or Clear Chat).  
**Fix:** Memory is in-process only (`MemorySaver`) — it does not persist across restarts. Start a fresh conversation from the beginning.

### Sidebar still shows "Powered by Gemini 2.0 Flash"
**Cause:** Cosmetic — the sidebar caption was not updated when the model was switched to OpenAI.  
**Fix:** Edit `ui/app.py` and update the caption to `"Powered by GPT-4o-mini + LangGraph"`.

---

## Project Structure

```
finance-agent/
├── .env                             # API keys — never commit
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
│
├── agents/
│   ├── orchestrator.py              # LangGraph state machine + chat() entry point
│   ├── rag_agent.py                 # FAISS RAG pipeline
│   ├── portfolio_agent.py           # Portfolio analysis + live price fetching
│   ├── market_data_agent.py         # Stock quotes, history, news
│   └── goal_planning_agent.py       # Compound interest + goal timelines
│
├── knowledge_base/
│   ├── ingest.py                    # One-time FAISS index builder
│   ├── glossary.json                # 20 financial term definitions
│   ├── sample_portfolio.json        # Demo portfolio for testing
│   ├── articles/                    # Markdown knowledge base articles
│   │   ├── investing_basics.md
│   │   ├── bonds_fixed_income.md
│   │   └── stocks_and_etfs.md
│   └── .faiss/                      # Generated FAISS index (not committed)
│
├── tools/
│   ├── alpha_vantage.py             # Alpha Vantage API wrapper (30-min TTL cache)
│   └── yfinance_tool.py             # yfinance fallback wrapper
│
├── memory/
│   └── conversation_memory.py       # Shared MemorySaver checkpointer instance
│
├── ui/
│   └── app.py                       # Streamlit app (Chat, Portfolio, Market tabs)
│
└── tests/
    ├── test_rag_agent.py
    ├── test_portfolio_agent.py
    ├── test_market_data_agent.py
    ├── test_goal_planning_agent.py
    └── test_orchestrator.py
```

---

> This tool is for educational purposes only and does not constitute financial advice.