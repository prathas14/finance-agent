# AI Finance Assistant

Democratizing financial literacy through intelligent conversational AI.

## Stack
- **LLM**: Google Gemini 2.0 Flash
- **Orchestration**: LangGraph (multi-agent state machine)
- **Vector DB**: FAISS (RAG over financial articles)
- **Market Data**: Alpha Vantage API + yfinance fallback
- **UI**: Streamlit (3 tabs: Chat, Portfolio Dashboard, Market Data)

## Setup

### 1. Clone & install
```bash
cd finance-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API keys
```bash
cp .env .env
# Edit .env and add your keys:
# GOOGLE_API_KEY=...
# ALPHA_VANTAGE_API_KEY=...
```

### 3. Build the FAISS knowledge base index
```bash
python knowledge_base/ingest.py
```

### 4. Run the app
```bash
streamlit run ui/app.py
```

### 5. Run tests
```bash
pytest tests/ -v
```

## Project Structure
```
finance-agent/
├── agents/
│   ├── rag_agent.py            # RAG Q&A over financial knowledge base
│   ├── portfolio_agent.py      # Portfolio analysis, net worth, allocation
│   ├── market_data_agent.py    # Alpha Vantage + yfinance market data
│   ├── goal_planning_agent.py  # Savings goals & compound interest
│   └── orchestrator.py         # LangGraph routing state machine
├── knowledge_base/
│   ├── articles/               # Financial education articles (Markdown)
│   ├── glossary.json           # 20 key financial terms
│   ├── sample_portfolio.json   # Sample portfolio for demo
│   └── ingest.py               # Builds FAISS index from articles
├── tools/
│   ├── alpha_vantage.py        # Alpha Vantage API wrapper (30-min cache)
│   └── yfinance_tool.py        # yfinance fallback
├── memory/
│   └── conversation_memory.py  # LangGraph MemorySaver checkpointer
├── ui/
│   └── app.py                  # Streamlit app (Chat / Portfolio / Market tabs)
└── tests/
```

## Agent Routing
```
User Query → Intent Classifier → RAG Agent        (education, definitions)
                               → Portfolio Agent   (holdings, net worth)
                               → Market Agent      (live prices, news)
                               → Goal Planner      (savings, compound interest)
```

## Disclaimer
This tool is for educational purposes only and does not constitute financial advice.
