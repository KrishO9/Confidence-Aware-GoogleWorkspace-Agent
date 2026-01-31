# Confidence-Aware Workspace Agent: An autonomous agent for Google Workspace

An autonomous email assistant for Google Workspace that safely executes real-world actions.
A confidence-aware NLI-based Judge to verify actions before execution. High-confidence actions run automatically, while ambiguous or risky steps are routed through Human-in-the-Loop (HITL) review to prevent hallucinations.

---

## ✨ Features

| Feature | Description |
|--------|-------------|
| **LLM-driven decisions** | No hardcoded routing — the agent chooses tools and strategy autonomously |
| **Confidence-aware execution** | NLI-based Judge scores actions before execution |
| **Human-in-the-Loop (HITL)** | Low-confidence actions require manual approval |
| **Multi-tool execution** | Use several tools in one query (e.g. search emails + create tasks) |
| **Persistent memory** | Keeps conversation history and user patterns |
| **Email search** | Semantic (vector) search over indexed emails + Gmail API fallback |
| **Calendar & tasks** | View events, create events, list and create Google Tasks |
| **Indexing** | One-time or scheduled indexing of recent emails for fast semantic search |

---

## 🏗️ Architecture

- **BaseAgent** (`src/agents/base_agent.py`) — the LLM chooses “use a tool” or “give final answer” each step.
- **AutonomousEmailAgent** (`src/workflows/autonomous_agent.py`) — Registers tools (email RAG, Gmail, calendar, tasks, memory) and runs the agent with conversation memory.
- **Memory** — ChromaDB for email embeddings, in-memory + optional persistence for conversation.
- **Tools** — Email (RAG + Gmail), calendar, tasks, and conversation recall.

To prevent hallucinations and unsafe autonomous actions, the system includes a **confidence-aware execution layer** powered by a **Natural Language Inference (NLI)-based Judge model**.

### How it works

1. **Plan generation**
   - The agent generates a multi-step plan (tool calls + parameters).

2. **NLI-based Judge evaluation**
   - A separate **BART NLI Judge model** evaluates each proposed action.
   - Outputs a **confidence score** for every action.
   - **High-confidence actions** → executed automatically  
   - **Low-confidence / ambiguous actions** → routed to **Human-in-the-Loop review**

4. **Human-in-the-Loop (HITL)**
   - The user is asked to confirm, modify, or reject the action.
   - Prevents incorrect emails, calendar updates, or task creation.

---

## 📋 Prerequisites

- **Python 3.10+**
- **Azure OpenAI** — API key, endpoint, and a chat + embedding deployment
- **Google Workspace** — Gmail, Calendar, and Tasks enabled; OAuth 2.0 desktop credentials

---

## 🚀 Quick Start

### 1. Clone and enter the project

```bash
git clone https://github.com/your-username/EmailAssistant.git
cd EmailAssistant
```

### 2. Virtual environment and dependencies

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configuration

Create a `.env` in the project root (do **not** commit it):

```env
# Azure OpenAI (required)
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002

# Google (optional if using defaults)
GOOGLE_CREDENTIALS_PATH=credentials.json
GOOGLE_TOKEN_PATH=token.json
```

For **Google Workspace**:

1. In [Google Cloud Console](https://console.cloud.google.com/), enable **Gmail API**, **Google Calendar API**, and **Google Tasks API**.
2. Create **OAuth 2.0** credentials (Desktop app), download the JSON, and save it as `credentials.json` in the project root.
3. On first run, the app will open a browser to sign in and create `token.json`.


## 📖 Usage

### Interactive mode

Start the app; then you can type natural-language queries and use these commands:

| Command      | Description                          |
|-------------|--------------------------------------|
| *(any query)* | Ask anything; the agent picks tools  |
| `autoindex`  | Index last 7 days of emails          |
| `status`     | Indexer status                       |
| `stats`      | Memory / vector store stats          |
| `clear`      | Clear conversation history           |
| `cleardb`    | Clear vector DB (requires confirmation) |
| `quit`       | Exit                                 |

Example:

```
You: Find placement emails from last week and create reminder tasks

Agent: [Uses search_emails_rag and create_task as needed, then answers]
🔧 Tools Used: search_emails_rag, create_task
```

### Single query from CLI

```bash
python main_autonomous.py "Find emails about project deadlines"
```

### Programmatic usage (autonomous agent)

```python
import asyncio
from src.workflows.autonomous_agent import AutonomousEmailAgent

async def main():
    agent = AutonomousEmailAgent()
    response = await agent.process_query("Find placement emails and create a task")
    print(response["answer"])
    print(response.get("tool_calls", []))

asyncio.run(main())
```

More examples: [examples/example_usage.py](examples/example_usage.py) (uses the workflow-based `EmailAssistant`).

---

## 📁 Project Structure

```
EmailAssistant/
├── main_autonomous.py    # Entry point — autonomous agent (recommended)
├── main.py               # Entry point — workflow-based assistant
├── run.bat / run.sh      # Quick run scripts
├── requirements.txt
├── src/
│   ├── agents/           # BaseAgent (ReAct)
│   ├── api/              # Azure OpenAI, Gmail, Calendar, Tasks clients
│   ├── config/           # Settings (env-based)
│   ├── memory/           # VectorStore, EmailStorage, MemoryManager
│   ├── services/         # Email indexer
│   ├── tools/            # Email, calendar, task tools + search planner
│   ├── workflows/        # Autonomous agent + email assistant workflow
│   └── utils/            # Logging, text helpers
├── examples/             # Example scripts
└── data/                 # Created at runtime (ChromaDB, etc.)
```

---

## ⚙️ Configuration

Important options (env or `src/config/settings.py`):

| Variable | Description | Default |
|----------|-------------|--------|
| `AZURE_OPENAI_*` | API key, endpoint, chat and embedding deployment | — |
| `CHROMA_PERSIST_DIRECTORY` | ChromaDB storage path | `./data/chroma_db` |
| `MAX_CONVERSATION_HISTORY` | Max messages kept in context | `50` |
| `MAX_ITERATIONS` | Max agent steps per query | `15` |
| `TOP_K_RESULTS` | Max results from semantic search | `10` |

---



