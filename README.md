# 📋 Tracellm

**A lightweight LLM logging framework for every agent and provider.**

Tracellm captures every LLM interaction — prompts, responses, tool calls, errors — into a local queryable database via MCP. Works with any LLM provider (OpenAI, Anthropic, Mistral, local models) and any agent framework (LangChain, OpenClaw, and more). No cloud account. No vendor lock-in. Your logs, on your machine.

---

## How it works

```
Your App / Agent
      ↓  MCP (SSE · port 8001)
 tracellm.py  ──write──►  tracellm.db  (DuckDB)
      ↓
tracellm_api.py  ──read──►  http://localhost:8000/docs
```

- **`tracellm.py`** — MCP server. Receives log writes from any agent or app.
- **`tracellm_api.py`** — Read-only query API. Browse and search logs in the browser.
- **`tracellm.db`** — Local DuckDB file. All logs stored on your machine.

---

## Quickstart

### 1. Install dependencies

```bash
pip install mcp[cli] duckdb fastapi uvicorn starlette
```

### 2. Clone the repo

```bash
git clone https://github.com/your-username/tracellm.git
cd tracellm
```

### 3. Start the MCP server

```bash
python tracellm.py
```

You should see:
```
Tracellm MCP server running (SSE) → http://127.0.0.1:8001/sse
```

### 4. Start the query API

Open a second terminal:

```bash
python tracellm_api.py
```

### 5. Connect your app

Add this to any Python app that calls an LLM:

```python
import asyncio
from mcp.client.sse import sse_client
from mcp import ClientSession

TRACELLM_SSE_URL = "http://127.0.0.1:8001/sse"

async def _log(session_id, agent_name, action_type, action_data, tokens=0):
    async with sse_client(TRACELLM_SSE_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            await session.call_tool("log_action", {
                "session_id":  session_id,
                "agent_name":  agent_name,
                "action_type": action_type,
                "action_data": action_data,
                "status":      "success",
                "metadata":    f'{{"tokens_used": {tokens}}}'
            })

def log_to_tracellm(session_id, agent_name, role, content, tokens=0):
    try:
        asyncio.run(_log(session_id, agent_name,
            "user_message" if role == "user" else "llm_response",
            content, tokens))
    except:
        pass  # logging never blocks the main flow
```

---

## Files

| File | Purpose |
|---|---|
| `tracellm.py` | MCP server — handles all log writes (port 8001) |
| `tracellm_api.py` | Read-only query API — browse logs in browser (port 8000) |
| `chat.py` | Example Streamlit chat app using Intel Foundry NPU models |
| `tracellm.db` | Created automatically on first run |

---

## Query your logs

Once logs are flowing, open the interactive browser UI:

```
http://localhost:8000/docs
```

Or hit the endpoints directly:

| Endpoint | What it returns |
|---|---|
| `GET /logs` | All logs, newest first |
| `GET /logs/sessions` | Every session with message counts |
| `GET /logs/session/{id}` | Full conversation replay |
| `GET /logs/summary` | Totals grouped by model and action type |
| `GET /logs/errors` | All error entries |

**Filter by model:**
```
http://localhost:8000/logs?agent_name=phi-4-mini (best)
```

**Replay a conversation:**
```
http://localhost:8000/logs/session/your-session-id-here
```

Or query directly in Python:

```python
import duckdb
con = duckdb.connect("tracellm.db")

# All recent logs
con.execute("SELECT * FROM chat_logs ORDER BY created_at DESC LIMIT 20").df()

# Token usage by model
con.execute("SELECT agent_name, COUNT(*) as calls FROM chat_logs GROUP BY agent_name").df()

# Full conversation replay
con.execute("""
    SELECT action_type, action_data, created_at
    FROM chat_logs WHERE session_id = 'your-session-id'
    ORDER BY created_at
""").df()
```

---

## MCP Tools

Tracellm exposes three MCP tools:

### `log_action`
Log an LLM interaction or agent action.

| Parameter | Required | Description |
|---|---|---|
| `session_id` | ✅ | UUID grouping a conversation or agent run |
| `agent_name` | ✅ | Model name or agent identifier |
| `action_type` | ✅ | `user_message`, `llm_response`, `tool_call`, or `error` |
| `action_data` | ✅ | The content to log |
| `status` | ❌ | `success` or `error` (default: `success`) |
| `metadata` | ❌ | JSON string for extras e.g. `{"tokens_used": 42}` |

### `get_logs`
Retrieve log entries, optionally filtered.

| Parameter | Required | Description |
|---|---|---|
| `session_id` | ❌ | Filter to a specific session |
| `agent_name` | ❌ | Filter to a specific model or agent |
| `limit` | ❌ | Max results (default: 50) |

### `get_summary`
Returns aggregated stats grouped by model and action type. No parameters required.

---

## Connecting different clients

### LangChain agent

```python
from langchain.tools import tool
from mcp.client.sse import sse_client
from mcp import ClientSession
import asyncio

TRACELLM_SSE_URL = "http://127.0.0.1:8001/sse"

@tool
def log_action_tool(session_id: str, action_type: str, action_data: str) -> str:
    """Log an agent action to Tracellm."""
    async def _log():
        async with sse_client(TRACELLM_SSE_URL) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                await session.call_tool("log_action", {
                    "session_id":  session_id,
                    "agent_name":  "langchain-agent",
                    "action_type": action_type,
                    "action_data": action_data,
                })
    try:
        asyncio.run(_log())
        return f"Logged: {action_type}"
    except:
        return "Log skipped"
```

### OpenClaw skill

Create `~/.openclaw/skills/tracellm/SKILL.md`:

```markdown
---
name: tracellm
description: Log agent actions and LLM responses to Tracellm via MCP.
---

You have access to a Tracellm MCP server at http://127.0.0.1:8001/sse.

After every meaningful action, call the log_action tool with:
- session_id: current session ID
- agent_name: your agent name
- action_type: tool_call, llm_response, or error
- action_data: summary of what happened
```

### stdio mode (MCP-native frameworks)

```bash
python tracellm.py --stdio
```

---

## Database schema

All logs are stored in a single `chat_logs` table:

| Column | Type | Description |
|---|---|---|
| `id` | VARCHAR | UUID, unique per log entry |
| `session_id` | VARCHAR | Groups all messages in one conversation |
| `agent_name` | VARCHAR | Model or agent name |
| `action_type` | VARCHAR | user_message, llm_response, tool_call, error |
| `action_data` | VARCHAR | The logged content |
| `status` | VARCHAR | success or error |
| `metadata` | VARCHAR | JSON string for extras (tokens, latency, cost) |
| `created_at` | TIMESTAMP | UTC timestamp |

The `metadata` field accepts any JSON — log token counts, latency, cost, model version, or any custom field without schema changes.

---
