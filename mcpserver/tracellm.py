"""
tracellm.py — Universal LLM logging via MCP
============================================
Run modes:
  python tracellm.py          → SSE mode (port 8001) for chat3.py / OpenClaw
  python tracellm.py --stdio  → stdio mode for Claude Desktop
"""

import asyncio
import sys
import uuid
from datetime import datetime
from typing import Optional

import duckdb
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.server.sse import SseServerTransport
from mcp import types
from starlette.applications import Starlette
from starlette.routing import Route
import uvicorn

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH  = "tracellm.db"
SSE_HOST = "127.0.0.1"
SSE_PORT = 8001

# ── Database ──────────────────────────────────────────────────────────────────
def get_db():
    return duckdb.connect(DB_PATH)

def init_db():
    con = get_db()
    con.execute("""
        CREATE TABLE IF NOT EXISTS chat_logs (
            id          VARCHAR,
            session_id  VARCHAR,
            agent_name  VARCHAR,
            action_type VARCHAR,
            action_data VARCHAR,
            status      VARCHAR,
            metadata    VARCHAR,
            created_at  TIMESTAMP
        )
    """)
    con.close()

init_db()

# ── MCP Server ────────────────────────────────────────────────────────────────
server = Server("tracellm")

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="log_action",
            description=(
                "Log an LLM interaction or agent action to Tracellm. "
                "Call this after every user message, LLM response, tool call, or error."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Unique ID for this conversation or agent run"
                    },
                    "agent_name": {
                        "type": "string",
                        "description": "Model name or agent identifier e.g. 'phi-4-mini', 'my-agent'"
                    },
                    "action_type": {
                        "type": "string",
                        "description": "One of: user_message, llm_response, tool_call, error"
                    },
                    "action_data": {
                        "type": "string",
                        "description": "The content to log — prompt, response, tool output, error message"
                    },
                    "status": {
                        "type": "string",
                        "description": "success or error",
                        "default": "success"
                    },
                    "metadata": {
                        "type": "string",
                        "description": "Optional JSON string for extras e.g. '{\"tokens_used\": 42}'"
                    },
                },
                "required": ["session_id", "agent_name", "action_type", "action_data"]
            }
        ),
        types.Tool(
            name="get_logs",
            description="Retrieve logs from Tracellm. Optionally filter by session or agent.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Filter to a specific session"
                    },
                    "agent_name": {
                        "type": "string",
                        "description": "Filter to a specific agent or model"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max number of results to return",
                        "default": 50
                    },
                },
                "required": []
            }
        ),
        types.Tool(
            name="get_summary",
            description="Get aggregated usage stats grouped by model and action type.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:

    # ── log_action ────────────────────────────────────────────────────────────
    if name == "log_action":
        con = get_db()
        con.execute("""
            INSERT INTO chat_logs VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            str(uuid.uuid4()),
            arguments["session_id"],
            arguments["agent_name"],
            arguments["action_type"],
            arguments["action_data"],
            arguments.get("status", "success"),
            arguments.get("metadata"),
            datetime.utcnow()
        ])
        con.close()
        return [types.TextContent(
            type="text",
            text=f"Logged: {arguments['action_type']} for {arguments['agent_name']} (session: {arguments['session_id']})"
        )]

    # ── get_logs ──────────────────────────────────────────────────────────────
    elif name == "get_logs":
        con = get_db()
        query = "SELECT * FROM chat_logs WHERE 1=1"
        params = []
        if arguments.get("session_id"):
            query += " AND session_id = ?"
            params.append(arguments["session_id"])
        if arguments.get("agent_name"):
            query += " AND agent_name = ?"
            params.append(arguments["agent_name"])
        query += f" ORDER BY created_at DESC LIMIT {arguments.get('limit', 50)}"
        df = con.execute(query, params).fetchdf()
        con.close()
        if df.empty:
            return [types.TextContent(type="text", text="No logs found.")]
        return [types.TextContent(type="text", text=df.to_string(index=False))]

    # ── get_summary ───────────────────────────────────────────────────────────
    elif name == "get_summary":
        con = get_db()
        df = con.execute("""
            SELECT
                agent_name,
                action_type,
                COUNT(*)        AS total,
                MIN(created_at) AS first_seen,
                MAX(created_at) AS last_seen
            FROM chat_logs
            GROUP BY agent_name, action_type
            ORDER BY last_seen DESC
        """).fetchdf()
        con.close()
        if df.empty:
            return [types.TextContent(type="text", text="No data yet.")]
        return [types.TextContent(type="text", text=df.to_string(index=False))]

    return [types.TextContent(type="text", text=f"Unknown tool: {name}")]


# ── Transport: stdio ──────────────────────────────────────────────────────────
async def run_stdio():
    print("Tracellm MCP server running (stdio)", file=sys.stderr)
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


# ── Transport: SSE ────────────────────────────────────────────────────────────
async def run_sse():
    sse = SseServerTransport("/messages")

    async def app(scope, receive, send):
        if scope["type"] == "http":
            path = scope.get("path", "")
            if path == "/sse":
                async with sse.connect_sse(scope, receive, send) as (read, write):
                    await server.run(read, write, server.create_initialization_options())
            elif path == "/messages":
                await sse.handle_post_message(scope, receive, send)
        elif scope["type"] == "lifespan":
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return

    print(f"Tracellm MCP server running (SSE) → http://{SSE_HOST}:{SSE_PORT}/sse", file=sys.stderr)
    config = uvicorn.Config(app, host=SSE_HOST, port=SSE_PORT, log_level="warning")
    await uvicorn.Server(config).serve()


# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if "--stdio" in sys.argv:
        asyncio.run(run_stdio())
    else:
        asyncio.run(run_sse())
