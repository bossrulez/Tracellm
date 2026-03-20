"""
tracellm_api.py — Read-only REST query layer for Tracellm
==========================================================
All writes go through tracellm.py (MCP).
This server is for querying logs via browser, curl, or Python.

Run: python tracellm_api.py
Docs: http://localhost:8000/docs
"""

import duckdb
import uvicorn
from fastapi import FastAPI, Query
from typing import Optional

app = FastAPI(
    title="Tracellm Query API",
    description="Read-only query layer for Tracellm logs. All writes go through the MCP server.",
    version="1.0.0"
)

DB_PATH = "tracellm.db"

def get_db():
    # Read-only connection — safe to run alongside MCP writer
    return duckdb.connect(DB_PATH, read_only=True)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/logs", summary="Retrieve log entries")
def get_logs(
    session_id: Optional[str] = Query(None, description="Filter by session ID"),
    agent_name: Optional[str] = Query(None, description="Filter by model or agent name"),
    action_type: Optional[str] = Query(None, description="Filter by action type e.g. user_message, llm_response"),
    limit: int = Query(50, description="Max results to return")
):
    con = get_db()
    query = "SELECT * FROM chat_logs WHERE 1=1"
    params = []
    if session_id:
        query += " AND session_id = ?"
        params.append(session_id)
    if agent_name:
        query += " AND agent_name = ?"
        params.append(agent_name)
    if action_type:
        query += " AND action_type = ?"
        params.append(action_type)
    query += f" ORDER BY created_at DESC LIMIT {limit}"
    df = con.execute(query, params).fetchdf()
    con.close()
    return df.to_dict(orient="records")


@app.get("/logs/summary", summary="Usage stats grouped by model and action type")
def get_summary():
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
    return df.to_dict(orient="records")


@app.get("/logs/sessions", summary="List all unique sessions")
def get_sessions():
    con = get_db()
    df = con.execute("""
        SELECT
            session_id,
            agent_name,
            COUNT(*)        AS total_messages,
            MIN(created_at) AS started_at,
            MAX(created_at) AS last_message_at
        FROM chat_logs
        GROUP BY session_id, agent_name
        ORDER BY last_message_at DESC
    """).fetchdf()
    con.close()
    return df.to_dict(orient="records")


@app.get("/logs/session/{session_id}", summary="Replay a full conversation by session ID")
def get_session(session_id: str):
    con = get_db()
    df = con.execute("""
        SELECT action_type, agent_name, action_data, metadata, created_at
        FROM chat_logs
        WHERE session_id = ?
        ORDER BY created_at ASC
    """, [session_id]).fetchdf()
    con.close()
    if df.empty:
        return {"error": f"No logs found for session {session_id}"}
    return df.to_dict(orient="records")


@app.get("/logs/errors", summary="All error entries across all sessions")
def get_errors(limit: int = Query(50)):
    con = get_db()
    df = con.execute("""
        SELECT *
        FROM chat_logs
        WHERE status = 'error'
        ORDER BY created_at DESC
        LIMIT ?
    """, [limit]).fetchdf()
    con.close()
    return df.to_dict(orient="records")


@app.get("/health", summary="Health check")
def health():
    return {"status": "ok", "db": DB_PATH}


# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("tracellm_api:app", host="127.0.0.1", port=8000, reload=True)
