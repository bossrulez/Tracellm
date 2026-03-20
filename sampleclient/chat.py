import asyncio
import uuid
import time
import streamlit as st
import requests
from mcp.client.sse import sse_client
from mcp import ClientSession

st.set_page_config(page_title="Local NPU Chat", page_icon="🧠", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
.stApp, .stApp > header { background-color: #0d0d0d !important; }
html, body, [class*="css"], .stMarkdown { font-family: 'IBM Plex Sans', sans-serif !important; color: #e8e8e8 !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; max-width: 820px !important; }
h1 { font-family: 'IBM Plex Mono', monospace !important; font-size: 1.4rem !important; color: #7effd4 !important;
     border-bottom: 1px solid #1e1e1e !important; padding-bottom: 0.4rem !important; margin-bottom: 0.8rem !important; }
.stSelectbox > div > div { background: #161616 !important; border: 1px solid #2a2a2a !important;
    border-radius: 6px !important; color: #e8e8e8 !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 0.82rem !important; }
.stSelectbox label { color: #555 !important; font-size: 0.72rem !important; font-family: 'IBM Plex Mono', monospace !important; }
.stChatMessage { background: #161616 !important; border: 1px solid #1e1e1e !important;
    border-radius: 8px !important; padding: 0.75rem 1rem !important; margin-bottom: 0.4rem !important; }
.stChatInputContainer textarea { font-family: 'IBM Plex Mono', monospace !important; background: #161616 !important;
    border: 1px solid #2a2a2a !important; color: #e8e8e8 !important; border-radius: 6px !important; font-size: 0.88rem !important; }
.badge { display: inline-flex; align-items: center; gap: 5px; font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem; border-radius: 4px; padding: 2px 9px; letter-spacing: 0.04em; white-space: nowrap; }
.badge-npu { background:#0d2a1a; color:#7effd4; border:1px solid #1a5a3a; }
.badge-ctx { background:#1a1a1a; color:#888; border:1px solid #2a2a2a; }
.dot { width:6px; height:6px; border-radius:50%; display:inline-block; animation:pulse 2s infinite; background:#7effd4; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.35} }
.ctx-wrap { display:flex; align-items:center; gap:8px; }
.ctx-bar-bg { width:100px; height:4px; background:#1e1e1e; border-radius:2px; overflow:hidden; }
.ctx-bar-fill { height:100%; border-radius:2px; transition:width .4s ease; }
.ctx-label { font-family:'IBM Plex Mono',monospace; font-size:0.66rem; }
.trim-notice { font-family:'IBM Plex Mono',monospace; font-size:0.68rem; color:#7ecfff;
    background:#0d1a2a; border:1px solid #1a3a5a; border-radius:4px; padding:3px 10px; margin-bottom:0.5rem; display:inline-block; }
.stButton > button { background:transparent !important; border:1px solid #2a2a2a !important; color:#555 !important;
    font-family:'IBM Plex Mono',monospace !important; font-size:0.74rem !important; border-radius:4px !important; padding:0.2rem 0.7rem !important; }
.stButton > button:hover { border-color:#ff6b6b !important; color:#ff6b6b !important; }
</style>
""", unsafe_allow_html=True)

# ── Foundry config ────────────────────────────────────────────────────────────
DEFAULT_PORT    = 50568
SYSTEM_PROMPT   = "You are a helpful, concise AI assistant running locally on an Intel NPU. Answer clearly and stay on topic."
RESERVED_TOKENS = 100

NPU_MODELS = {
    "phi-4-mini (best)": {
        "id":          "phi-4-mini-instruct-openvino-npu:2",
        "max_input":   1344,
        "max_output":  192,
        "description": "Best overall",
    },
    "phi-4-mini-reasoning": {
        "id":          "Phi-4-mini-reasoning-openvino-npu:2",
        "max_input":   3584,
        "max_output":  512,
        "description": "Chain-of-thought",
    },
    "Phi-3-mini": {
        "id":          "Phi-3-mini-4k-instruct-openvino-npu:1",
        "max_input":   1344,
        "max_output":  192,
        "description": "Lightest & fastest",
    },
    "Mistral-7B (NPU)": {
        "id":          "Mistral-7B-Instruct-v0-2-openvino-npu:1",
        "max_input":   1344,
        "max_output":  192,
        "description": "Mistral · NPU",
    },
    "Mistral-7B (CPU)": {
        "id":          "mistralai-Mistral-7B-Instruct-v0-2-generic-cpu:2",
        "max_input":   4096,
        "max_output":  512,
        "description": "Mistral · CPU fallback",
    },
    "DeepSeek-R1-7B": {
        "id":          "DeepSeek-R1-Distill-Qwen-7B-openvino-npu:1",
        "max_input":   3584,
        "max_output":  512,
        "description": "Reasoning model",
    },
    "Qwen2.5-0.5B": {
        "id":          "qwen2.5-0.5b-instruct-openvino-npu:3",
        "max_input":   1344,
        "max_output":  192,
        "description": "Tiny & instant",
    },
}

# ── Tracellm MCP config ───────────────────────────────────────────────────────
TRACELLM_SSE_URL = "http://127.0.0.1:8001/sse"
TRACELLM_ENABLED = True  # set False to disable logging without removing code

async def _mcp_log(session_id, agent_name, action_type, action_data, tokens_used):
    """Call the Tracellm MCP log_action tool over SSE."""
    async with sse_client(TRACELLM_SSE_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            await session.call_tool("log_action", {
                "session_id":  session_id,
                "agent_name":  agent_name,
                "action_type": action_type,
                "action_data": action_data,
                "status":      "success",
                "metadata":    f'{{"tokens_used": {tokens_used}}}'
            })

def log_to_tracellm(session_id, agent_name, role, content, tokens_used):
    """Fire-and-forget MCP log call. Never blocks the chat."""
    if not TRACELLM_ENABLED:
        return
    try:
        asyncio.run(_mcp_log(
            session_id  = session_id,
            agent_name  = agent_name,
            action_type = "user_message" if role == "user" else "llm_response",
            action_data = content,
            tokens_used = tokens_used
        ))
    except:
        pass  # logging is best-effort, chat must never fail because of it

# ── API helpers ───────────────────────────────────────────────────────────────
def check_foundry(url):
    try:
        return requests.get(url, timeout=3).ok
    except:
        return False

def chat(model_id, messages, max_tokens):
    try:
        r = requests.post(CHAT_URL, json={
            "model":      model_id,
            "messages":   messages,
            "max_tokens": max_tokens,
        }, timeout=600)
        if r.ok:
            return r.json()["choices"][0]["message"]["content"], None
        return None, f"HTTP {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return None, str(e)

def est(text): return max(1, len(text) // 4)
def messages_tokens(messages):
    return est(SYSTEM_PROMPT) + sum(est(m["content"]) + 10 for m in messages)
def trim_to_window(messages, max_tokens):
    budget = max_tokens - RESERVED_TOKENS
    trimmed, dropped = list(messages), 0
    while messages_tokens(trimmed) > budget and len(trimmed) >= 2:
        trimmed = trimmed[2:]
        dropped += 1
    return trimmed, dropped

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("# 🧠 Local NPU Chat")

port = st.number_input("Foundry port", value=DEFAULT_PORT, step=1, format="%d",
    help="Run `foundry service start` to see the current port")
FOUNDRY_BASE = f"http://127.0.0.1:{int(port)}"
CHAT_URL     = f"{FOUNDRY_BASE}/v1/chat/completions"
MODELS_URL   = f"{FOUNDRY_BASE}/v1/models"

if not check_foundry(MODELS_URL):
    st.error(f"❌ Foundry service not reachable at port {int(port)}")
    st.info("Run `foundry service start` in PowerShell then refresh.")
    st.stop()

selected = st.selectbox("Model", list(NPU_MODELS.keys()), index=0)
cfg      = NPU_MODELS[selected]
max_ctx  = cfg["max_input"]

# ── Diagnostics ───────────────────────────────────────────────────────────────
with st.expander(f"🔍 Diagnostics — {selected}", expanded=True):
    st.markdown(f"**Model ID:** `{cfg['id']}`")
    st.markdown(f"**Foundry URL:** `{FOUNDRY_BASE}`")
    st.markdown(f"**max_input:** `{cfg['max_input']}` · **max_output:** `{cfg['max_output']}`")

    try:
        t0 = time.time()
        r = requests.get(MODELS_URL, timeout=5)
        ms = round((time.time() - t0) * 1000)
        st.success(f"✅ Step 1 — GET /v1/models → HTTP {r.status_code} ({ms}ms)")
        model_ids = [m["id"] for m in r.json().get("data", [])]
        if cfg["id"] in model_ids:
            st.success(f"✅ Step 2 — Model found in Foundry list")
        else:
            st.error(f"❌ Step 2 — Model NOT in Foundry list")
            st.write("Available:", model_ids)
    except Exception as e:
        st.error(f"❌ Step 1 — Cannot reach Foundry: {e}")
        st.stop()

    for m in r.json().get("data", []):
        if m["id"] == cfg["id"]:
            st.json(m)
            break

    st.markdown("**Step 3 — Sending chat request…**")
    t0 = time.time()
    try:
        r2 = requests.post(CHAT_URL, json={
            "model":      cfg["id"],
            "messages":   [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
        }, timeout=600)
        elapsed = round(time.time() - t0, 1)
        if r2.ok:
            st.success(f"✅ Step 3 — Chat OK → HTTP {r2.status_code} ({elapsed}s)")
        else:
            st.error(f"❌ Step 3 — HTTP {r2.status_code} ({elapsed}s)")
            st.code(r2.text[:800])
    except requests.exceptions.Timeout:
        st.error(f"❌ Step 3 — Timed out after {round(time.time()-t0,1)}s")
    except Exception as e:
        st.error(f"❌ Step 3 — {e}")

# ── Session state ─────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "model_states" not in st.session_state:
    st.session_state.model_states = {}
if selected not in st.session_state.model_states:
    st.session_state.model_states[selected] = {"messages": [], "dropped": 0}

state    = st.session_state.model_states[selected]
messages = state["messages"]

# ── Status bar ────────────────────────────────────────────────────────────────
used = messages_tokens(messages)
pct  = min(used / max_ctx * 100, 100)
bar_color = "#7effd4" if pct < 60 else ("#ffd47e" if pct < 85 else "#ff6b6b")

col_npu, col_ctx, col_desc, col_clr = st.columns([1.2, 2.5, 3.5, 1])
with col_npu:
    st.markdown('<div class="badge badge-npu"><span class="dot"></span>NPU</div>', unsafe_allow_html=True)
with col_ctx:
    st.markdown(f"""<div class="ctx-wrap">
        <div class="ctx-bar-bg"><div class="ctx-bar-fill" style="width:{pct:.1f}%;background:{bar_color};"></div></div>
        <span class="ctx-label" style="color:{bar_color};">{used:,}&nbsp;/&nbsp;{max_ctx:,} tok</span>
    </div>""", unsafe_allow_html=True)
with col_desc:
    st.markdown(f'<div class="badge badge-ctx">{cfg["description"]}</div>', unsafe_allow_html=True)
with col_clr:
    if st.button("Clear"):
        state["messages"] = []
        state["dropped"]  = 0
        st.rerun()

if state["dropped"] > 0:
    st.markdown(f'<div class="trim-notice">↻ {state["dropped"]} oldest pair(s) trimmed</div>', unsafe_allow_html=True)

# ── Chat ──────────────────────────────────────────────────────────────────────
for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input(f"Message {selected}…"):
    messages.append({"role": "user", "content": prompt})

    # ── log user message via Tracellm MCP ──
    log_to_tracellm(st.session_state.session_id, selected, "user", prompt, used)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            windowed, dropped = trim_to_window(messages, max_ctx)
            state["dropped"] += dropped
            if dropped:
                state["messages"] = windowed
                messages = windowed
            api_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + windowed
            response, err = chat(cfg["id"], api_messages, cfg["max_output"])
            if err:
                response = f"⚠️ {err}"
        st.markdown(response)

    # ── log assistant response via Tracellm MCP ──
    log_to_tracellm(st.session_state.session_id, selected, "assistant", response, messages_tokens(messages))

    messages.append({"role": "assistant", "content": response})
    state["messages"] = messages
    st.rerun()
