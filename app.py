import streamlit as st
import requests
import uuid

API_URL = "http://localhost:8000"

st.set_page_config(page_title="AI Assistant Comparison", layout="wide")
st.title("🤖 AI Assistant Comparison")
st.caption("Compare OpenAI GPT-4.1 vs Open Source Qwen 2.5 — side by side")

# ── Session IDs (one per model, stable for this browser session) ──
if "openai_thread_id" not in st.session_state:
    st.session_state.openai_thread_id = f"openai-{uuid.uuid4().hex[:8]}"
if "oss_thread_id" not in st.session_state:
    st.session_state.oss_thread_id = f"oss-{uuid.uuid4().hex[:8]}"

# ── Chat histories (displayed in UI) ──
if "openai_messages" not in st.session_state:
    st.session_state.openai_messages = []
if "oss_messages" not in st.session_state:
    st.session_state.oss_messages = []


def send_message(query: str, thread_id: str, is_open_source: bool) -> str:
    try:
        res = requests.post(f"{API_URL}/query", json={
            "query": query,
            "thread_id": thread_id,
            "is_open_source": is_open_source
        }, timeout=30)
        return res.json().get("response", "Error: no response")
    except Exception as e:
        return f"❌ Error: {e}"


def clear_session(thread_id: str):
    try:
        requests.delete(f"{API_URL}/session/{thread_id}", timeout=5)
    except Exception:
        pass


# ── Layout: two columns, one per model ──
col1, col2 = st.columns(2)

# ─────────────────────────────────────────
# LEFT: OpenAI
# ─────────────────────────────────────────
with col1:
    st.subheader("🟢 OpenAI — GPT-4.1")

    if st.button("🗑️ Clear", key="clear_openai"):
        clear_session(st.session_state.openai_thread_id)
        st.session_state.openai_thread_id = f"openai-{uuid.uuid4().hex[:8]}"
        st.session_state.openai_messages = []
        st.rerun()

    # Chat display
    chat_container_left = st.container(height=500)
    with chat_container_left:
        for msg in st.session_state.openai_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])


with col2:
    st.subheader("Open Source — Qwen 2.5")

    if st.button("Clear", key="clear_oss"):
        clear_session(st.session_state.oss_thread_id)
        st.session_state.oss_thread_id = f"oss-{uuid.uuid4().hex[:8]}"
        st.session_state.oss_messages = []
        st.rerun()


    chat_container_right = st.container(height=500)
    with chat_container_right:
        for msg in st.session_state.oss_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])



st.divider()
mode = st.radio(
    "Send to:",
    ["Both models", "OpenAI only", "OSS only"],
    horizontal=True
)

if prompt := st.chat_input("Type your message here..."):
    send_to_openai = mode in ("Both models", "OpenAI only")
    send_to_oss    = mode in ("Both models", "OSS only")

    if send_to_openai:
        st.session_state.openai_messages.append({"role": "user", "content": prompt})

    if send_to_oss:
        st.session_state.oss_messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        if send_to_openai:
            openai_reply = send_message(prompt, st.session_state.openai_thread_id, is_open_source=False)
            st.session_state.openai_messages.append({"role": "assistant", "content": openai_reply})

        if send_to_oss:
            oss_reply = send_message(prompt, st.session_state.oss_thread_id, is_open_source=True)
            st.session_state.oss_messages.append({"role": "assistant", "content": oss_reply})

    st.rerun()