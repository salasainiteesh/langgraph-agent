import streamlit as st
import requests
import os

# ---- Config ----
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# ---- Page Setup ----
st.set_page_config(
    page_title="LangGraph AI Agent",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 LangGraph AI Agent")
st.caption("Powered by LangGraph + Ollama + FastAPI")

# ---- Session State ----
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "user-1"

# ---- Sidebar ----
with st.sidebar:
    st.header("⚙️ Settings")
    thread_id = st.text_input("Thread ID", value=st.session_state.thread_id)
    st.session_state.thread_id = thread_id
    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.caption("Backend: " + BACKEND_URL)

# ---- Chat History ----
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---- Chat Input ----
if prompt := st.chat_input("Ask me anything..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={
                        "message": prompt,
                        "thread_id": st.session_state.thread_id
                    },
                    timeout=120
                )
                if response.status_code == 200:
                    answer = response.json()["response"]
                else:
                    answer = f"Error: {response.status_code}"
            except Exception as e:
                answer = f"Could not connect to backend: {str(e)}"

            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})