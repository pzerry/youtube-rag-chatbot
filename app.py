import streamlit as st
import requests

API_URL = "http://localhost:8000"  # Point to FastAPI backend

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="YouTube RAG Chat",
    page_icon="🎥",
    layout="wide",
)

st.title("🎥 YouTube RAG Chat")
st.caption("Ask questions about any YouTube video using AI")

# ─────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "video_id" not in st.session_state:
    st.session_state.video_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "video_processed" not in st.session_state:
    st.session_state.video_processed = False

# ─────────────────────────────────────────────
# SIDEBAR — Video Input
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("📹 Load a Video")
    youtube_url = st.text_input(
        "YouTube URL or Video ID",
        placeholder="https://youtube.com/watch?v=... or JMUxmLyrhSk",
    )

    process_btn = st.button("🚀 Process Video", type="primary", use_container_width=True)

    if process_btn and youtube_url:
        with st.spinner("Fetching transcript and building index..."):
            try:
                resp = requests.post(
                    f"{API_URL}/process-video",
                    json={"youtube_url": youtube_url},
                    timeout=120,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state.session_id = data["session_id"]
                    st.session_state.video_id = data["video_id"]
                    st.session_state.messages = []  # Reset chat on new video
                    st.session_state.video_processed = True

                    cached_tag = "⚡ Loaded from cache" if data["cached"] else "✅ Freshly indexed"
                    st.success(f"{cached_tag}\n\n**Video ID:** `{data['video_id']}`\n\n**Chunks:** {data['chunk_count']}")
                else:
                    try:
                        error_detail = resp.json().get('detail', 'Unknown error')
                    except:
                        error_detail = resp.text or "Unknown error"
                    st.error(f"Error: {error_detail}")
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API. Is the FastAPI server running?\n\nStart it with: `uvicorn main:app --reload`")

    # Show current session info
    if st.session_state.video_processed:
        st.divider()
        st.markdown(f"**Active Video:** `{st.session_state.video_id}`")
        st.markdown(f"**Session:** `{st.session_state.session_id[:8]}...`")
        st.markdown(f"**Messages:** {len(st.session_state.messages)}")

        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        if st.button("❌ End Session", use_container_width=True):
            requests.delete(f"{API_URL}/session/{st.session_state.session_id}")
            st.session_state.session_id = None
            st.session_state.video_id = None
            st.session_state.messages = []
            st.session_state.video_processed = False
            st.rerun()

# ─────────────────────────────────────────────
# MAIN AREA — Chat Interface
# ─────────────────────────────────────────────
if not st.session_state.video_processed:
    st.info("👈 Enter a YouTube URL in the sidebar to get started.")
    st.markdown("""
    ### How it works
    1. **Paste** a YouTube URL in the sidebar
    2. **Click Process** — the transcript is fetched and embedded
    3. **Ask questions** about the video content
    4. The AI answers using only the transcript (no hallucinations)
    
    ### Features
    - 🧠 **Conversation memory** — Ask follow-up questions naturally
    - ⚡ **Caching** — Same video loads instantly the second time
    - 🎯 **Grounded answers** — Only answers from the transcript
    """)
else:
    # Render existing messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if question := st.chat_input("Ask anything about the video..."):
        # Add user message to UI immediately
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Call API
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    resp = requests.post(
                        f"{API_URL}/chat",
                        json={
                            "session_id": st.session_state.session_id,
                            "question": question,
                        },
                        timeout=60,
                    )
                    if resp.status_code == 200:
                        answer = resp.json()["answer"]
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    else:
                        err = resp.json().get("detail", "Unknown error")
                        st.error(f"API Error: {err}")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Lost connection to API server.")
