from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from models import (
    ProcessVideoRequest, ProcessVideoResponse,
    ChatRequest, ChatResponse,
    HistoryResponse, HistoryItem,
)
import session_store
from rag import extract_video_id, load_or_build_vectorstore, answer_question
from database_service import (
    save_video_metadata,
    save_message,
    get_messages_by_session
)

app = FastAPI(
    title="YouTube RAG API",
    description="Chat with any YouTube video using RAG + LangChain + Groq",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "message": "YouTube RAG API is running"}


# ─────────────────────────────────────────────
# PROCESS VIDEO → Create session
# ─────────────────────────────────────────────
@app.post("/process-video", response_model=ProcessVideoResponse)
def process_video(req: ProcessVideoRequest):
    """
    Accept a YouTube URL, fetch + embed transcript, return a session_id.
    Subsequent /chat calls use this session_id.
    """
    try:
        video_id = extract_video_id(req.youtube_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        vectorstore, metadata = load_or_build_vectorstore(video_id)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    was_cached = metadata.get("cached", False) or _already_indexed(video_id)
    session_id = session_store.create_session(video_id)

    save_video_metadata(
        video_id=video_id,
        chunk_count=metadata.get("chunk_count", 0),
        transcript_length=metadata.get("transcript_length", 0)
    )

    return ProcessVideoResponse(
        session_id=session_id,
        video_id=video_id,
        message="Video processed successfully. Start chatting!",
        chunk_count=metadata.get("chunk_count", 0),
        cached=was_cached,
    )


def _already_indexed(video_id: str) -> bool:
    import os
    return os.path.exists(f"cache/faiss_indexes/{video_id}")


# ─────────────────────────────────────────────
# CHAT
# ─────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Send a question for an active session. Returns the answer.
    Conversation history is maintained automatically per session.
    """
    session = session_store.get_session(req.session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Please process a video first via /process-video.",
        )

    video_id = session["video_id"]
    vectorstore, _ = load_or_build_vectorstore(video_id)

    try:
        answer = answer_question(vectorstore, req.question, session["history"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")

    session_store.add_turn(req.session_id, req.question, answer)

    save_message(
        session_id=req.session_id,
        question=req.question,
        answer=answer
    )

    return ChatResponse(
        session_id=req.session_id,
        question=req.question,
        answer=answer,
    )


# ─────────────────────────────────────────────
# HISTORY
# ─────────────────────────────────────────────
@app.get("/history/{session_id}", response_model=HistoryResponse)
def get_history(session_id: str):
    """Return full conversation history for a session."""
    session = session_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    messages = get_messages_by_session(session_id)

    return HistoryResponse(
        session_id=session_id,
        video_id=session["video_id"],
        history=[
            HistoryItem(question=m.question, answer=m.answer)
            for m in messages
        ],
    )


# ─────────────────────────────────────────────
# CLEAR SESSION
# ─────────────────────────────────────────────
@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    """Delete a session and free its memory."""
    deleted = session_store.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"message": f"Session {session_id} deleted."}


# ─────────────────────────────────────────────
# LIST SESSIONS (dev/debug)
# ─────────────────────────────────────────────
@app.get("/sessions")
def list_sessions():
    return {"sessions": session_store.list_sessions()}
