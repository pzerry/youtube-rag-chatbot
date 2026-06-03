from pydantic import BaseModel
from typing import Optional


class ProcessVideoRequest(BaseModel):
    youtube_url: str  # Full URL or raw video ID


class ProcessVideoResponse(BaseModel):
    session_id: str
    video_id: str
    message: str
    chunk_count: int
    cached: bool


class ChatRequest(BaseModel):
    session_id: str
    question: str


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    question: str


class HistoryItem(BaseModel):
    question: str
    answer: str


class HistoryResponse(BaseModel):
    session_id: str
    video_id: str
    history: list[HistoryItem]


class ErrorResponse(BaseModel):
    detail: str
