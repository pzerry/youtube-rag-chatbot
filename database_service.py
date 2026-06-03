"""
Database Service Module
Handles all database operations for videos and messages.
"""

from sqlalchemy.orm import Session
from db_models import Video, Message


# --------------------------------------------------
# VIDEO METADATA
# --------------------------------------------------


def save_video_metadata(
    video_id: str,
    chunk_count: int,
    transcript_length: int,
) -> Video:
    """
    Save or update video metadata in database.
    """
    from database import SessionLocal
    
    db: Session = SessionLocal()

    try:
        # Check if video already exists
        existing = db.query(Video).filter(
            Video.video_id == video_id
        ).first()

        if existing:
            # Update existing
            existing.chunk_count = chunk_count
            existing.transcript_length = transcript_length
            db.commit()
            db.refresh(existing)
            return existing

        # Create new
        video = Video(
            video_id=video_id,
            chunk_count=chunk_count,
            transcript_length=transcript_length,
        )

        db.add(video)
        db.commit()
        db.refresh(video)

        return video

    finally:
        db.close()


def get_video_metadata(
    video_id: str,
) -> dict | None:
    """
    Retrieve video metadata from database.
    """
    from database import SessionLocal
    
    db: Session = SessionLocal()

    try:
        video = db.query(Video).filter(
            Video.video_id == video_id
        ).first()

        if not video:
            return None

        return {
            "video_id": video.video_id,
            "chunk_count": video.chunk_count,
            "transcript_length": video.transcript_length,
            "created_at": video.created_at,
        }

    finally:
        db.close()


# --------------------------------------------------
# MESSAGES
# --------------------------------------------------


def save_message(
    session_id: str,
    question: str,
    answer: str,
) -> Message:
    """
    Save a Q&A pair to the database.
    """
    from database import SessionLocal
    
    db: Session = SessionLocal()

    try:
        message = Message(
            session_id=session_id,
            question=question,
            answer=answer,
        )

        db.add(message)
        db.commit()
        db.refresh(message)

        return message

    finally:
        db.close()


def get_messages_by_session(
    session_id: str,
) -> list[Message]:
    """
    Retrieve all messages for a session.
    """
    from database import SessionLocal
    
    db: Session = SessionLocal()

    try:
        messages = db.query(Message).filter(
            Message.session_id == session_id
        ).order_by(
            Message.created_at
        ).all()

        return messages

    finally:
        db.close()


def delete_messages_by_session(
    session_id: str,
) -> int:
    """
    Delete all messages for a session.
    Returns count of deleted messages.
    """
    from database import SessionLocal
    
    db: Session = SessionLocal()

    try:
        count = db.query(Message).filter(
            Message.session_id == session_id
        ).delete()

        db.commit()

        return count

    finally:
        db.close()
