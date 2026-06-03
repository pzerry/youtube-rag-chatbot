from sqlalchemy import (
    Column,
    String,
    Integer,
    Text,
    DateTime
)
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()


class Video(Base):

    __tablename__ = "videos"

    video_id = Column(
        String,
        primary_key=True
    )

    chunk_count = Column(Integer)

    transcript_length = Column(Integer)

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )


class Message(Base):

    __tablename__ = "messages"

    id = Column(
        Integer,
        primary_key=True
    )

    session_id = Column(String)

    question = Column(Text)

    answer = Column(Text)

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )