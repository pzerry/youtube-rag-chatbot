import json
import uuid
import redis

redis_client = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=True
)

SESSION_PREFIX = "session:"


def create_session(video_id: str) -> str:
    session_id = str(uuid.uuid4())

    session = {
        "session_id": session_id,
        "video_id": video_id,
        "history": []
    }

    redis_client.set(
        f"{SESSION_PREFIX}{session_id}",
        json.dumps(session)
    )

    return session_id


def get_session(session_id: str):
    data = redis_client.get(
        f"{SESSION_PREFIX}{session_id}"
    )

    if not data:
        return None

    return json.loads(data)


def add_turn(
    session_id: str,
    question: str,
    answer: str
):
    session = get_session(session_id)

    if not session:
        return

    session["history"].append(
        {
            "question": question,
            "answer": answer
        }
    )

    redis_client.set(
        f"{SESSION_PREFIX}{session_id}",
        json.dumps(session)
    )


def delete_session(session_id: str) -> bool:

    deleted = redis_client.delete(
        f"{SESSION_PREFIX}{session_id}"
    )

    return deleted > 0


def list_sessions():

    sessions = []

    for key in redis_client.scan_iter(
        f"{SESSION_PREFIX}*"
    ):

        session = json.loads(
            redis_client.get(key)
        )

        sessions.append(
            {
                "session_id": session["session_id"],
                "video_id": session["video_id"],
                "turns": len(session["history"])
            }
        )

    return sessions