from qdrant_client.models import PointStruct
from uuid import uuid4
import time


def store_user_memory(client, user_id, embedding, content):
    """
    Store long-term user memory in Qdrant.
    """

    point = PointStruct(
        id=str(uuid4()),
        vector=embedding,
        payload={
            "user_id": user_id,
            "memory_type": "long_term",
            "content": content,
            "timestamp": time.time(),
            "confidence": 0.8
        }
    )

    client.upsert(
        collection_name="user_memory_collection",
        points=[point]
    )


def store_session_memory(client, user_id, session_id, embedding, content):
    """
    Store session-level memory in Qdrant.
    """

    point = PointStruct(
        id=str(uuid4()),
        vector=embedding,
        payload={
            "user_id": user_id,
            "session_id": session_id,
            "memory_type": "session",
            "content": content,
            "timestamp": time.time()
        }
    )

    client.upsert(
        collection_name="session_memory_collection",
        points=[point]
    )