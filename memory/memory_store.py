from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from uuid import uuid4
import time

client = QdrantClient(
    url="https://<your-cluster>.cloud.qdrant.io",
    api_key="YOUR_API_KEY"
)

def store_user_memory(user_id, embedding, content):

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
