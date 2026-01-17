from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

client = QdrantClient(
    url="https://<your-cluster>.cloud.qdrant.io",
    api_key="YOUR_API_KEY"
)

VECTOR_SIZE = 384  # same as SBERT

def create_memory_collections():

    client.recreate_collection(
        collection_name="user_memory_collection",
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE
        )
    )

    client.recreate_collection(
        collection_name="session_memory_collection",
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE
        )
    )

    print("✅ Memory collections created")

if __name__ == "__main__":
    create_memory_collections()
