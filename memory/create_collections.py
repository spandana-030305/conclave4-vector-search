from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

client = QdrantClient(
    url="https://51758a36-06ea-44e8-89a5-d6f5a627a3fd.us-east-1-1.aws.cloud.qdrant.io",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.TvQEfRSlRhuHzkugU0l0KI_tfjeqmWmqC5p-3NrA-C8"
)

VECTOR_SIZE = 384  

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

    print("Memory collections created")

if __name__ == "__main__":
    create_memory_collections()
