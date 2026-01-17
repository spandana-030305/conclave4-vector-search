from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

client = QdrantClient(
    url="https://<your-cluster>.cloud.qdrant.io",
    api_key="YOUR_API_KEY"
)

def get_user_memory(user_id, query_embedding, limit=5):

    results = client.search(
        collection_name="user_memory_collection",
        query_vector=query_embedding,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="user_id",
                    match=MatchValue(value=user_id)
                )
            ]
        ),
        limit=limit
    )

    return results
