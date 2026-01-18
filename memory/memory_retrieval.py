from qdrant_client.models import Filter, FieldCondition, MatchValue


def get_user_memory(client, user_id, query_embedding, limit=5):
    return client.search(
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
        limit=limit   # ✅ keyword
    )


def get_session_memory(client, user_id, session_id, query_embedding, limit=5):
    return client.search(
        collection_name="session_memory_collection",
        query_vector=query_embedding,
        query_filter=Filter(
            must=[
                FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                FieldCondition(key="session_id", match=MatchValue(value=session_id))
            ]
        ),
        limit=limit   
    )