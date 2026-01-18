from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType

client = QdrantClient(
    url="https://51758a36-06ea-44e8-89a5-d6f5a627a3fd.us-east-1-1.aws.cloud.qdrant.io",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.TvQEfRSlRhuHzkugU0l0KI_tfjeqmWmqC5p-3NrA-C8"
)
# Index for user_id (long-term memory)
client.create_payload_index(
    collection_name="user_memory_collection",
    field_name="user_id",
    field_schema=PayloadSchemaType.KEYWORD
)

# Index for user_id in session memory
client.create_payload_index(
    collection_name="session_memory_collection",
    field_name="user_id",
    field_schema=PayloadSchemaType.KEYWORD
)

# Index for session_id in session memory
client.create_payload_index(
    collection_name="session_memory_collection",
    field_name="session_id",
    field_schema=PayloadSchemaType.KEYWORD
)
print("Payload indexes created successfully")