from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

QDRANT_URL = "https://51758a36-06ea-44e8-89a5-d6f5a627a3fd.us-east-1-1.aws.cloud.qdrant.io"
QDRANT_API_KEY = "YOUR_API_KEY"

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

client.collection_exists(
    collection_name="conclave4_healthcare_image_search",
    vectors_config=VectorParams(
        size=512,
        distance=Distance.COSINE
    )
)

print("Image collection created")
