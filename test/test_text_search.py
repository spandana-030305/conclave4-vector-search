import requests
from sentence_transformers import SentenceTransformer

# =========================
# CONFIG
# =========================
QDRANT_URL = "https://51758a36-06ea-44e8-89a5-d6f5a627a3fd.us-east-1-1.aws.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.TvQEfRSlRhuHzkugU0l0KI_tfjeqmWmqC5p-3NrA-C8"
COLLECTION_NAME = "conclave4_healthcare_vector_search"

TOP_K = 5

# =========================
# LOAD MODEL
# =========================
print("Loading text embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# QUERY
# =========================
query = "clinical symptoms of COVID-19 infection"
print(f"\nQuery: {query}")

query_embedding = model.encode(query).tolist()

# =========================
# REST SEARCH REQUEST
# =========================
url = f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/search"

headers = {
    "Content-Type": "application/json",
    "api-key": QDRANT_API_KEY
}

payload = {
    "vector": query_embedding,
    "limit": TOP_K,
    "with_payload": True
}

response = requests.post(url, headers=headers, json=payload)
response.raise_for_status()

results = response.json()["result"]

# =========================
# PRINT RESULTS
# =========================
print("\nTop results:\n")

for i, r in enumerate(results, start=1):
    payload = r["payload"]
    print(f"Result {i}")
    print(f"Score   : {r['score']:.4f}")
    print(f"PMC ID  : {payload.get('pmc_id')}")
    print(f"ChunkID : payload.get('chunk_id')")
    print(f"Text    : {payload.get('text', '')[:300]}...")
    print("-" * 50)

