import requests
import torch
from transformers import CLIPProcessor, CLIPModel

# =========================
# CONFIG
# =========================
QDRANT_URL = "https://51758a36-06ea-44e8-89a5-d6f5a627a3fd.us-east-1-1.aws.cloud.qdrant.io"
QDRANT_API_KEY = "YOUR_API_KEY"

IMAGE_COLLECTION = "conclave4_healthcare_image_search"

TOP_K = 5
DEVICE = "cpu"

# =========================
# LOAD CLIP MODEL
# =========================
print("Loading CLIP text encoder...")
clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
).to(DEVICE)

clip_processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

# =========================
# QUERY
# =========================
query = "a medical x-ray image of the lungs showing infection"
print(f"\nQuery: {query}")

inputs = clip_processor(
    text=[query],
    return_tensors="pt",
    padding=True
)

with torch.no_grad():
    text_embedding = clip_model.get_text_features(**inputs)

query_vector = text_embedding.cpu().numpy()[0].tolist()

# =========================
# QDRANT SEARCH (REST)
# =========================
search_url = f"{QDRANT_URL}/collections/{IMAGE_COLLECTION}/points/search"

headers = {
    "Content-Type": "application/json",
    "api-key": QDRANT_API_KEY
}

payload = {
    "vector": query_vector,
    "limit": TOP_K,
    "with_payload": True
}

response = requests.post(
    search_url,
    headers=headers,
    json=payload
)
response.raise_for_status()

results = response.json()["result"]

# =========================
# PRINT RESULTS
# =========================
print("\nTop image results:\n")

for i, r in enumerate(results, start=1):
    payload = r["payload"]

    print(f"Result {i}")
    print(f"Score      : {r['score']:.4f}")
    print(f"Image Name : {payload.get('image_name')}")
    print(f"Dataset    : {payload.get('dataset')}")
    print(f"Source     : {payload.get('source')}")
    print("-" * 50)
