import os
import uuid
import gc
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient

# =========================
# CONFIG
# =========================
IMAGE_DIR = r"D:\conclave4-vector-search\data\data\images"

EMBED_BATCH_SIZE = 8      # CPU-safe for CLIP
QDRANT_BATCH_SIZE = 25    # Free-tier safe

QDRANT_URL = "https://51758a36-06ea-44e8-89a5-d6f5a627a3fd.us-east-1-1.aws.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.TvQEfRSlRhuHzkugU0l0KI_tfjeqmWmqC5p-3NrA-C8"
COLLECTION_NAME = "conclave4_healthcare_image_search"

DEVICE = "cpu"

# =========================
# LOAD CLIP MODEL
# =========================
print("Loading CLIP image model...")
clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
).to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

# =========================
# CONNECT TO QDRANT
# =========================
print("Connecting to Qdrant...")
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

# =========================
# COLLECT IMAGE PATHS
# =========================
image_files = []

for root, _, files in os.walk(IMAGE_DIR):
    for f in files:
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            image_files.append(os.path.join(root, f))

total_images = len(image_files)
print(f"\nFound {total_images} images\n")

# =========================
# MAIN IMAGE PIPELINE
# =========================
uploaded_images = 0

for i in range(0, total_images, EMBED_BATCH_SIZE):
    batch_files = image_files[i:i + EMBED_BATCH_SIZE]

    images = []
    payloads = []

    for img_path in batch_files:
        try:
            img = Image.open(img_path).convert("RGB")
            images.append(img)

            payloads.append({
                "image_name": os.path.basename(img_path),
                "dataset": os.path.basename(os.path.dirname(img_path)),
                "source": "Medical Images",
                "type": "image"
            })

        except Exception as e:
            print(f"Skipping {img_path}: {e}")

    if not images:
        continue

    # =========================
    # IMAGE EMBEDDINGS
    # =========================
    inputs = clip_processor(
        images=images,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        image_embeddings = clip_model.get_image_features(**inputs)

    image_embeddings = image_embeddings.cpu().numpy()

    # =========================
    # UPSERT TO QDRANT
    # =========================
    points = []

    for idx, emb in enumerate(image_embeddings):
        points.append({
            "id": str(uuid.uuid4()),
            "vector": emb.tolist(),
            "payload": payloads[idx]
        })

    for j in range(0, len(points), QDRANT_BATCH_SIZE):
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points[j:j + QDRANT_BATCH_SIZE]
        )

    uploaded_images += len(points)
    print(f"Uploaded {uploaded_images}/{total_images} images")

    # =========================
    # FREE MEMORY
    # =========================
    del images, payloads, points, image_embeddings
    gc.collect()

# =========================
# FINAL VERIFY
# =========================
count = client.count(collection_name=COLLECTION_NAME)
print("\nIMAGE INGESTION COMPLETE")
print(f"Total images uploaded: {uploaded_images}")
print(f"Qdrant reports vectors: {count}")
