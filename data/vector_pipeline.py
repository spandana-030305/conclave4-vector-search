import os
import tarfile
from io import BytesIO
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import uuid
import gc

# =========================
# CONFIG
# =========================
TAR_DIR = r"D:\conclave4-vector-search\data\tar"

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

EMBED_BATCH_SIZE = 32
QDRANT_BATCH_SIZE = 50

QDRANT_URL = "https://51758a36-06ea-44e8-89a5-d6f5a627a3fd.us-east-1-1.aws.cloud.qdrant.io"
QDRANT_API_KEY = "YOUR_API_KEY"
COLLECTION_NAME = "conclave4_healthcare_vector_search"

# =========================
# LOAD MODEL
# =========================
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# CONNECT TO QDRANT
# =========================
print("Connecting to Qdrant...")
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

print("Recreating collection (cloud-only)...")
client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)

# =========================
# HELPERS
# =========================
def extract_clean_text(nxml_bytes):
    soup = BeautifulSoup(nxml_bytes, "lxml-xml")
    parts = []

    if soup.find("article-title"):
        parts.append(soup.find("article-title").get_text(" ", strip=True))
    if soup.find("abstract"):
        parts.append(soup.find("abstract").get_text(" ", strip=True))
    if soup.find("body"):
        parts.append(soup.find("body").get_text(" ", strip=True))

    return "\n\n".join(parts)


def chunk_text(text):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + CHUNK_SIZE]))
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

# =========================
# MAIN STREAMING PIPELINE
# =========================
tar_files = [f for f in os.listdir(TAR_DIR) if f.endswith(".tar.gz")]
total_tars = len(tar_files)

print(f"\nFound {total_tars} tar files\n")

processed_tars = 0
total_chunks = 0

for tar_name in tar_files:
    tar_path = os.path.join(TAR_DIR, tar_name)

    chunks = []
    payloads = []

    try:
        with open(tar_path, "rb") as f:
            tar_bytes = f.read()

        with tarfile.open(fileobj=BytesIO(tar_bytes), mode="r:gz") as tar:
            for member in tar.getmembers():
                if member.name.endswith(".nxml"):
                    pmc_id = os.path.basename(member.name).replace(".nxml", "")
                    extracted = tar.extractfile(member)
                    if not extracted:
                        continue

                    text = extract_clean_text(extracted.read())
                    if len(text) < 1000:
                        continue

                    article_chunks = chunk_text(text)

                    for idx, chunk in enumerate(article_chunks):
                        chunks.append(chunk)
                        payloads.append({
                            "pmc_id": pmc_id,
                            "chunk_id": idx,
                            "source": "PMC Open Access",
                            "text": chunk
                        })

        if not chunks:
            continue

        # =========================
        # EMBEDDING (BATCHED)
        # =========================
        embeddings = model.encode(
            chunks,
            batch_size=EMBED_BATCH_SIZE,
            show_progress_bar=False
        )

        # =========================
        # UPSERT TO QDRANT
        # =========================
        points = []
        for i, emb in enumerate(embeddings):
            points.append({
                "id": str(uuid.uuid4()),
                "vector": emb.tolist(),
                "payload": payloads[i]
            })

        for i in range(0, len(points), QDRANT_BATCH_SIZE):
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points[i:i + QDRANT_BATCH_SIZE]
            )

        total_chunks += len(points)
        processed_tars += 1

        print(f"[{processed_tars}/{total_tars}] {tar_name} → {len(points)} chunks uploaded")

        # =========================
        # FREE MEMORY
        # =========================
        del chunks, payloads, embeddings, points
        gc.collect()

    except Exception as e:
        print(f"Failed {tar_name}: {e}")

# =========================
# FINAL VERIFY
# =========================
count = client.count(collection_name=COLLECTION_NAME)
print(f"\nDONE")
print(f"Total tar files processed: {processed_tars}")
print(f"Total chunks uploaded: {total_chunks}")
print(f"Qdrant reports vectors: {count}")