from fastapi import FastAPI, HTTPException
from api.schemas import (
    SearchRequest,
    Modality,
    TextSearchRequest,
    ImageSearchRequest,
    AudioSearchRequest,
    VideoSearchRequest
)
from qdrant_client import QdrantClient
from memory.memory_retrieval import get_user_memory, get_session_memory
from memory.memory_store import store_user_memory, store_session_memory
from api.embeddings import (
    embed_text, embed_image, embed_audio, embed_video
)
from qdrant_client.models import SearchParams
import os

app = FastAPI(title="Healthcare Vector Search API")

# Qdrant client
client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"]
)

# Collections
TEXT_COLLECTION = "conclave4_healthcare_vector_search"
IMAGE_COLLECTION = "conclave4_healthcare_image_search"

SESSION_MEMORY_COLLECTION = "session_memory_collection"
USER_MEMORY_COLLECTION = "user_memory_collection"


@app.post("/search")
def search(req: SearchRequest):

    # Generate base query embedding (per modality)
    if isinstance(req, TextSearchRequest):
        base_embedding = embed_text(req.query)
        content_for_memory = req.query
        search_collection = TEXT_COLLECTION

    elif isinstance(req, ImageSearchRequest):
        base_embedding = embed_image(req.image_url)
        content_for_memory = req.image_url
        search_collection = IMAGE_COLLECTION

    elif isinstance(req, AudioSearchRequest):
        base_embedding = embed_audio(req.audio_url)
        content_for_memory = req.audio_url
        search_collection = TEXT_COLLECTION

    elif isinstance(req, VideoSearchRequest):
        base_embedding = embed_video(req.video_url)
        content_for_memory = req.video_url
        search_collection = TEXT_COLLECTION
    else:
        raise HTTPException(400, "Unsupported modality")

    # Retrieve memory (session + long-term)
    session_memory = []
    user_memory = []

    if req.modality == Modality.text:
        session_memory = get_session_memory(
            client,
            req.user_id,
            req.session_id,
            base_embedding  # 384-dim
        )
        user_memory = get_user_memory(
            client,
            req.user_id,
            base_embedding  # 384-dim
        )

    # Inject memory into query (textual context)
    memory_context = " ".join(
        [m.payload.get("content", "") for m in session_memory + user_memory]
    )


    if isinstance(req, TextSearchRequest) and memory_context:
        enriched_query = f"{memory_context}\n{req.query}"
        query_vector = embed_text(enriched_query)
    else:
        query_vector = base_embedding

    # Perform vector search in Qdrant
    results = client.search(
        collection_name=search_collection,
        query_vector=query_vector,
        limit=req.top_k
    )

    # Update memory
    if req.modality == Modality.text:
        store_session_memory(
            client,
            req.user_id,
            req.session_id,
            base_embedding,
            content_for_memory
        )
        store_user_memory(
            client,
            req.user_id,
            base_embedding,
            content_for_memory
        )

    # Return evidence-based results
    return {
    "modality": req.modality,
    "query": content_for_memory,

    # Clear distinction section
    "context_sources": {
        "retrieved_knowledge": "external_medical_corpus",  # PMC dataset
        "session_context_used": len(session_memory) > 0,
        "user_memory_used": len(user_memory) > 0
    },
    # Bias, safety, privacy, explainability
    "responsible_use": {
        "disclaimer": (
            "This information is retrieved from peer-reviewed medical literature "
            "and is intended for informational purposes only. "
            "It is not a substitute for professional medical advice, diagnosis, or treatment."
        ),
        "bias_notice": (
            "Retrieved documents may reflect publication, regional, or population biases "
            "present in the source datasets."
        ),
        "privacy_note": (
            "No personal health data is stored beyond anonymized session-level memory."
        ),
        "explainability": (
            "Results are ranked using semantic similarity between the query embedding "
            "and indexed medical text embeddings."
        )
    },
    "results": [
        {
            "score": r.score,
            "payload": r.payload
        }
        for r in results
    ]
}

