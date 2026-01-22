import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

text_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(text: str) -> list:
    return text_model.encode(text).tolist()


# ---- Stub encoders (architecture-complete, implementation-light) ----

# =========================
# IMAGE MODEL (CLIP)
# =========================
DEVICE = "cpu"

clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
).to(DEVICE)

clip_processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

def embed_image(image_path: str) -> list:
    image = Image.open(image_path).convert("RGB")

    inputs = clip_processor(
        images=image,
        return_tensors="pt"
    )

    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)

    emb = emb.cpu().numpy()[0]

    # MUST MATCH INGESTION NORMALIZATION
    emb = emb / np.linalg.norm(emb)

    return emb.tolist()

def embed_image_text(text: str) -> list:
    """
    CLIP TEXT embedding
    Used for Text → Image retrieval
    MUST match embed_image() vector space
    """

    inputs = clip_processor(
        text=[text],
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        emb = clip_model.get_text_features(**inputs)

    emb = emb.cpu().numpy()[0]

    # MUST MATCH IMAGE INGESTION NORMALIZATION
    emb = emb / np.linalg.norm(emb)

    return emb.tolist()

def embed_audio(audio_url: str) -> list:
    # Placeholder: replace with Whisper / wav2vec later
    return text_model.encode(f"audio:{audio_url}").tolist()

def embed_video(video_url: str) -> list:
    # Placeholder: replace with video encoder later
    return text_model.encode(f"video:{video_url}").tolist()
