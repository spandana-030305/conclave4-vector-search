from sentence_transformers import SentenceTransformer

text_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(text: str) -> list:
    return text_model.encode(text).tolist()


# ---- Stub encoders (architecture-complete, implementation-light) ----

def embed_image(image_url: str) -> list:
    """
    Temporary image embedding placeholder.
    Returns a 512-dim vector to match image collection schema.
    """
    import numpy as np

    np.random.seed(abs(hash(image_url)) % (2**32))  # deterministic per image
    return np.random.rand(512).tolist()

def embed_audio(audio_url: str) -> list:
    # Placeholder: replace with Whisper / wav2vec later
    return text_model.encode(f"audio:{audio_url}").tolist()

def embed_video(video_url: str) -> list:
    # Placeholder: replace with video encoder later
    return text_model.encode(f"video:{video_url}").tolist()
