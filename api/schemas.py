from pydantic import BaseModel
from typing import Optional, Dict, Union
from enum import Enum


# -----------------------------
# Modality Enum
# -----------------------------
class Modality(str, Enum):
    text = "text"
    image = "image"
    audio = "audio"
    video = "video"


# -----------------------------
# Base Request (shared fields)
# -----------------------------
class BaseSearchRequest(BaseModel):
    user_id: str
    session_id: str
    filters: Optional[Dict] = None
    top_k: int = 5


# -----------------------------
# Modality-specific Requests
# -----------------------------
class TextSearchRequest(BaseSearchRequest):
    modality: Modality = Modality.text
    query: str


class ImageSearchRequest(BaseSearchRequest):
    modality: Modality = Modality.image
    image_url: str


class AudioSearchRequest(BaseSearchRequest):
    modality: Modality = Modality.audio
    audio_url: str


class VideoSearchRequest(BaseSearchRequest):
    modality: Modality = Modality.video
    video_url: str


# -----------------------------
# Unified Search Request Type
# -----------------------------
SearchRequest = Union[
    TextSearchRequest,
    ImageSearchRequest,
    AudioSearchRequest,
    VideoSearchRequest
]
