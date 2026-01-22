# Overview
This project implements a multimodal semantic search system that supports text and image search using vector embeddings and Qdrant Cloud as the vector database. The system also incorporates session-level and long-term memory to enhance contextual retrieval.
The entire pipeline is exposed via a FastAPI service, making it easy to test and interact with using Swagger UI.

# Dataset Preparation
Downloaded and prepared text and image datasets.
Text and images are processed independently to support modality-specific embeddings and search.

# Embedding Generation
Different embedding models are used based on the data modality:
Text Embeddings
- Model: all-MiniLM-L6-v2
- Purpose: Converts textual data into dense vector representations for semantic search.
- Used for:
  Text-to-text search
  Memory storage and retrieval
Image Embeddings
- Model: OpenAI clip-vit-base-patch32
- Purpose: Converts images into vector embeddings.
- Enables:
  Image-to-image search
  Text-to-image retrieval using CLIP’s shared embedding space

# Vector Storage with Qdrant Cloud
Qdrant Cloud is used to store and manage all vector data efficiently.
Collections Created
- Text Collection
  Stores text embeddings generated using MiniLM
- Image Collection
  Stores image embeddings generated using CLIP
- Session Memory Collection
  Stores short-term, session-specific interactions
- User (Long-Term) Memory Collection
  Stores long-term user context for personalized retrieval
Each collection is optimized for fast similarity search and scalable storage.

# Backend Service
Framework: FastAPI
Provides a unified /search endpoint for:
- Text search
- Image search
- Cross-modal search (Text → Image)
Integrates:
- Embedding generation
- Qdrant vector search
- Session and long-term memory retrieval

# API Testing
The API is exposed using Swagger UI (FastAPI’s built-in documentation).
Swagger UI is used to:
- Test different modalities (text, image)
- Validate search results
- Verify memory-based contextual retrieval
(The score for text-to-text retrieval was around 0.65, for image-to-to retrieval it was around 0.9 and for text-to-image retrieval it was around 0.36)

# Tech Stack
- FastAPI – Backend API service
- Qdrant Cloud – Vector database
- Sentence Transformers (MiniLM) – Text embeddings
- OpenAI CLIP – Image and cross-modal embeddings
- Swagger UI – API testing and validation

# Setup Instructions
1. Prerequisites
Make sure you have the following installed:
- Python 3.9 or higher
- pip (Python package manager)
- Git
- A Qdrant Cloud account

2. Clone the Repository

git clone https://github.com/<your-username>/<your-repo-name>.git

cd <your-repo-name>

3. Create a Virtual Environment (Recommended)

python -m venv venv

Activate it:

venv\Scripts\activate (Windows)

source venv/bin/activate (Linux / macOS)

4. Install Dependencies

pip install -r requirements.txt

5. Set Up Qdrant Cloud
- Create an account in Qdrant Cloud
- Create the required collections:
  Text embeddings collection
  Image embeddings collection
  Session memory collection
  User (long-term) memory collection
- Copy the Qdrant Cloud URL and API Key

6. Configure Environment Variables

Set the following environment variables:
  
  Windows (PowerShell)
  
  setx QDRANT_URL "https://<your-qdrant-cluster-url>"
  
  setx QDRANT_API_KEY "<your-api-key>"
  
  Linux / macOS
  
  export QDRANT_URL="https://<your-qdrant-cluster-url>"
  
  export QDRANT_API_KEY="<your-api-key>"

7. Run the FastAPI Server

uvicorn main:app --reload

Once started, the API will be available at:
  
  http://127.0.0.1:8000

8. Access Swagger UI

Open your browser and navigate to:
  
  http://127.0.0.1:8000/docs

Use Swagger UI to:
- Test text search
- Test image search
- Test cross-modal (Text → Image) retrieval
- Validate memory-based retrieval
