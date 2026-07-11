"""
Pinecone-backed retrieval index over the incident postmortems — replaces
the earlier TF-IDF + cosine-similarity approach with real sentence
embeddings and a real vector database.

WHAT CHANGED FROM TF-IDF, AND WHY IT MATTERS: TF-IDF only measures literal
word overlap between a query and each document — "pods keep dying with no
logs" and "CrashLoopBackOff with no application output" score as barely
related, because they share almost no words, even though they describe
the same failure. An embedding model instead maps text to a vector (a
list of numbers, here 384 of them) positioned so that texts with similar
MEANING end up close together in that space, regardless of exact wording.
Pinecone then does the actual "find the nearest vectors" search — for a
3-document corpus a plain Python loop over embeddings would work exactly
as well, but this is the real infrastructure you'd reach for once that
corpus is thousands of postmortems and a linear scan stops being cheap.

EMBEDDING MODEL: sentence-transformers/all-MiniLM-L6-v2 — runs locally, no
API key or per-call cost, produces small 384-dimension vectors (cheap to
store and search), and is a well-benchmarked default for semantic search.
A production setup with more budget might use OpenAI's or Voyage's hosted
embedding APIs for marginally higher quality; noted as an option, not a
requirement — nothing about the retrieval or generation code downstream
cares which embedding model produced the vectors.
"""

import glob
import os

from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

INCIDENTS_DIR = os.path.join(os.path.dirname(__file__), "sample_incidents")
INDEX_NAME = "ai-ops-poc-incidents"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # fixed by the model above — must match the Pinecone index's own dimension

_model = None  # loaded lazily so importing this module doesn't pay the model-load cost every time


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    """
    Turns a list of strings into a list of embedding vectors. Normalized
    to unit length so that cosine similarity (what we configure the
    Pinecone index to use) behaves the way you'd expect — with normalized
    vectors, cosine similarity reduces to a simple dot product.
    """
    model = get_embedding_model()
    vectors = model.encode(texts, normalize_embeddings=True)
    return vectors.tolist()


def load_incidents(incidents_dir: str = INCIDENTS_DIR) -> list[dict]:
    """Loads every .md file in the incidents directory into memory."""
    incidents = []
    for path in sorted(glob.glob(os.path.join(incidents_dir, "*.md"))):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        incidents.append({"path": path, "filename": os.path.basename(path), "text": text})
    return incidents


def get_pinecone_client() -> Pinecone:
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "PINECONE_API_KEY is not set. Get a free API key at pinecone.io "
            "and set it as an environment variable before running this."
        )
    return Pinecone(api_key=api_key)


def ensure_index(pc: Pinecone):
    """Creates the Pinecone index if it doesn't already exist, then returns a handle to it."""
    if not pc.has_index(INDEX_NAME):
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),  # free-tier-eligible region
        )
    return pc.Index(INDEX_NAME)


def build_index(incidents_dir: str = INCIDENTS_DIR):
    """
    Embeds every incident doc and upserts them into Pinecone.

    Pinecone metadata has a 40KB-per-vector size limit; our postmortems
    are only a few KB each, so storing the full text as metadata directly
    is fine — retrieval doesn't need a second lookup back to local files,
    which matters once this runs somewhere that doesn't have the repo
    checked out (e.g. a KServe/Lambda-style deployment).
    """
    incidents = load_incidents(incidents_dir)
    if not incidents:
        raise RuntimeError(f"No incident .md files found in {incidents_dir}")

    pc = get_pinecone_client()
    index = ensure_index(pc)

    vectors = embed([doc["text"] for doc in incidents])
    upsert_payload = [
        {
            "id": doc["filename"],
            "values": vector,
            "metadata": {"filename": doc["filename"], "text": doc["text"]},
        }
        for doc, vector in zip(incidents, vectors)
    ]
    index.upsert(vectors=upsert_payload)
    return index


if __name__ == "__main__":
    idx = build_index()
    print(f"Upserted incidents into Pinecone index '{INDEX_NAME}'.")
    print(f"Index stats: {idx.describe_index_stats()}")
