"""
Builds a retrieval index over the sample incident postmortems.

SCOPE TRADE-OFF (be upfront about this in an interview): this uses TF-IDF +
cosine similarity, not real embeddings. That's a deliberate same-day-POC
decision, not a technical limitation I'm unaware of:
  - TF-IDF needs zero external dependencies beyond scikit-learn, no API
    calls, no vector DB, and is fully deterministic/offline — good for a
    quick, explainable demo with only 3 documents
  - A production version would use sentence embeddings (e.g. an embedding
    model via the Anthropic/OpenAI API, or a local model like
    sentence-transformers) stored in a real vector store (FAISS for
    local/small-scale, or Pinecone/pgvector/OpenSearch for a managed,
    scalable setup) — that gets you semantic similarity ("pods failing
    silently after a base image change" matching an incident that never
    uses those exact words), which TF-IDF's exact-token-overlap approach
    cannot do
  - At 3 documents, the gap between TF-IDF and embeddings is invisible; at
    thousands of postmortems it would matter a lot. Flagging that gap here
    is the point of the trade-off note.

This script builds the index in-memory each run (no persistence) since the
corpus is tiny and static for this POC. A real version would persist the
index and support incremental updates as new postmortems are written.
"""

import glob
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

INCIDENTS_DIR = os.path.join(os.path.dirname(__file__), "sample_incidents")


def load_incidents(incidents_dir: str = INCIDENTS_DIR) -> list[dict]:
    """Loads every .md file in the incidents directory into memory."""
    incidents = []
    for path in sorted(glob.glob(os.path.join(incidents_dir, "*.md"))):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        incidents.append({"path": path, "filename": os.path.basename(path), "text": text})
    return incidents


class IncidentIndex:
    """
    Thin wrapper around a TF-IDF vectorizer + cosine similarity search.
    Kept intentionally simple — this is the whole "retrieval" step of the
    RAG pipeline. `query_rca.py` swaps this class out for a real vector
    store in the imagined "production" version without changing the
    downstream synthesis step.
    """

    def __init__(self, incidents: list[dict]):
        self.incidents = incidents
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform([doc["text"] for doc in incidents])

    def search(self, query: str, top_k: int = 1) -> list[tuple[dict, float]]:
        """Returns the top_k most similar incidents as (incident, score) pairs."""
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix)[0]
        ranked = sorted(zip(self.incidents, scores), key=lambda pair: pair[1], reverse=True)
        return ranked[:top_k]


def build_index(incidents_dir: str = INCIDENTS_DIR) -> IncidentIndex:
    incidents = load_incidents(incidents_dir)
    if not incidents:
        raise RuntimeError(f"No incident .md files found in {incidents_dir}")
    return IncidentIndex(incidents)


if __name__ == "__main__":
    # Quick manual sanity check: build the index and show what it contains.
    index = build_index()
    print(f"Indexed {len(index.incidents)} incidents:")
    for doc in index.incidents:
        print(f"  - {doc['filename']}")
