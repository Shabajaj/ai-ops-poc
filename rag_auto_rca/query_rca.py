"""
Given a new alert description, retrieves the most similar past incident and
synthesizes a root-cause-analysis (RCA) suggestion from it.

Two synthesis modes:
  1. Real LLM synthesis — if ANTHROPIC_API_KEY is set, calls the Anthropic
     Messages API to write a proper RCA using the retrieved incident as
     context (this is the "real RAG" path: retrieve, then generate).
  2. Template fallback — if no API key is set, pulls the Root Cause and Fix
     sections directly out of the retrieved markdown doc and presents them
     as a suggestion, clearly labeled as a template match rather than a
     generated analysis. This mode has no external dependency and is what's
     been tested end-to-end for this POC.

RETRIEVAL: now backed by Pinecone + sentence-transformer embeddings
(see ingest.py) instead of TF-IDF. This file's synthesis step (the LLM
call / template fallback below) didn't need to change at all when that
swap happened — it only cares about "here's the retrieved incident text
and a similarity score," not how retrieval produced them. That's the
intended benefit of separating retrieval from generation in a RAG
pipeline: you can upgrade one without touching the other.
"""

import argparse
import json
import os
import re
import urllib.request

from ingest import INDEX_NAME, embed, get_pinecone_client

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
# Model id is read from the environment rather than hardcoded, since
# Anthropic model names change over time — set ANTHROPIC_MODEL to whatever
# current model id is listed at anthropic.com/docs before using this path.
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL")


def extract_section(markdown_text: str, heading: str) -> str:
    """
    Pulls the body of a `## <heading>` section out of an incident markdown
    file, stopping at the next `##` heading. Simple regex-based parsing —
    fine because these files are hand-written with a consistent structure;
    would not be robust against arbitrary markdown.
    """
    pattern = rf"##\s+{re.escape(heading)}\s*\n(.*?)(?=\n##\s+|\Z)"
    match = re.search(pattern, markdown_text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else "(section not found)"


def template_fallback_rca(alert_description: str, incident: dict, score: float) -> str:
    """Builds an RCA suggestion purely from the retrieved doc's own sections."""
    root_cause = extract_section(incident["text"], "Root Cause")
    fix = extract_section(incident["text"], "Fix")

    return f"""[TEMPLATE-BASED MATCH — no LLM call, ANTHROPIC_API_KEY not set]

Alert: {alert_description}

Closest matching past incident: {incident['filename']} (similarity={score:.3f})

Suggested Root Cause (from matched incident):
{root_cause}

Suggested Fix (from matched incident):
{fix}

Note: this is a direct excerpt from the matched postmortem, not a generated
analysis of the new alert. Review before assuming it applies as-is — the
new alert may share surface symptoms with the matched incident without
sharing the same root cause.
"""


def llm_synthesized_rca(alert_description: str, incident: dict, score: float, api_key: str) -> str:
    """
    Calls the real Anthropic API to synthesize an RCA suggestion, using the
    retrieved incident as grounding context. This is the "generation" half
    of RAG — retrieval already happened in ingest.py/build_index().
    """
    prompt = f"""You are assisting an SRE with root-cause analysis for a new alert.

New alert description:
{alert_description}

The most similar past incident postmortem (retrieved via similarity search,
similarity score {score:.3f}) is below. Use it as grounding context, but
reason about whether it actually applies to the new alert rather than
assuming it does.

--- RETRIEVED POSTMORTEM: {incident['filename']} ---
{incident['text']}
--- END POSTMORTEM ---

Write a concise RCA suggestion for the new alert: a likely root cause
hypothesis (informed by, but not copied from, the retrieved postmortem),
and a suggested next diagnostic step or fix. Be explicit if you think the
retrieved incident may NOT be a good match despite being the closest one
available."""

    body = json.dumps(
        {
            "model": ANTHROPIC_MODEL,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        ANTHROPIC_API_URL,
        data=body,
        headers={
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.loads(response.read().decode("utf-8"))

    text_parts = [block["text"] for block in result.get("content", []) if block.get("type") == "text"]
    return "[LLM-SYNTHESIZED — via Anthropic API]\n\n" + "\n".join(text_parts)


def retrieve_top_match(alert_description: str) -> tuple[dict, float]:
    """
    Embeds the incoming alert text with the same model used to embed the
    incident postmortems (embeddings from different models aren't
    comparable — this MUST match ingest.py's model), then asks Pinecone
    for the single nearest stored vector by cosine similarity.
    """
    pc = get_pinecone_client()
    index = pc.Index(INDEX_NAME)

    query_vector = embed([alert_description])[0]
    response = index.query(vector=query_vector, top_k=1, include_metadata=True)

    if not response.matches:
        raise RuntimeError("Pinecone index is empty — run ingest.py first.")

    match = response.matches[0]
    incident = {"filename": match.metadata["filename"], "text": match.metadata["text"]}
    return incident, match.score


def query(alert_description: str) -> str:
    incident, score = retrieve_top_match(alert_description)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key and not ANTHROPIC_MODEL:
        print("[warn] ANTHROPIC_API_KEY is set but ANTHROPIC_MODEL is not; falling back to template mode.")
    elif api_key:
        try:
            return llm_synthesized_rca(alert_description, incident, score, api_key)
        except Exception as exc:  # noqa: BLE001 - deliberately broad: any API failure should fall back, not crash the demo
            print(f"[warn] Anthropic API call failed ({exc}); falling back to template mode.")

    return template_fallback_rca(alert_description, incident, score)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "alert_description",
        nargs="?",
        default=(
            "Service pods entering CrashLoopBackOff with no application "
            "log output right after a container image rebuild"
        ),
        help="Free-text description of the new alert/symptom to investigate",
    )
    args = parser.parse_args()

    print(query(args.alert_description))


if __name__ == "__main__":
    main()
