import os
from typing import Any, Dict, List

import requests
from fastapi import FastAPI, Query, HTTPException
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

app = FastAPI()

# --- CONFIG ---
INDEX_DIR = "faiss_index"
EMBED_MODEL = "all-MiniLM-L6-v2"

SYSTEM_PROMPT = """You are a resume assistant.
Answer ONLY using the provided CONTEXT.
- Be concise (2-5 bullet points max unless user asks more).
- If the answer is not in the context, say: "I don't know from the provided resume."
- Do NOT add unrelated info.
- Do NOT hallucinate.
"""

TOP_K = 4


def ollama_generate(prompt: str, model: str = "mistral") -> str:
    """
    Calls Ollama local server (free) and returns plain text response.
    Make sure Ollama is running on: http://localhost:11434
    """
    url = "http://localhost:11434/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}

    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def _load_db() -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    if not os.path.exists(INDEX_DIR):
        raise RuntimeError(f"FAISS index folder not found: {INDEX_DIR}")

    return FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def _format_citations(docs: List[Any]) -> List[Dict[str, Any]]:
    cites = []
    for i, d in enumerate(docs, start=1):
        meta = getattr(d, "metadata", {}) or {}
        cites.append(
            {
                "rank": i,
                "source": meta.get("source") or meta.get("file") or meta.get("filename") or "unknown",
                "page": meta.get("page") or meta.get("page_number") or None,
                "snippet": (getattr(d, "page_content", "") or "")[:350],
            }
        )
    return cites


def _build_context(docs: List[Any]) -> str:
    parts = []
    for i, d in enumerate(docs, start=1):
        txt = (getattr(d, "page_content", "") or "").strip()
        if not txt:
            continue
        parts.append(f"[Chunk {i}]\n{txt}")
    return "\n\n".join(parts)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ask")
def ask(
    q: str = Query(..., description="Question"),
    citations: bool = Query(False, description="Return citations/chunks used"),
):
    # 1) Load DB
    try:
        db = _load_db()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector DB load failed: {e}")

    # 2) Retrieve
    try:
        docs = db.similarity_search(q, k=TOP_K)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")

    # 3) Build context
    context = _build_context(docs)
    context = context[:3500]  # keep prompt small & fast

    if not context.strip():
        result = {"question": q, "answer": "I don't know from the provided resume."}
        if citations:
            result["citations"] = []
        return result

    # 4) LLM (Ollama - Free)
    user_prompt = f"""SYSTEM:
{SYSTEM_PROMPT}

CONTEXT:
{context}

QUESTION:
{q}

Return the final answer only (no preface)."""

    try:
        answer = ollama_generate(user_prompt, model="mistral")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama call failed: {e}")

    result = {"question": q, "answer": answer}

    if citations:
        result["citations"] = _format_citations(docs)

    return result
