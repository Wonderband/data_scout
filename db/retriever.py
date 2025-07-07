from __future__ import annotations
import os, pickle, re
from typing import List, Tuple, Dict
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_aws import BedrockEmbeddings
from utils.text_utils import normalise, DATE_ISO, DOC_NO


# ---------- helpers -------------------------------------------------
def _load_stores(base_dir: str, persist_name: str = "chroma_db"):
    """Return (Chroma, docs list, BM25Okapi)."""
    vect_dir = os.path.join(base_dir, "data", persist_name)
    docs = pickle.load(open(os.path.join(vect_dir, "docs.pkl"), "rb"))
    bm25 = pickle.load(open(os.path.join(vect_dir, "bm25.pkl"), "rb"))

    vect = Chroma(
        persist_directory=vect_dir,
        embedding_function=BedrockEmbeddings(
            region_name="us-east-1",
            model_id="amazon.titan-embed-text-v2:0",
        ),
    )
    return vect, docs, bm25


def _looks_like_exact(q: str) -> bool:
    """Detect ISO date, doc number, or single token."""
    if DATE_ISO.fullmatch(q):
        return True
    if DOC_NO.fullmatch(q):
        return True
    # Single, token-like (no spaces) alnum string ≥4 chars
    return bool(re.fullmatch(r"[a-zA-Z\u0400-\u04FF0-9/\-]{2,}", q))


def _scale_dense(dist: float) -> float:
    """Convert Chroma cosine *distance* to [0,1] similarity."""
    return max(0.0, min(1.0, 1.0 - dist / 2.0))


def _chunk_key(d: Document) -> tuple:
    """Immutable identifier for a chunk."""
    return (
        d.metadata.get("source_file"),
        d.metadata.get("json_path"),
        d.metadata.get("chunk_type"),
    )


# -------------------------------------------------------------------
def search_hybrid(
        query: str,
        base_dir: str,
        top_k: int = 10,
        persist_name: str = "chroma_db",
) -> List[Tuple[str, float, str, str]]:
    """
    Return (file, fused_score, json_path, preview_text) – max one chunk per file.
    """
    q_norm = normalise(query)
    vect, docs, bm25 = _load_stores(base_dir, persist_name)

    # ===== 1) exact-value shortcut ====================================

    matching = [
        h for h in docs
        if "kv_value" in h.metadata and q_norm in h.metadata["kv_value"]
    ]
    if matching:
        seen_files = set()
        unique = []
        for h in matching:
            fn = h.metadata["source_file"]
            if fn in seen_files:
                continue
            seen_files.add(fn)
            unique.append((
                fn,
                1.0,
                h.metadata["json_path"],
                h.page_content
            ))
            if len(unique) >= top_k:
                break
        return unique

    # ===== 2) hybrid search ==========================================
    dense_hits = vect.similarity_search_with_score(q_norm, k=100)

    # Pre-compute mappings for speed
    key_to_idx: Dict[tuple, int] = {_chunk_key(d): i for i, d in enumerate(docs)}
    key_to_doc: Dict[tuple, Document] = {_chunk_key(d): d for d in docs}

    bm25_scores = bm25.get_scores(q_norm.split())
    bm25_max = max(bm25_scores) or 1.0

    # — collect dense top-100
    cand: Dict[tuple, Tuple[float, float]] = {}  # key -> (denseSim, sparseSim)
    for d, dist in dense_hits:
        k = _chunk_key(d)
        dense_sim = _scale_dense(dist)
        sparse_norm = bm25_scores[key_to_idx[k]] / bm25_max
        cand[k] = (dense_sim, sparse_norm)

    # — add sparse-only top-100
    top_sparse_idx = sorted(
        range(len(bm25_scores)),
        key=bm25_scores.__getitem__,
        reverse=True,
    )[:100]
    for idx in top_sparse_idx:
        k = _chunk_key(docs[idx])
        if k in cand:
            continue
        cand[k] = (0.0, bm25_scores[idx] / bm25_max)

    # — fuse scores
    dense_w = 0.7
    fused: List[Tuple[float, tuple]] = [
        (dense_w * d + (1 - dense_w) * s, k) for k, (d, s) in cand.items()
    ]

    # ===== 3) best chunk per file =====================================
    best: Dict[str, Tuple[float, tuple]] = {}
    for score, key in sorted(fused, key=lambda t: t[0], reverse=True):
        fn = key[0]  # source_file
        if fn not in best:
            best[fn] = (score, key)
        if len(best) == top_k:
            break

    result = []
    for fn, (score, key) in best.items():
        doc = key_to_doc[key]
        result.append(
            (
                fn,
                round(score, 4),
                doc.metadata["json_path"],
                doc.page_content,
            )
        )
    return result
