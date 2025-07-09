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


def search_hybrid_multi(
        query: str,
        base_dir: str,
        top_k: int = 10,
        persist_name: str = "chroma_db",
) -> List[Tuple[str, float, str, str]]:
    """
    Support multi-item queries separated by commas:
      e.g. "акти, 2024-06, UA32643240596946"
    We score each file on each sub-query, then sum the scores.
    """
    # 1) split and normalize each sub-query
    sub_queries = [q.strip() for q in query.split(",") if q.strip()]
    if not sub_queries:
        return []

    # load vect, docs, bm25 once
    vect, docs, bm25 = _load_stores(base_dir, persist_name)

    # precompute keys and BM25 base scores
    key_to_idx = {_chunk_key(d): i for i, d in enumerate(docs)}
    key_to_doc = {_chunk_key(d): d for d in docs}
    bm25_base_scores = bm25.get_scores  # function

    # this will hold per-file cumulative score
    file_scores: Dict[str, float] = {}

    # weight between dense and sparse
    dense_w = 0.7

    # for each sub-query, compute fused scores per file
    for raw_q in sub_queries:
        q_norm = normalise(raw_q)

        # (A) exact value shortcut
        matching = [
            h for h in docs
            if "kv_value" in h.metadata and q_norm in h.metadata["kv_value"]
        ]
        if matching:
            # any exact match → score 1.0 for that file on this sub-query
            for h in matching:
                fn = h.metadata["source_file"]
                file_scores[fn] = file_scores.get(fn, 0.0) + 1.0
            # move on to next sub-query
            continue

        # (B) dense + sparse hybrid
        # 1. dense hits
        dense_hits = vect.similarity_search_with_score(q_norm, k=100)
        bm25_scores = bm25_base_scores(q_norm.split())
        bm25_max = max(bm25_scores) or 1.0

        # build candidate dict for this sub-query
        cand: Dict[str, float] = {}  # file_name -> best fused score
        # process dense
        for d, dist in dense_hits:
            k = _chunk_key(d)
            fn = k[0]
            dense_sim = _scale_dense(dist)
            sparse_sim = bm25_scores[key_to_idx[k]] / bm25_max
            fused = dense_w * dense_sim + (1 - dense_w) * sparse_sim
            cand[fn] = max(cand.get(fn, 0.0), fused)

        # process sparse-only top
        top_sparse_idx = sorted(
            range(len(bm25_scores)),
            key=bm25_scores.__getitem__,
            reverse=True,
        )[:100]
        for idx in top_sparse_idx:
            k = _chunk_key(docs[idx])
            fn = k[0]
            if fn in cand:
                continue
            sparse_sim = bm25_scores[idx] / bm25_max
            cand[fn] = sparse_sim  # dense=0

        # accumulate into file_scores
        for fn, score in cand.items():
            file_scores[fn] = file_scores.get(fn, 0.0) + score

    # now we have a summed score per file; pick top_k
    top_files = sorted(
        file_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )[:top_k]

    # assemble the final output tuples
    results: List[Tuple[str, float, str, str]] = []
    for fn, agg_score in top_files:
        # find the doc for preview (just grab the first chunk we see)
        doc = next(d for d in docs if d.metadata["source_file"] == fn)
        results.append((fn, round(agg_score, 4), doc.metadata["json_path"], doc.page_content))

    return results

