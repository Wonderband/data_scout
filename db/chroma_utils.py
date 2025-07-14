import os, json, logging, pickle, re, pathlib
from typing import List, Tuple
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import BedrockEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
import boto3
from rank_bm25 import BM25Okapi
from utils.text_utils import normalise
from utils.json_chunker import iter_chunks


def create_db(json_paths: List[str],
              base_dir: str,
              persist_name: str = "chroma_acts_db") -> str:
    """
    Build a *hybrid* index:
      • Chroma vector store for semantic search
      • BM25Okapi corpus pickle for exact / fuzzy term scoring
    Each JSON is chunked by utils.json_chunker.iter_chunks, which already
    attaches:
        kv_key, kv_value, doc_no, doc_date, json_path, chunk_type …
    """
    docs: List[Document] = []

    # ---------- ingest ---------- #
    for p in json_paths:
        try:
            data = json.load(open(p, encoding="utf-8"))
        except json.JSONDecodeError as e:
            logging.warning(f"Skip {p}: {e}")
            continue

        for d in iter_chunks(data, []):
            d.metadata["source_file"] = os.path.basename(p)
            docs.append(d)

    if not docs:
        return "No valid JSON files found."

    # ---------- vector index (Chroma) ---------- #
    embeddings = BedrockEmbeddings(
        client=boto3.client("bedrock-runtime", region_name="us-east-1"),
        model_id="amazon.titan-embed-text-v2:0",
    )
    vect_dir = os.path.join(base_dir, "data", "acts", persist_name)
    Chroma.from_documents(docs, embeddings, persist_directory=vect_dir)

    # ---------- lexical index (BM25) ---------- #
    tokenized_corpus = [normalise(doc.page_content).split() for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus)

    # ---------- persist metadata ---------- #
    with open(os.path.join(vect_dir, "docs.pkl"), "wb") as f:
        pickle.dump(docs, f)
    with open(os.path.join(vect_dir, "bm25.pkl"), "wb") as f:
        pickle.dump(bm25, f)

    return (
        f"Hybrid DB built: {len(docs)} chunks "
        f"(field, value, full_doc) across {len(json_paths)} files."
    )

#
# def _init_store(base_dir: str, persist_name: str = "chroma_db") -> Tuple:
#     """Load Chroma vector store and raw docs metadata."""
#
#     vectordb_dir = os.path.join(base_dir, "data", "acts", persist_name)
#
#     embeddings = BedrockEmbeddings(
#         client=boto3.client("bedrock-runtime", region_name="us-east-1"),
#         model_id="amazon.titan-embed-text-v2:0"
#     )
#     vect = Chroma(
#         persist_directory=vectordb_dir,
#         embedding_function=embeddings
#     )
#     docs = pickle.load(open(os.path.join(vectordb_dir, "docs.pkl"), "rb"))
#     return vect, docs
#
#
# # ---------- search phase ---------- #
#
# def retrieve_files(
#         query: str,
#         base_dir: str,
#         top_k: int = 10,
#         persist_name="chroma_db"
# ) -> List[Tuple[str, float, str, str]]:
#     """
#     Hybrid retrieval using BM25 + Dense (chunk-level).
#     Returns up to top_k files with best-matching chunk per file:
#         (source_file, hybrid_score, json_path, chunk_content)
#     """
#     # Load vector store and raw docs
#     vectordb_dir = os.path.join(base_dir, "data", "acts", persist_name)
#     embeddings = BedrockEmbeddings(
#         client=boto3.client("bedrock-runtime", region_name="us-east-1"),
#         model_id="amazon.titan-embed-text-v2:0"
#     )
#     vect = Chroma(
#         persist_directory=vectordb_dir,
#         embedding_function=embeddings
#     )
#
#     with open(os.path.join(vectordb_dir, "docs.pkl"), "rb") as f:
#         docs = pickle.load(f)
#
#     # Hybrid retriever: dense + BM25
#     dense_retriever = vect.as_retriever(search_kwargs={"k": 100})
#     sparse_retriever = BM25Retriever.from_documents(docs, k=100)
#     hybrid = EnsembleRetriever(
#         retrievers=[dense_retriever, sparse_retriever],
#         weights=[0.7, 0.3]
#     )
#
#     # Get top hybrid chunks
#     hits = hybrid.get_relevant_documents(normalise(query))
#
#     # Select best chunk per file
#     best_score_per_file = {}
#     best_info_per_file = {}
#     for d in hits:
#         fn = d.metadata["source_file"]
#         score = getattr(d, "score", None)
#         if score is None:
#             continue  # Should not happen but just in case
#
#         if fn not in best_score_per_file or score > best_score_per_file[fn]:
#             best_score_per_file[fn] = score
#             best_info_per_file[fn] = (d.metadata["json_path"], d.page_content)
#
#     # Sort by best score and return top_k
#     top_stage1 = sorted(best_score_per_file.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
#     return [
#         (fn, score, *best_info_per_file[fn])
#         for fn, score in top_stage1
#     ]
#
#     # Stage 2: full-doc reranking
#     # Prepare a full‑doc retriever
#     # retriever_full = vect.as_retriever(
#     #     search_kwargs={"k": 1},
#     #     filter={"chunk_type": "full_doc"}
#     # )
#     #
#     # final_results = []
#     # for fn, s1 in top_stage1:
#     #     # Query the single full-doc chunk for this file
#     #     hits_full = retriever_full.get_relevant_documents(normalise(query))
#     #     # Find the hit whose metadata.source_file == fn
#     #     s2 = 0.0
#     #     for hf in hits_full:
#     #         if hf.metadata["source_file"] == fn:
#     #             s2 = hf.score
#     #             break
#     #     # Combined score
#     #     final_score = (1.0 - stage2_weight) * s1 + stage2_weight * s2
#     #     json_path, snippet = best_info_per_file[fn]
#     #     final_results.append((fn, final_score, json_path, snippet))
#     #
#     # # Return sorted by final_score
#     # return sorted(final_results, key=lambda x: x[1], reverse=True)
#
#
# def search_hybrid(query: str, base_dir: str, top_k: int = 5, persist_name="chroma_db"):
#     vectordb_dir = os.path.join(base_dir, "data", persist_name)
#     embeddings = BedrockEmbeddings(
#         client=boto3.client("bedrock-runtime", region_name="us-east-1"),
#         model_id="amazon.titan-embed-text-v2:0"
#     )
#     vector_store = Chroma(
#         persist_directory=vectordb_dir,
#         embedding_function=embeddings
#     )
#
#     # load docs for BM25
#     with open(os.path.join(vectordb_dir, "docs.pkl"), "rb") as f:
#         docs = pickle.load(f)
#
#     sparse_retriever = BM25Retriever.from_documents(docs, k=top_k)  # :contentReference[oaicite:5]{index=5}
#     dense_retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
#
#     hybrid = EnsembleRetriever(  # :contentReference[oaicite:6]{index=6}
#         retrievers=[dense_retriever, sparse_retriever],
#         weights=[0.7, 0.3]
#     )
#
#     hits = hybrid.get_relevant_documents(normalise(query))
#     # de‑duplicate by source file + field
#     seen, out_lines = set(), []
#     for d in hits:
#         key = (d.metadata["source_file"], d.metadata['json_path'])
#         if key in seen:
#             continue
#         seen.add(key)
#         out_lines.append(f"📄 {d.metadata['source_file']} → {d.metadata['json_path']}\n{d.page_content}")
#
#     return "\n\n---\n\n".join(out_lines[:top_k])
