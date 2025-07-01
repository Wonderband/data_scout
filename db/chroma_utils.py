import os, json, logging, pickle, re
from typing import List
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import BedrockEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
import boto3

from utils.text_utils import normalise, extract_facets


# ---------- build phase ---------- #
def create_db(json_paths: List[str], base_dir: str, persist_name="chroma_db"):
    docs: List[Document] = []
    for path in json_paths:
        raw = open(path, encoding="utf-8").read().strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            logging.warning(f"Skip {path}: {e}")
            continue

        if not isinstance(data, dict):
            continue

        for k, v in data.items():
            text = normalise(f"{k}: {v}")
            meta = {"source_file": os.path.basename(path), "field": k}
            meta.update(extract_facets(str(v)))
            docs.append(Document(page_content=text, metadata=meta))

    if not docs:
        return "No valid JSON found."

    # --- Multilingual embeddings (Titan v2) ---
    embeddings = BedrockEmbeddings(
        client=boto3.client("bedrock-runtime", region_name="us-east-1"),
        model_id="amazon.titan-embed-text-v2:0"  # multilingual :contentReference[oaicite:4]{index=4}
    )

    vectordb_dir = os.path.join(base_dir, "data", persist_name)
    vector_store = Chroma.from_documents(
        docs, embedding=embeddings, persist_directory=vectordb_dir
    )

    # Persist raw docs for the BM25 layer
    with open(os.path.join(vectordb_dir, "docs.pkl"), "wb") as f:
        pickle.dump(docs, f)

    return f"Hybrid DB built: {len(docs)} field‑level chunks."


# ---------- search phase ---------- #
def search_hybrid(query: str, base_dir: str, top_k: int = 5, persist_name="chroma_db"):
    vectordb_dir = os.path.join(base_dir, "data", persist_name)
    embeddings = BedrockEmbeddings(
        client=boto3.client("bedrock-runtime", region_name="us-east-1"),
        model_id="amazon.titan-embed-text-v2:0"
    )
    vector_store = Chroma(
        persist_directory=vectordb_dir,
        embedding_function=embeddings
    )

    # load docs for BM25
    with open(os.path.join(vectordb_dir, "docs.pkl"), "rb") as f:
        docs = pickle.load(f)

    sparse_retriever = BM25Retriever.from_documents(docs, k=top_k)  # :contentReference[oaicite:5]{index=5}
    dense_retriever = vector_store.as_retriever(search_kwargs={"k": top_k})

    hybrid = EnsembleRetriever(  # :contentReference[oaicite:6]{index=6}
        retrievers=[dense_retriever, sparse_retriever],
        weights=[0.7, 0.3]
    )

    hits = hybrid.get_relevant_documents(normalise(query))
    # de‑duplicate by source file + field
    seen, out_lines = set(), []
    for d in hits:
        key = (d.metadata["source_file"], d.metadata["field"])
        if key in seen:
            continue
        seen.add(key)
        out_lines.append(f"📄 {d.metadata['source_file']} → {d.metadata['field']}\n{d.page_content}")

    return "\n\n---\n\n".join(out_lines[:top_k])
