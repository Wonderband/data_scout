import os
import json
import logging
from langchain.schema import Document
import boto3
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import BedrockEmbeddings


def create_db(json_paths, base_dir):
    documents = []
    for path in json_paths:
        # 1) Read raw file
        raw = open(path, encoding="utf-8").read().strip()

        # 2) Remove Markdown‐style fences ```json … ```
        if raw.startswith("```json"):
            lines = raw.splitlines()
            # drop the first line (```json) and last line (```)
            raw = "\n".join(lines[1:-1]).strip()

        # 3) Drop any leading/trailing triple‐single‐quotes
        if raw.startswith("'''"):
            raw = raw[3:].strip()
        if raw.endswith("'''"):
            raw = raw[:-3].strip()

        # 4) Drop a lone leading “json”
        if raw.lower().startswith("json"):
            # remove the word “json” and any following punctuation
            raw = raw[len("json"):].lstrip(" :\n")

        # 5) Skip if now empty or just an empty object
        if raw in ("", "{}", "{}\n"):
            logging.info(f"Skipping empty JSON in: {path}")
            continue

        # 6) Parse JSON
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            logging.error(f"JSON parse error in {path}: {e}")
            continue

        # 7) Skip if it’s still empty
        if not isinstance(data, dict) or not data:
            logging.info(f"Skipping JSON with no fields in: {path}")
            continue

        # 8) Build Document
        metadata = {"source_file": os.path.basename(path)}
        text = "\n".join(f"{k}: {v}" for k, v in data.items())
        documents.append(Document(page_content=text, metadata=metadata))

    if not documents:
        return "No valid JSON content found to index."

    # --- embed & index ---
    bedrock = boto3.client(
        service_name="bedrock-runtime",
        region_name='us-east-1'
    )
    embeddings = BedrockEmbeddings(
        client=bedrock,
        model_id="amazon.titan-embed-text-v1"
    )

    vector_store = Chroma.from_documents(
        documents,
        embedding=embeddings,
        persist_directory=os.path.join(base_dir, "data", "chroma_db")
    )

    return f"Chroma DB built with {len(documents)} documents."
