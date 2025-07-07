# inspect_chunks.py  ────────────────────────────────────────────────
import os, pickle, pandas as pd
from textwrap import shorten


def preview_chunks(base_dir: str, persist_name: str = "chroma_db", n: int = 25):
    vect_dir = os.path.join(base_dir, "data", persist_name)
    path = os.path.join(vect_dir, "docs.pkl")

    with open(path, "rb") as f:
        docs = pickle.load(f)

    rows = []
    for d in docs[:n]:
        rows.append({
            "file": d.metadata.get("source_file"),
            "type": d.metadata.get("chunk_type"),
            "json_path": d.metadata.get("json_path"),
            "kv_key": d.metadata.get("kv_key"),
            "kv_value": d.metadata.get("kv_value"),
            "doc_no": d.metadata.get("doc_no"),
            "doc_date": d.metadata.get("doc_date"),
            "content": shorten(d.page_content, width=80, placeholder="…"),
        })

    df = pd.DataFrame(rows)
    print(df.to_markdown(index=False))


if __name__ == "__main__":
    BASE_DIR = r"D:\python_projects\data_scout"  # ← adjust if needed
    preview_chunks(BASE_DIR, n=30)
