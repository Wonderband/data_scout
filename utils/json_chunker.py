from __future__ import annotations
import json
from typing import Any, Generator, List
from langchain.schema import Document
from utils.text_utils import normalise, extract_facets


def make_doc(text_chunk: str, path: List[str], chunk_type: str,
             kv_key: str | None = None, kv_value: str | None = None) -> Document:
    """
    Build one Document with all useful metadata in a single pass.
    """
    meta = {
        "json_path": "/".join(path) or "$",
        "chunk_type": chunk_type,
        **extract_facets(text_chunk),   # doc_no, doc_date, …
    }
    # Only attach when supplied (keeps full_doc lean)
    if kv_key is not None:
        meta["kv_key"] = kv_key
    if kv_value is not None:
        meta["kv_value"] = kv_value

    return Document(page_content=normalise(text_chunk), metadata=meta)


def iter_chunks(node: Any, path: List[str]) -> Generator[Document, None, None]:
    """
    Emit three granularities:
      • full_doc  – entire JSON once
      • field     – “k: v” string for every primitive pair
      • value     – value-only chunk for every primitive pair
    """
    # ① Full document (root only)
    if not path:
        yield make_doc(json.dumps(node, ensure_ascii=False), [], "full_doc")

    # ② Recurse
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(v, (dict, list)):
                yield from iter_chunks(v, path + [k])
            else:
                raw_v = str(v).strip()
                if not raw_v:
                    continue
                norm_k = normalise(k)
                norm_v = normalise(str(v))

                # k: v field chunk
                yield make_doc(f"{k}: {v}", path + [k], "field",
                               kv_key=norm_k, kv_value=norm_v)

                # value-only chunk (helps exact/fuzzy matches)
                yield make_doc(str(v), path + [k], "value",
                               kv_key=norm_k, kv_value=norm_v)

    elif isinstance(node, list):
        for i, item in enumerate(node):
            yield from iter_chunks(item, path + [f"[{i}]"])

    else:  # primitive list item
        raw_node = str(node).strip()
        if not raw_node:
            return
        parent_key = path[-2] if len(path) >= 2 else path[-1]
        yield make_doc(str(node), path, "value",
                       kv_key=normalise(parent_key),
                       kv_value=normalise(str(node)))
