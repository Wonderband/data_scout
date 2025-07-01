from __future__ import annotations
from typing import Any, Generator, List
from langchain.schema import Document
from utils.text_utils import normalise, extract_facets

MAX_TOKENS = 180  # limit token count per Document


def iter_chunks(node: Any, path: List[str]) -> Generator[Document, None, None]:
    """
    Recursively yield LangChain Documents for primitive fields of nested data.
    Long blocks are split by newline. No sentence-based splitting is used.
    """

    def make_doc(text_chunk: str) -> Document:
        return Document(
            page_content=normalise(text_chunk),
            metadata={"json_path": "/".join(path) or "$", **extract_facets(text_chunk)}
        )

    if isinstance(node, dict):
        prim = {k: v for k, v in node.items() if not isinstance(v, (dict, list))}
        kids = {k: v for k, v in node.items() if isinstance(v, (dict, list))}

        if prim:
            text = "\n".join(f"{k}: {v}" for k, v in prim.items())
            words = text.split()

            if len(words) > MAX_TOKENS:
                lines = text.split("\n")
                chunk_lines, chunk_words = [], 0

                for line in lines:
                    lw = len(line.split())
                    if chunk_lines and chunk_words + lw > MAX_TOKENS:
                        yield make_doc("\n".join(chunk_lines))
                        chunk_lines, chunk_words = [], 0
                    chunk_lines.append(line)
                    chunk_words += lw

                if chunk_lines:
                    yield make_doc("\n".join(chunk_lines))
            else:
                yield make_doc(text)

        for k, v in kids.items():
            yield from iter_chunks(v, path + [k])

    elif isinstance(node, list):
        for i, item in enumerate(node):
            yield from iter_chunks(item, path + [f"[{i}]"])

    else:
        field = path[-2]
        text = f"{field}: {node}"
        yield make_doc(text)
