import re
import unicodedata

DATE_ISO = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
DOC_NO = re.compile(r"[№N]\s*(\d+)")


def normalise(text: str) -> str:
    """Unicode + trivial domain‑specific normalisation."""
    text = unicodedata.normalize("NFKC", text)  # canon. equivalence
    text = text.replace("№", "N").replace("–", "-")  # homoglyphs
    return text.lower().strip()


def extract_facets(value: str) -> dict[str, str]:
    """Pull small, filter‑worthy attributes from an arbitrary string."""
    facets = {}
    if m := DOC_NO.search(value):
        facets["doc_no"] = m.group(1)
    if m := DATE_ISO.search(value):
        facets["doc_date"] = m.group(0)
    return facets
