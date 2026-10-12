import math
import os
import re
import sys

import chromadb
from chromadb.utils import embedding_functions

from llm import ask_llm
from ocr import ocr_file

CHUNK_WORDS = 150        # embedding model ~128 tokens hi padhta hai
OVERLAP = 30
MAX_SECTIONS = 15        # summary ke liye document ke max itne hisse
SECTION_MAX_WORDS = 12000

client = chromadb.PersistentClient(path="./chroma_db")
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)
chunks_col = client.get_or_create_collection("notes", embedding_function=embed_fn)
summary_col = client.get_or_create_collection("summaries", embedding_function=embed_fn)


def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text, size=CHUNK_WORDS, overlap=OVERLAP):
    words = text.split()
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start:start + size]
        if piece:
            chunks.append(" ".join(piece))
        if start + size >= len(words):
            break
    return chunks


def build_summary(pages):
    """Map-reduce: pehle har section ka summary, phir sabka overview."""
    pages = [p for p in pages if p["text"].strip()]
    if not pages:
        return ""

    n_sections = min(MAX_SECTIONS, max(1, math.ceil(len(pages) / 10)))
    per_section = math.ceil(len(pages) / n_sections)

    section_summaries = []
    for idx, s in enumerate(range(0, len(pages), per_section), start=1):
        group = pages[s:s + per_section]
        start_page, end_page = group[0]["page"], group[-1]["page"]
        text = " ".join(clean_text(p["text"]) for p in group)
        text = " ".join(text.split()[:SECTION_MAX_WORDS])
        print(f"Summary section {idx}/{n_sections} (pages {start_page}-{end_page})")
        summary = ask_llm(
            "Summarize the following part of a document in 5-6 lines. "
            "Mention the main topics and key ideas. Use the same language as the text. "
            "If the text is Hinglish (Hindi written in English letters), write the summary in Hinglish.\n\n"
            f"{text}"
        )
        section_summaries.append(f"Pages {start_page}-{end_page}: {summary.strip()}")

    joined = "\n\n".join(section_summaries)
    overview = ask_llm(
        "Below are summaries of consecutive parts of one document. "
        "Write a clear overview: what the document is about, who it is useful for, "
        "and a bullet list of the main topics covered, in order. "
        "Write the overview in the same language as the summaries.\n\n"
        f"{joined}"
    )
    return f"OVERVIEW:\n{overview.strip()}\n\nSECTION-WISE SUMMARY:\n{joined}"


def index_file(path, make_summary=True, max_pages=None):
    filename = os.path.basename(path)
    pages = ocr_file(path, max_pages=max_pages)

    # purana data hata do, taaki dobara index karne pe duplicate/stale chunks na rahein
    chunks_col.delete(where={"file": filename})

    ids, docs, metas = [], [], []
    for page in pages:
        text = clean_text(page["text"])
        for i, chunk in enumerate(chunk_text(text)):
            ids.append(f"{filename}-p{page['page']}-c{i}")
            docs.append(chunk)
            metas.append({"file": filename, "page": page["page"]})
    if docs:
        chunks_col.upsert(ids=ids, documents=docs, metadatas=metas)

    summary = build_summary(pages) if make_summary else ""
    summary_col.upsert(
        ids=[filename],
        documents=[summary or "(summary nahi bani)"],
        metadatas=[{"file": filename, "pages": len(pages), "has_summary": bool(summary)}],
    )
    return {"chunks": len(docs), "pages": len(pages), "summary": bool(summary)}


def search(query, n=5, file=None):
    kwargs = {"query_texts": [query], "n_results": n}
    if file:
        kwargs["where"] = {"file": file}
    res = chunks_col.query(**kwargs)
    hits = []
    for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
        hits.append({"text": doc, "file": meta["file"], "page": meta["page"]})
    return hits


def list_files():
    res = summary_col.get()
    return sorted(m["file"] for m in res["metadatas"])


def get_summaries(file=None):
    res = summary_col.get(where={"file": file}) if file else summary_col.get()
    out = []
    for doc, meta in zip(res["documents"], res["metadatas"]):
        if meta.get("has_summary"):
            out.append((meta["file"], doc))
    return out


if __name__ == "__main__":
    if sys.argv[1] == "index":
        pages_limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
        print(index_file(sys.argv[2], max_pages=pages_limit))
    elif sys.argv[1] == "search":
        for h in search(sys.argv[2]):
            print(f"[{h['file']} | page {h['page']}]")
            print(h["text"][:200], "\n")





