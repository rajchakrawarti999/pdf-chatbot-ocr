from index import get_summaries, search
from llm import ask_llm, LANG_RULE


def format_history(history):
    return "\n".join(f"{m['role']}: {m['content']}" for m in history) or "(none)"


def route(question, history):
    """Question broad hai ya specific, aur follow-up ko standalone question me badlo."""
    prompt = f"""Classify the user's question about a document and rewrite it.

Reply in exactly 2 lines:
Line 1: BROAD or SPECIFIC
Line 2: the question rewritten so it is fully understandable without the chat history (same language as the question)

BROAD = asks for an overview, summary, what the document is about, list of topics, or what is covered overall.
SPECIFIC = asks about a particular concept, fact or detail.

Chat history:
{format_history(history[-4:])}

Question: {question}"""
    try:
        lines = [l.strip() for l in ask_llm(prompt).strip().splitlines() if l.strip()]
        kind = "broad" if lines and lines[0].upper().startswith("BROAD") else "specific"
        rewritten = lines[1] if len(lines) > 1 else question
    except Exception:
        kind, rewritten = "specific", question
    return kind, rewritten


def answer(question, history=None, file=None):
    history = history or []
    kind, q = route(question, history)

    # Broad question: stored summary se answer
    if kind == "broad":
        summaries = get_summaries(file)
        if summaries:
            context = "\n\n".join(f"[{f}]\n{t}" for f, t in summaries)
            prompt = f"""You are a study assistant. Using ONLY the document summaries below, answer the question.
{LANG_RULE}

Summaries:
{context}

Question: {q}"""
            return {
                "answer": ask_llm(prompt),
                "sources": [{"file": f, "page": None} for f, _ in summaries],
                "mode": "overview",
            }

    # Specific question (ya summary available nahi): normal RAG
    hits = search(q, n=15 if kind == "broad" else 5, file=file)
    if not hits:
        return {
            "answer": "Notes me kuch nahi mila. Pehle koi file index karo.",
            "sources": [],
            "mode": "none",
        }

    context = "\n\n".join(f"[{h['file']} | page {h['page']}]\n{h['text']}" for h in hits)
    prompt = f"""You are a study assistant. Answer ONLY using the notes below.
If the answer is not in the notes, say you could not find it in the notes.
Mention page numbers like (p.12) where useful.
{LANG_RULE}

Chat history:
{format_history(history[-4:])}

Notes:
{context}

Question: {q}"""

    seen, sources = set(), []
    for h in hits:
        key = (h["file"], h["page"])
        if key not in seen:
            seen.add(key)
            sources.append({"file": h["file"], "page": h["page"]})

    return {"answer": ask_llm(prompt), "sources": sources, "mode": "rag"}