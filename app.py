import os

import streamlit as st

from index import index_file, list_files
from retriever import answer

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(page_title="Chat with your notes", page_icon="📚")
st.title("📚 Chat with your notes")

with st.sidebar:
    st.header("upload your notes")
    uploaded = st.file_uploader(
        "PDF or photo", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True
    )
    make_summary = st.checkbox(
        "make summary and overview ",
        value=True,
        help="Isse 'overview do' jaise sawal ka jawab milta hai. Badi PDF me time aur API calls lagti hain.",
    )
    if st.button("Index karo") and uploaded:
        for f in uploaded:
            path = os.path.join(UPLOAD_DIR, f.name)
            with open(path, "wb") as out:
                out.write(f.getbuffer())
            try:
                with st.spinner(f"{f.name} processing...."):
                    res = index_file(path, make_summary=make_summary)
                st.success(f"{f.name}: {res['pages']} pages, {res['chunks']} chunks")
            except Exception as e:
                st.error(f"{f.name}: {e}")

    files = list_files()
    choice = st.selectbox("which file are you asking about?", ["all files"] + files)

if "messages" not in st.session_state:
    st.session_state.messages = []


def show_sources(sources):
    if not sources:
        return
    with st.expander("Sources"):
        for s in sources:
            if s["page"] is None:
                st.write(f"{s['file']} (overview summary)")
            else:
                st.write(f"{s['file']} — page {s['page']}")


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        show_sources(m.get("sources"))

if question := st.chat_input("Ask about your notes"):
    history = [
        {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
    ]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking...."):
                result = answer(
                    question, history, file=None if choice == "all files" else choice
                )
            st.markdown(result["answer"])
            show_sources(result["sources"])
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                }
            )
        except Exception as e:
            st.error(f"Error: {e}")



           

           