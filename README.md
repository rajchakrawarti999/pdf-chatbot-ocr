# Chat With Your Notes (OCR + RAG)

I was learning about OCR and wanted to build something real with it instead of just running Tesseract on a random image. This is what I ended up with: you upload your notes (a photo or a PDF), and then you can chat with them. The app answers from your own notes and shows which file and which page the answer came from.

It works with English and Hindi text, and you can ask questions in English or Hinglish.

## What it can do

- Upload PDFs and images (png, jpg, jpeg)
- Reads digital PDFs directly and only uses OCR for scanned pages
- Splits the text into chunks and stores them in a local vector database
- Answers questions using only the retrieved notes, with file and page number as the source
- Handles overview type questions like "give me an overview of this PDF" (more on this below)
- Follow-up questions work, because the question is rewritten using the chat history
- Lets you pick one file or search across all uploaded files

## How it works

There are two phases.

```
Indexing (when you upload a file)

  PDF / image -> text (direct or OCR) -> clean -> chunks -> embeddings -> ChromaDB
                                                    \
                                                     -> summary of the document -> ChromaDB

Querying (every question)

  question -> router (broad or specific?)
      broad    -> use the saved summary -> LLM -> answer
      specific -> search top chunks in ChromaDB -> LLM -> answer + sources
```

## The problem I had to solve

My first version was a normal RAG: take the question, fetch the top 5 chunks, give them to the LLM. It worked fine for specific questions like "what is gradient descent?". But when I uploaded a PDF I had never read and asked "what is this document about?", it failed. The question does not match any single chunk, so the retrieved chunks were random and the LLM said it could not find anything.

How I fixed it:

1. When a file is uploaded, I build a summary of the whole document (summarize each section, then summarize those summaries) and store it in a separate collection.
2. When a question comes in, a small router step asks the LLM if the question is broad or specific. Broad questions are answered from the saved summary. Specific questions go through the normal retrieval.

Other things I changed along the way:

- **Hybrid text extraction.** My first version ran OCR on every page, which was very slow on a long book that already had selectable text. Now it tries to read the text directly and falls back to OCR only if a page has almost no text.
- **Smaller chunks.** I started with 500 word chunks, but the embedding model only reads around 128 tokens, so the rest was being ignored. Chunks are now 150 words with a 30 word overlap.
- **Old data cleanup.** Re-indexing the same file used to leave old chunks behind. It now deletes the old chunks first.
- **Retries.** The Gemini API sometimes returns 503 when it is busy, so the call is retried a few times with a delay.
- **Follow-up questions.** A question like "explain it more" has no meaning for the search. The router rewrites it into a full question using the last few messages.

## Tech stack

- Python
- EasyOCR (English + Hindi) and OpenCV for OCR and image cleanup
- PyMuPDF for reading PDFs
- sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2) for embeddings
- ChromaDB as the local vector database
- Gemini API for the LLM
- Streamlit for the UI

Only the LLM needs an API key and internet. OCR, embeddings and the database all run on the laptop.

## Project structure

```
app.py            Streamlit UI (upload, index button, chat)
ocr.py            text extraction: direct text for digital PDFs, EasyOCR for scans and images
index.py          chunking, embeddings, ChromaDB, summary generation, search
retriever.py      router, prompt building and final answer
llm.py            the only file that talks to the LLM API
Requirements.txt  dependencies
```

`llm.py` is kept separate on purpose. If I want to switch to another provider or a local model, that is the only file I need to touch.

## Setup

1. Clone the repo and go into the folder.

```
git clone https://github.com/rajchakrawarti999/ocr-rag-notes-chat.git
cd ocr-rag-notes-chat
```

2. Create a virtual environment and install the packages (tested on Python 3.12).

```
python -m venv venv
venv\Scripts\activate
pip install -r Requirements.txt
```

On Mac or Linux the activate command is `source venv/bin/activate`.

3. Get a free API key from https://aistudio.google.com and put it in a `.env` file in the project folder. You can copy `.env.example` and rename it.

```
GEMINI_API_KEY=your_key_here
```

4. Run the app.

```
streamlit run app.py
```

Upload a PDF or image from the sidebar, click "Index karo", and start asking questions.

The first run downloads the embedding model and the EasyOCR models, so it takes a few minutes.

## Testing without the UI

These are useful for checking each part separately.

```
python ocr.py yourfile.pdf 5                 # extract text from the first 5 pages
python index.py index yourfile.pdf 30        # index the first 30 pages
python index.py search "your question"       # see which chunks come back
```

## Things to know

- The free tier of the Gemini API has a small daily limit per model. Every question uses two LLM calls (router and answer), and building a summary uses one call per section plus one. I hit the limit while testing.
- Model names change over time. If you get a "model not found" error, set a different model with the `LLM_MODEL` environment variable or edit `llm.py`.
- Summaries on very large books only read the first 12000 words of each section, so they are approximate.
- OCR on scanned pages runs on the CPU, so it is slow for long documents. I tested mostly on short PDFs.
- Hinglish support is basic. The answer language follows the question, but I have not measured how well the embedding model retrieves Hindi written in English letters.

## What I want to add next

- Run a local LLM with Ollama so there is no API limit
- A separate collection for each user so uploads do not get mixed
- A FastAPI backend and a deployed demo
- Better handling of tables and equations in PDFs
