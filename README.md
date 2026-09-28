# Free RAG Project — $0 cost, no credit card, anywhere

A complete RAG (Retrieval-Augmented Generation) system where every single
component is genuinely free — no billing setup, no card required, no
minimum spend, unlike API-metered services like OpenAI.

---

## What's free here, and why

| Component     | Tool                | Cost                          |
|---------------|----------------------|--------------------------------|
| Embeddings    | fastembed (local)   | $0 — runs on your machine, ONNX-based, no PyTorch needed |
| Vector store  | ChromaDB (local)    | $0 — stores to disk, no signup |
| Generation    | Groq API            | $0 — free tier, no card required, fast open models (Llama 3) |

**Why fastembed instead of sentence-transformers:** sentence-transformers
depends on PyTorch, a multi-GB install that's slow and version-fragile
(this is what caused the `faiss-cpu` version errors you may have hit
before). fastembed does the same embedding job on a much smaller ONNX
runtime — faster install, fewer compatibility headaches, same free cost.

**Why Groq instead of a local LLM:** running an LLM locally (like
flan-t5) works, but is slow and gives weaker answers on CPU-only
machines. Groq gives free API access to real, capable models (Llama 3.1)
with fast responses — no local model to download, no GPU needed, and
still $0 cost.

---

## 1. Get your free Groq API key

1. Go to https://console.groq.com/keys
2. Sign up (email or Google/GitHub login — no card anywhere in this flow)
3. Click "Create API Key", copy it

That's the only account you need to create for this whole project.

---

## 2. Run it locally

```bash
cd free_rag_project
cp .env.example .env
# open .env and paste your real GROQ_API_KEY

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

uvicorn app:app --reload --port 8000
```

Open http://localhost:8000, upload `sample_docs/rag_intro.txt`, and ask:
- "What are the three stages of a RAG pipeline?"
- "Why does RAG reduce hallucination?"

First run downloads the small fastembed model (~70MB, one-time,
much smaller/faster than the old sentence-transformers + torch setup).

---

## 3. (Optional) Deploy it for free too

Render's free Web Service tier works the same way as before:

1. Push this folder to a GitHub repo (`.env` won't be committed — it's
   in `.gitignore`).
2. On https://render.com → New + → Web Service → connect your repo.
3. Build command: `pip install -r requirements.txt`
   Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
   (already set in the `Procfile`)
4. Add one environment variable: `GROQ_API_KEY`
5. Deploy. You'll get a public URL, free, no card needed.

Same free-tier caveat as before: it sleeps after ~15 min idle and takes
30-50s to wake up on the next request.

---

## 4. Project structure

```
free_rag_project/
├── app.py              # FastAPI server
├── rag_engine.py         # chunking, fastembed, ChromaDB, Groq generation
├── requirements.txt
├── .env.example
├── Procfile              # for optional Render deployment
├── static/index.html      # chat UI
├── sample_docs/           # test document
└── storage/               # ChromaDB data, auto-created
```

---

## 5. OCR support for image-based PDFs (like scanned notes or infographics)

Some PDFs — especially hand-drawn notes, infographic-style slides, or
scanned documents — have no real selectable text at all; everything is
part of an image. `pypdf` can't extract text from those. This project
automatically detects that case (very little/no text extracted per page)
and falls back to **OCR** (Optical Character Recognition) using Tesseract.

**This needs one extra install that pip can't do for you**, because
Tesseract is a compiled program, not a Python package.

### Installing Tesseract for OCR (Windows)

1. Download the installer from
   https://github.com/UB-Mannheim/tesseract/wiki (look for the latest
   `tesseract-ocr-w64-setup-*.exe` link).
2. Run the installer. Note the install path it shows (default is usually
   `C:\Program Files\Tesseract-OCR`).
3. Add that folder to your Windows PATH:
   - Search "Environment Variables" in the Start menu → "Edit the system
     environment variables" → "Environment Variables" button.
   - Under "System variables", find `Path`, click Edit, click New, paste
     `C:\Program Files\Tesseract-OCR` (or wherever it installed).
   - Click OK on everything to save.
4. **Close and reopen** your terminal (PATH changes don't apply to
   already-open windows).
5. Verify it worked: run `tesseract --version` in a new terminal — it
   should print a version number, not "command not found".

### Installing Tesseract (Mac / Linux)

```bash
# Mac
brew install tesseract

# Ubuntu/Debian
sudo apt install tesseract-ocr
```

### Using it

Nothing else to do — just upload the PDF as normal. If the project
detects the PDF has little/no real text, it automatically switches to
OCR and prints a message in the terminal like:

```
[RAGEngine] 'yourfile.pdf' has little/no extractable text (12 chars/page)
-- falling back to OCR...
```

**Note on OCR quality:** OCR reads pixels, not structure — it does a
reasonable job on typed/printed text but won't preserve layout (columns,
arrows, boxes) and can struggle with heavily stylized handwriting or
very small text in dense infographics. For hand-drawn diagram-heavy PDFs,
expect decent-but-imperfect extraction, not a perfect transcription.

## 6. Things worth knowing about Groq's free tier

- Rate limits exist (requests per minute, tokens per day) — generous
  enough for building and demoing a portfolio project, but check
  https://console.groq.com/docs/rate-limits if you hit a 429 error.
- Models available change over time; if `llama-3.1-8b-instant` stops
  working, check https://console.groq.com/docs/models for the current
  list and swap the `GROQ_MODEL` constant in `rag_engine.py`.

---

## 6. What to say about this in an interview

- "Built a RAG pipeline entirely on free-tier infrastructure — local
  ONNX embeddings, a local vector store, and Groq's free inference API
  — to keep the whole thing runnable with zero operating cost."
- Good talking point: the deliberate choice of fastembed over
  sentence-transformers shows you think about deployment weight and
  dependency footprint, not just "does it work on my machine."
