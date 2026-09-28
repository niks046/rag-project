"""
app.py
------
Same API surface as the other versions: /upload, /query, /reset, /health,
plus the static frontend. Backed entirely by free components (fastembed +
ChromaDB + Groq) -- see rag_engine.py for details.

Run:
    uvicorn app:app --reload --port 8000

Then open http://localhost:8000
"""

import os
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()  # reads a local .env file if present

from rag_engine import RAGEngine

app = FastAPI(title="Free RAG Project API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path(__file__).parent / "uploaded_docs"
UPLOAD_DIR.mkdir(exist_ok=True)

# Created once at startup. Raises immediately with a clear error if
# GROQ_API_KEY is missing -- fail fast, not on the first request.
engine = RAGEngine()


class QueryRequest(BaseModel):
    query: str
    top_k: int = 4


@app.get("/")
def serve_frontend():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


@app.get("/health")
def health():
    return {"status": "ok", **engine.stats()}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".txt", ".pdf", ".md"):
        raise HTTPException(400, "Only .txt, .md, and .pdf files are supported.")

    dest = UPLOAD_DIR / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        num_chunks = engine.ingest_file(dest)
    except Exception as e:
        raise HTTPException(500, f"Failed to process file: {e}")

    stats = engine.stats()
    return {
        "filename": file.filename,
        "chunks_added": num_chunks,
        "total_chunks_in_index": stats["total_chunks"],
    }


@app.post("/query")
def query(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty.")
    return engine.answer(req.query, top_k=req.top_k)


@app.post("/reset")
def reset_index():
    engine.reset()
    return {"status": "index cleared"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
