"""
rag_engine.py  (free version)
------------------------------
Every component here is genuinely free, with no billing setup required:

  Embeddings   -> fastembed (runs locally, ONNX-based, no PyTorch,
                  ~70MB model download, works fine on CPU)
  Vector store -> ChromaDB (runs locally, stores to disk, no signup)
  Generation   -> Groq API (free tier, no credit card required,
                  serves open models like Llama 3 very fast)

Why fastembed instead of sentence-transformers: sentence-transformers
pulls in PyTorch as a dependency, which is a multi-GB install and was
the actual source of your earlier install/version problems. fastembed
does the same embedding job using ONNX Runtime instead -- much smaller,
much faster to install, no GPU/CPU compatibility headaches.

Required environment variable:
  GROQ_API_KEY   (free -- get one at https://console.groq.com/keys)
"""

import os
import re
import uuid
from pathlib import Path
from typing import List, Dict

import chromadb
from fastembed import TextEmbedding
from groq import Groq
from pypdf import PdfReader

# OCR fallback deps -- only imported/used when normal text extraction
# comes up empty (i.e. the PDF is actually a set of images/graphics).
import fitz  # PyMuPDF -- renders PDF pages to images, pure pip install
import pytesseract
from PIL import Image
import io

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"   # small, fast, good quality
GROQ_MODEL = "openai/gpt-oss-20b"               # fast + free-tier friendly (current as of Sep 2026)
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 6

# If normal PDF text extraction returns fewer than this many characters
# per page on average, treat the PDF as image-based and fall back to OCR.
OCR_FALLBACK_THRESHOLD_CHARS_PER_PAGE = 40

STORAGE_DIR = Path(__file__).parent / "storage"
STORAGE_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Text extraction & chunking
# ---------------------------------------------------------------------------
def _ocr_pdf(file_path: Path) -> str:
    """
    Fallback for image-based PDFs: render each page to an image and run
    Tesseract OCR on it. This requires the Tesseract OCR *program* to be
    installed on the machine (not just a pip package) -- see README.
    """
    doc = fitz.open(str(file_path))
    all_text = []
    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            # Render at 2x zoom for better OCR accuracy on small text
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            page_text = pytesseract.image_to_string(img)
            all_text.append(page_text)
    except pytesseract.TesseractNotFoundError:
        raise RuntimeError(
            "OCR fallback triggered, but the Tesseract OCR program isn't "
            "installed (or isn't on your PATH). This PDF needs OCR because "
            "it has no real selectable text -- it's an image/graphic-based "
            "PDF. Install Tesseract: see README.md section 'Installing "
            "Tesseract for OCR (Windows)'."
        )
    finally:
        doc.close()
    return "\n".join(all_text)


def extract_text(file_path: Path) -> str:
    suffix = file_path.suffix.lower()

    if suffix in (".txt", ".md"):
        return file_path.read_text(encoding="utf-8", errors="ignore")
    elif suffix == ".pdf":
        reader = PdfReader(str(file_path))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)

        num_pages = len(reader.pages) or 1
        avg_chars_per_page = len(text.strip()) / num_pages

        if avg_chars_per_page < OCR_FALLBACK_THRESHOLD_CHARS_PER_PAGE:
            print(f"[RAGEngine] '{file_path.name}' has little/no extractable "
                  f"text ({avg_chars_per_page:.0f} chars/page) -- falling "
                  f"back to OCR...")
            text = _ocr_pdf(file_path)

        return text
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks


# ---------------------------------------------------------------------------
# The free RAG engine
# ---------------------------------------------------------------------------
class RAGEngine:
    def __init__(self):
        groq_key = os.environ.get("GROQ_API_KEY")
        if not groq_key:
            raise RuntimeError(
                "GROQ_API_KEY environment variable is not set. "
                "Get a free key at https://console.groq.com/keys"
            )
        self.groq = Groq(api_key=groq_key)

        print("[RAGEngine] Loading embedding model (fastembed, first run only)...")
        self.embedder = TextEmbedding(model_name=EMBED_MODEL_NAME)

        self.chroma_client = chromadb.PersistentClient(path=str(STORAGE_DIR))
        self.collection = self.chroma_client.get_or_create_collection(
            name="documents", metadata={"hnsw:space": "cosine"}
        )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def ingest_file(self, file_path: Path) -> int:
        text = extract_text(file_path)
        chunks = chunk_text(text)
        if not chunks:
            return 0

        embeddings = list(self.embedder.embed(chunks))
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [{"source": file_path.name} for _ in chunks]

        self.collection.add(
            ids=ids,
            embeddings=[e.tolist() for e in embeddings],
            documents=chunks,
            metadatas=metadatas,
        )
        return len(chunks)

    def reset(self):
        """Wipe the collection and start fresh."""
        self.chroma_client.delete_collection("documents")
        self.collection = self.chroma_client.get_or_create_collection(
            name="documents", metadata={"hnsw:space": "cosine"}
        )

    def stats(self) -> Dict:
        return {"total_chunks": self.collection.count()}

    # ------------------------------------------------------------------
    # Retrieval + generation
    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        if self.collection.count() == 0:
            return []

        query_embedding = list(self.embedder.embed([query]))[0].tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
        )

        retrieved = []
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            retrieved.append({
                "text": doc,
                "source": meta["source"],
                "score": round(1 - dist, 4),  # cosine distance -> similarity
            })
        return retrieved

    def answer(self, query: str, top_k: int = TOP_K) -> Dict:
        retrieved = self.retrieve(query, top_k=top_k)

        if not retrieved:
            return {
                "answer": "No documents have been ingested yet. Upload a "
                          "document first.",
                "sources": [],
            }

        context = "\n\n---\n\n".join(r["text"] for r in retrieved)
        prompt = (
            "Answer the question using only the context below.\n\n"
            "Rules:\n"
            "- If the context answers the question, answer it directly and "
            "concisely, and mention which section/heading of the document "
            "it came from if the context shows one (e.g. a title in caps, "
            "a bolded heading, or a labeled section name).\n"
            "- If the specific thing asked about (a term, concept, or topic) "
            "does NOT appear anywhere in the context, do not just say 'I "
            "don't know'. Instead, state what the document IS actually "
            "about (based on the context you can see) and clearly say it "
            "does not cover what was asked. For example: 'This document "
            "covers object-oriented programming concepts (classes, "
            "encapsulation, inheritance, etc.) and does not include the "
            "concept of transactions.'\n"
            "- Only say 'I don't know' in the rare case where you can't "
            "even tell what the document's general subject is from the "
            "context provided.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}"
        )

        resp = self.groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        answer_text = resp.choices[0].message.content.strip()

        return {
            "answer": answer_text,
            "sources": [
                {"source": r["source"], "score": r["score"], "preview": r["text"][:180]}
                for r in retrieved
            ],
        }
