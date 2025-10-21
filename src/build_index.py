import json
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle

# --- Paths ---
CORPUS_FILE = Path("../data_processed/corpus.jsonl")
INDEX_DIR = Path("../index_faiss")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# --- Load model ---
print("🔹 Loading embedding model...")
model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")

# --- Read and clean corpus ---
print("🔹 Reading JSONL corpus...")
docs = []
with CORPUS_FILE.open("r", encoding="utf-8") as f:
    for line in f:
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue

        # fix key name
        url = d.get("url") or d.get("ur;") or ""
        text = d.get("content") or ""
        title = d.get("title") or "Untitled"
        doc_id = d.get("id")

        # skip empty
        if not text.strip():
            continue

        docs.append({
            "id": doc_id,
            "title": title,
            "url": url,
            "text": text
        })

print(f"✅ Loaded {len(docs)} documents")

# --- Embed documents ---
texts = [d["text"] for d in docs]
print("🔹 Generating embeddings...")
embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

# --- Build FAISS index ---
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)  # Inner product (cosine similarity with normalized vectors)
index.add(embeddings)
print(f"✅ FAISS index built with {index.ntotal} vectors (dim={dim})")

# --- Save index and metadata ---
faiss.write_index(index, str(INDEX_DIR / "corpus.index"))

with open(INDEX_DIR / "metadata.pkl", "wb") as f:
    pickle.dump(docs, f)

print(f"💾 Saved FAISS index and metadata to: {INDEX_DIR}")
