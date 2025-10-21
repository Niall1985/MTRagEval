#Task A

import json, faiss, numpy as np, pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
load_dotenv()

INDEX_DIR = Path(os.getenv('index_dir'))  # fixed path
CORPUS_FILE = Path(os.getenv('corpus_file'))

model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
index = faiss.read_index(str(INDEX_DIR / "corpus.index"))

with open(INDEX_DIR / "metadata.pkl", "rb") as f:
    id_map = pickle.load(f)

# ✅ Safe JSONL load
docs = []
with CORPUS_FILE.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        docs.append(json.loads(line))

def build_query(conversation, n_turns=3):
    ctx = conversation[-n_turns:]
    return " ".join([f"{turn['role']}: {turn['text']}" for turn in ctx])

def retrieve(query, k):
    q_emb = model.encode([query], normalize_embeddings=True)
    D, I = index.search(q_emb.astype("float32"), k)
    hits = []
    for idx, score in zip(I[0], D[0]):
        if idx < 0: continue
        doc = docs[idx]
        hits.append({
            "id": doc.get("id", ""),
            "title": doc.get("title", ""),
            "content": doc.get("content", ""),
            "score": float(score)
        })
    return hits
