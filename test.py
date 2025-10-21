# from src.retrieval import retrieve, build_query
# q = build_query([{"role":"user","text":"Which California state parks allow dogs on the beach?"}])
# print(retrieve(q, k=3))

import json
from pathlib import Path

CORPUS_FILE = Path("C:\\Users\\Niall Dcunha\\MTRAG-Evaluator\\data_processed\\corpus.jsonl")

matching_ids = []

with CORPUS_FILE.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        doc = json.loads(line)
        if doc.get("title", "").strip() == "Visiting State Parks With Your Dog":
            matching_ids.append(doc.get("id", None))

print("Matching document IDs:")
print(matching_ids)
