import json
import re
from pathlib import Path
from bs4 import BeautifulSoup
from tqdm.auto import tqdm

DATA_DIR = Path("../data")
OUTPUT_FILE = Path("../data_processed/corpus.jsonl")

def clean_text(html_text):
    text = BeautifulSoup(html_text or "", "html.parser").get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def ingest():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as fout:
        for f in sorted(DATA_DIR.glob("*.jsonl")):
            with f.open("r", encoding="utf-8") as fin:
                for line in fin:
                    doc = json.loads(line)
                    cleaned = {
                        "id": doc.get("document_id") or doc.get("id"),
                        "title": doc.get("title", ""),
                        "ur;": doc.get("url", ""),
                        "content": clean_text(doc.get("text", ""))
                    }
                    fout.write(json.dumps(cleaned, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    ingest()
    print("Done. Processed corpus written to", OUTPUT_FILE)