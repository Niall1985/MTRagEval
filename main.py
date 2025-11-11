import streamlit as st
import matplotlib.pyplot as plt
import json
import os
import uuid
import pandas as pd
from src.retrieval import retrieve
from src.generation_referenced import reference_based_generation
from src.generation_retrieved import generator_using_retrieval
from evaluation import compute_retrieval_metrics, compute_generation_metrics
from src.summarizer import summarizer_func
from dotenv import load_dotenv
from run_retrieval_eval import run_retrieval_evaluation

# --- Load environment variables ---
load_dotenv()

# Read output directory path from .env
OUTPUT_DIR = os.getenv("OUTPUT_JSONL_DIR", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Streamlit setup ---
st.set_page_config(page_title="RAG Evaluation Dashboard", layout="wide")
st.title("📊 RAG Evaluation Dashboard")
st.markdown(
    "This app compares **Retrieval**, **Reference-based Generation**, and **RAG** outputs "
    "with metrics and visualization. JSONL results are automatically saved to your environment folder."
)

# --- Inputs ---
query = st.text_input("Enter your query:", placeholder="Type your question...")

query_id = st.text_input("Enter query id: ", placeholder="Enter your query id...")

top_k = st.slider("Select Top K retrievals:", min_value=1, max_value=10, value=5, step=1)

gold_ids_input = st.text_area(
    "Enter Gold Passage IDs (comma or newline separated):",
    placeholder="e.g. 7585085659962d72-2-2011, 7585085659962d72-1588-3763"
)

gold_ids = []
if gold_ids_input.strip():
    gold_ids = [gid.strip() for gid in gold_ids_input.replace("\n", ",").split(",") if gid.strip()]

# --- Run button ---
if st.button("Run Evaluation"):
    if not query.strip():
        st.warning("⚠️ Please enter a query before running the evaluation.")
    elif not gold_ids:
        st.warning("⚠️ Please enter at least one Gold Passage ID.")
    else:
        with st.spinner("Running RAG evaluation pipeline... please wait ⏳"):

            # --- Generate unique query_id ---
            query_id = f"query_{query_id}"

            # --- Task A: Retrieval ---
            retrieved = retrieve(query, top_k)
            retrieval_metrics = compute_retrieval_metrics(retrieved, gold_ids)

            retrieval_output = {
                "task_id": query_id,
                "query": query,
                "contexts": [
                    {
                        "document_id": r.get("id", ""),
                        "source": "local_corpus",
                        "score": r.get("score", 0),
                        "text": r.get("content", ""),
                        "title": r.get("title", "")
                    }
                    for r in retrieved
                ],
                "Collection": "custom-eval"
            }

            retrieval_file = os.path.join(OUTPUT_DIR, "retrieval_predictions.jsonl")
            # Append new retrieval output
            with open(retrieval_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(retrieval_output) + "\n")

            # --- Task B: Reference-based generation ---
            corpus_path = os.getenv("corpus_file")
            gold_contents = []
            if os.path.exists(corpus_path):
                with open(corpus_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                            if record.get("id") in gold_ids:
                                gold_contents.append(record.get("content", ""))
                        except json.JSONDecodeError as e:
                            print(f"⚠️ Skipping invalid JSON line: {line[:80]}... ({e})")

            if gold_contents:
                data = "\n".join(gold_contents)
                reference_answer = summarizer_func(data)
            else:
                st.warning("⚠️ No matching IDs found in corpus.jsonl.")
                reference_answer = ""

            reference_answer = summarizer_func(reference_answer)
            generated_using_references = reference_based_generation(query, reference_answer)
            generation_metrics_ref = compute_generation_metrics(
                generated_using_references, reference_answer
            )

            # --- Task C: RAG generation ---
            generated_using_retrieval = generator_using_retrieval(query, top_k)
            generation_metrics_rag = compute_generation_metrics(
                generated_using_retrieval, reference_answer
            )

            # --- Save Generation Results ---
            generation_output = {
                "task_id": query_id,
                "query": query,
                "answers": [reference_answer],
                "predictions": [
                    {"text": generated_using_references},
                    {"text": generated_using_retrieval}
                ]
            }

            generation_file = os.path.join(OUTPUT_DIR, "generation_predictions.jsonl")
            # Append new generation output
            with open(generation_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(generation_output) + "\n")

        # --- Display Results ---
        st.success("✅ Evaluation Completed! JSONL results saved successfully.")
        st.write(f"📁 Saved files inside: `{OUTPUT_DIR}`")
        st.write(f"- Retrieval results → `{retrieval_file}`")
        st.write(f"- Generation results → `{generation_file}`")

        # --- Task Outputs ---
        st.subheader("📘 Retrieval")
        for r in retrieved:
            with st.expander(f"📄 {r['title']} — (Score: {r['score']:.3f})"):
                st.write(r["content"])
        st.write("**Retrieval Metrics:**", retrieval_metrics)

        st.subheader("🧠 Reference-based Generation")
        st.info(generated_using_references)
        st.write("**Metrics:**", generation_metrics_ref)

        st.subheader("🔍 RAG-based Generation")
        st.info(generated_using_retrieval)
        st.write("**Metrics:**", generation_metrics_rag)

        # --- Metrics Visualization ---
        st.subheader("📊 Metrics Comparison")
        categories = ["Retrieval", "Reference-Based Gen", "RAG"]
        precision_like = [
            retrieval_metrics.get("precision@k", 0),
            generation_metrics_ref.get("BLEU", 0),
            generation_metrics_rag.get("BLEU", 0),
        ]
        recall_like = [
            retrieval_metrics.get("recall@k", 0),
            generation_metrics_ref.get("ROUGE-L", 0),
            generation_metrics_rag.get("ROUGE-L", 0),
        ]

        fig, ax = plt.subplots(figsize=(3.2, 2.2))
        x = range(len(categories))
        ax.bar(x, precision_like, width=0.35, label="Precision / BLEU")
        ax.bar([p + 0.35 for p in x], recall_like, width=0.35, label="Recall / ROUGE-L")
        ax.set_xticks([p + 0.17 for p in x])
        ax.set_xticklabels(categories, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Score", fontsize=9)
        ax.set_title("Evaluation Metrics Comparison", fontsize=10)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        plt.tight_layout(pad=1.0)
        st.pyplot(fig, use_container_width=False)

        # --- Run Retrieval Evaluation ---
        results = run_retrieval_evaluation(retrieval_file, "my_eval_results.jsonl")

        st.subheader("📘 Retrieval Metrics")
        st.write(results["global_scores"])
        st.write("Weighted Average:", results["weighted_avg"])
