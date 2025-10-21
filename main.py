import matplotlib.pyplot as plt
from src.retrieval import retrieve
from src.generation_referenced import reference_based_generation
from src.generation_retrieved import generator_using_retrieval
from evaluation import compute_retrieval_metrics, compute_generation_metrics
from src.summarizer import summarizer_func
from dotenv import load_dotenv
import os
load_dotenv()


# --- Plotting Function ---
def plot_metrics(retrieval_metrics, generation_metrics_ref, generation_metrics_rag):
    categories = ['Retrieval', 'Reference-Based Gen', 'RAG']

    # Safely extract values, defaulting to 0 if missing
    retrieval_precision = retrieval_metrics.get('precision@k', 0)
    retrieval_recall = retrieval_metrics.get('recall@k', 0)

    ref_bleu = generation_metrics_ref.get('BLEU', 0)
    ref_rouge = generation_metrics_ref.get('ROUGE-L', 0)

    rag_bleu = generation_metrics_rag.get('BLEU', 0)
    rag_rouge = generation_metrics_rag.get('ROUGE-L', 0)

    # Prepare normalized data for plotting (aligning retrieval/generation metrics)
    precision_like = [retrieval_precision, ref_bleu, rag_bleu]
    recall_like = [retrieval_recall, ref_rouge, rag_rouge]

    plt.figure(figsize=(9, 6))
    x = range(len(categories))
    plt.bar(x, precision_like, width=0.35, label='Precision / BLEU', align='center')
    plt.bar([p + 0.35 for p in x], recall_like, width=0.35, label='Recall / ROUGE-L', align='center')

    plt.xticks([p + 0.17 for p in x], categories)
    plt.ylabel("Score")
    plt.title("Evaluation Metrics Comparison")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()


# --- Query ---
query = "Which California state parks allow dogs on the beach?"
top_k = 5

# --- Task A: Retrieval ---
retrieved = retrieve(query, top_k)
gold_ids = [
    '7585085659962d72-2-2011', '7585085659962d72-1588-3763', '7585085659962d72-3341-5260',
    '7585085659962d72-4869-6772', '7585085659962d72-6288-8343', '7585085659962d72-7960-9788',
    '7585085659962d72-9393-11229', '7585085659962d72-10768-12728', '7585085659962d72-12303-14246',
    '7585085659962d72-13894-15841', '7585085659962d72-15346-17352', '7585085659962d72-16947-18818',
    '7585085659962d72-18419-20343', '7585085659962d72-19996-21870', '7585085659962d72-21488-23380',
    '7585085659962d72-23020-24857', '7585085659962d72-24424-26283', '7585085659962d72-25922-27831',
    '7585085659962d72-27401-29383', '7585085659962d72-28994-30805', '7585085659962d72-30428-32216',
    '7585085659962d72-31801-33736', '7585085659962d72-33378-35235', '7585085659962d72-34825-36748',
    '7585085659962d72-36398-36895'
]
retrieval_metrics = compute_retrieval_metrics(retrieved, gold_ids)

# --- Task B: Generation with Reference (Reference-Based Gen) ---
with open(os.getenv('datafile'), "r", encoding='utf-8') as f:
    data = f.read()

reference_answer = summarizer_func(data)
generated_using_references = reference_based_generation(query)
generation_metrics_ref = compute_generation_metrics(generated_using_references, reference_answer)

# --- Task C: Generation with Retrieved Passages (RAG) ---
generated_using_retrieval = generator_using_retrieval(query, top_k)
generation_metrics_rag = compute_generation_metrics(generated_using_retrieval, reference_answer)

# --- Print Results ---
print("Task A — Retrieval:\n", retrieved, "\n")
print("Retrieval Metrics:", retrieval_metrics, "\n")

print("Task B — Generation w/ Reference:\n", generated_using_references, "\n")
print("Generation (Reference) Metrics:", generation_metrics_ref, "\n")

print("Task C — Generation w/ Retrieval:\n", generated_using_retrieval, "\n")
print("Generation (RAG) Metrics:", generation_metrics_rag, "\n")

# --- Plot Results ---
plot_metrics(retrieval_metrics, generation_metrics_ref, generation_metrics_rag)
