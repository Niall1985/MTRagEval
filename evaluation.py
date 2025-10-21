import fractions
# 🩹 Fix NLTK BLEU Fraction bug for Python 3.12+
if not hasattr(fractions.Fraction, "_normalize"):
    _old_fraction_new = fractions.Fraction.__new__

    def _fraction_new(cls, numerator=0, denominator=None, _normalize=True):
        # call the original Fraction constructor safely
        return _old_fraction_new(cls, numerator, denominator)

    fractions.Fraction.__new__ = staticmethod(_fraction_new)


from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
import numpy as np

nltk.download('punkt')

# -----------------------------
# 🧮 Task A Metrics (Retrieval)
# -----------------------------
def compute_retrieval_metrics(retrieved, gold_ids):
    retrieved_ids = [doc['id'] for doc in retrieved]
    k = len(retrieved_ids)

    hits = len([i for i in retrieved_ids if i in gold_ids])
    recall = hits / len(gold_ids) if gold_ids else 0
    precision = hits / k if k > 0 else 0
    return {"recall@k": recall, "precision@k": precision}

# -----------------------------
# 🧮 Task B/C Metrics (Generation)
# -----------------------------
def compute_generation_metrics(generated, reference):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_score = scorer.score(reference, generated)['rougeL'].fmeasure

    smoothie = SmoothingFunction().method4
    bleu = sentence_bleu(
        [reference.split()],
        generated.split(),
        smoothing_function=smoothie
    )

    return {"ROUGE-L": rouge_score, "BLEU": bleu}
