Perfect — here’s an **updated, realistic, presentation-ready README.md** for your current project setup.

This version:
✅ Matches your **actual code flow** (no separate plotting file/folder)
✅ Lists only your **used metrics**: Recall@K, Precision@K, BLEU, ROUGE-L
✅ Includes an **ASCII-style architecture flow diagram** (so it looks great even on GitHub)
✅ Keeps the structure clean and professional

---

# 🧠 Multi-Turn RAG (Retrieval-Augmented Generation) Evaluator

This project implements a **multi-turn RAG evaluation workflow**, combining **retrieval** and **generation** modules to assess how effectively a language model can generate factual, context-grounded answers.

The pipeline performs:

* Passage **retrieval** using **FAISS** or **Chroma**
* Text **generation** using **Gemini API**
* Metric-based **evaluation** (Recall@K, Precision@K, BLEU, ROUGE-L)

It includes 3 evaluation tasks covering both retrieval and generation aspects of RAG systems.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture Flow](#architecture-flow)
3. [Project Structure](#project-structure)
4. [Tasks](#tasks)
5. [Setup & Installation](#setup--installation)
6. [Running the Workflow](#running-the-workflow)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Outputs](#outputs)
9. [Example Run](#example-run)
10. [Future Work](#future-work)

---

## 🧩 Overview

**Retrieval-Augmented Generation (RAG)** enhances LLMs by grounding their answers in external documents.
In **multi-turn** settings, user queries depend on previous context, requiring strong retrieval and reasoning.

This project:

* Builds and indexes a text corpus
* Retrieves relevant passages per query
* Generates factual responses using Gemini
* Evaluates performance using objective metrics

---

## ⚙️ Architecture Flow

```
                 ┌─────────────────────────────┐
                 │      Input Queries          │
                 │  (multi-turn conversations) │
                 └──────────────┬──────────────┘
                                │
                                ▼
                 ┌───────────────────────────┐
                 │   Corpus Preparation      │
                 │ (clean JSONL -> corpus)   │
                 └──────────────┬────────────┘
                                │
                                ▼
                 ┌───────────────────────────┐
                 │    Embedding + Indexing   │
                 │ (FAISS / Chroma Index)    │
                 └──────────────┬────────────┘
                                │
                                ▼
                 ┌───────────────────────────┐
                 │   Retrieval (Task A)      │
                 │ Return top-k passages     │
                 └──────────────┬────────────┘
                                │
                                ▼
                 ┌─────────────────────────────┐
                 │  Generation                 │
                 │  Task B: Gold references    │
                 │  Task C: Retrieved passages │
                 └──────────────┬──────────────┘
                                │
                                ▼
                 ┌───────────────────────────┐
                 │   Evaluation Metrics      │
                 │ Recall@K, Precision@K,    │
                 │ BLEU, ROUGE-L             │
                 └──────────────┬────────────┘
                                │
                                ▼
                 ┌───────────────────────────┐
                 │  Printed Results Summary  │
                 │  + Graph (if applicable)  │
                 └───────────────────────────┘
```

---

## 🏗️ Project Structure

```
MTRagEval/
│
├── data/                          # Raw source datasets
│   ├── cloud.jsonl
│   ├── fiqa.jsonl
│   └── govt.jsonl
│
├── data_processed/                # Cleaned / merged corpus
│   ├── corpus.jsonl
│   └── corpus copy.jsonl
│
├── index_faiss/                   # FAISS index and metadata
│   ├── corpus.index
│   └── metadata.pkl
│
├── src/                           # Core project source code
│   ├── build_index.py             # Builds vector index from processed corpus
│   ├── generation_referenced.py   # Task B: Generation using gold references
│   ├── generation_retrieved.py    # Task C: Generation using retrieved passages (RAG)
│   ├── ingest_and_clean.py        # Cleans and preprocesses raw data
│   ├── retrieval.py               # Task A: Retrieval logic (FAISS or Chroma)
│   └── summarizer.py              # Summarizes long retrieved texts
│
├── .env                           # Environment variables (Gemini API key)
├── .gitignore                     # Git ignore rules
├── data.txt                       # Text file used as reference context
├── evaluation.py                  # Metric computations (Recall, Precision, BLEU, ROUGE)
├── LICENSE                        # License info
├── main.py                        # Main script to run Tasks A, B, and C + metrics plotting
├── README.md                      # Project documentation
└── test.py                        # Script for debugging or data inspection

```

---

## 🧠 Tasks

| **Task** | **Name**               | **Goal**                                   | **Input**             | **Output**      |
| -------- | ---------------------- | ------------------------------------------ | --------------------- | --------------- |
| **A**    | Retrieval              | Retrieve top-K most relevant passages      | Query                 | Ranked passages |
| **B**    | Generation (Reference) | Generate using **gold** reference passages | Query + Gold passages | Generated text  |
| **C**    | Generation (RAG)       | Full retrieval + generation pipeline       | Query only            | Generated text  |

---

## 🧰 Setup & Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Niall1985/MTRagEval.git
cd MTRAG-Evaluator
```

### 2️⃣ Create a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Mac/Linux
```

### 3️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

### 4️⃣ Add Your Gemini API Key

Create a `.env` file in the root:

```
gemini_api=YOUR_GEMINI_API_KEY
```

---

## ▶️ Running the Workflow

### Step 1 — Prepare Corpus

Ensure you have a valid `corpus.jsonl` file:

```bash
python ingest_and_clean.py
```

### Step 2 — Build Index

You can use **FAISS** or **Chroma**.

```bash
python build_index.py
```

### Step 3 — Run Evaluation

```bash
python main.py
```

This will:

* Perform retrieval (Task A)
* Generate using references (Task B)
* Run full RAG generation (Task C)
* Compute all metrics and print them to the console

---

## 📊 Evaluation Metrics

| **Metric**      | **Type**   | **Meaning**                                                |
| --------------- | ---------- | ---------------------------------------------------------- |
| **Recall@K**    | Retrieval  | Fraction of relevant docs retrieved in top-K               |
| **Precision@K** | Retrieval  | Fraction of retrieved docs that are relevant               |
| **BLEU-4**      | Generation | N-gram overlap between generated and reference text        |
| **ROUGE-L**     | Generation | Longest common subsequence between generated and reference |

---

## 📤 Outputs

After execution, you will see printed outputs such as:

```
Retrieval Evaluation:
  Recall@5: 0.82
  Precision@5: 0.74

Generation Evaluation (Reference):
  BLEU-4: 0.68
  ROUGE-L: 0.72

Generation Evaluation (RAG):
  BLEU-4: 0.63
  ROUGE-L: 0.69
```

If graph plotting is enabled in your code (e.g. via `matplotlib`), it will display inline — no separate `plots/` folder is needed.

---

## 🧪 Example Run

**User Query:**

> “Which California state parks allow dogs on the beach?”

**Task A (Retrieval)**
→ Returns top 5 passages from the corpus related to dog-friendly beaches.

**Task B (Generation w/ Reference)**

> “Dogs are allowed in most California state parks if kept on a leash not exceeding six feet...”

**Task C (Full RAG)**

> “Some California state parks like Huntington and Fort Funston allow dogs in designated beach areas when leashed.”

---

## 🚀 Future Work

* Add **nDCG** and **MRR** for deeper retrieval analysis
* Integrate **reranking** models like `bge-reranker`
* Extend to **multilingual corpora**
* Add **human evaluation interface**
* Compare Gemini with **OpenAI GPT or Mistral models**

---

## 👤 Author

**Niall D’cunha**
RAG Evaluation Workflow Developer
