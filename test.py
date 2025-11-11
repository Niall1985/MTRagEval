# # from src.retrieval import retrieve, build_query
# # q = build_query([{"role":"user","text":"Which California state parks allow dogs on the beach?"}])
# # print(retrieve(q, k=3))

# import json
# from pathlib import Path

# CORPUS_FILE = Path("C:\\Users\\Niall Dcunha\\MTRAG-Evaluator\\data_processed\\corpus.jsonl")

# matching_ids = []

# with CORPUS_FILE.open("r", encoding="utf-8") as f:
#     for line in f:
#         line = line.strip()
#         if not line:
#             continue
#         doc = json.loads(line)
#         if doc.get("title", "").strip() == "Visiting State Parks With Your Dog":
#             matching_ids.append(doc.get("id", None))

# print("Matching document IDs:")
# print(matching_ids)




# # import matplotlib.pyplot as plt
# # from src.retrieval import retrieve
# # from src.generation_referenced import reference_based_generation
# # from src.generation_retrieved import generator_using_retrieval
# # from evaluation import compute_retrieval_metrics, compute_generation_metrics
# # from src.summarizer import summarizer_func
# # from dotenv import load_dotenv
# # import os
# # load_dotenv()


# # # --- Plotting Function ---
# # def plot_metrics(retrieval_metrics, generation_metrics_ref, generation_metrics_rag):
# #     categories = ['Retrieval', 'Reference-Based Gen', 'RAG']

# #     # Safely extract values, defaulting to 0 if missing
# #     retrieval_precision = retrieval_metrics.get('precision@k', 0)
# #     retrieval_recall = retrieval_metrics.get('recall@k', 0)

# #     ref_bleu = generation_metrics_ref.get('BLEU', 0)
# #     ref_rouge = generation_metrics_ref.get('ROUGE-L', 0)

# #     rag_bleu = generation_metrics_rag.get('BLEU', 0)
# #     rag_rouge = generation_metrics_rag.get('ROUGE-L', 0)

# #     # Prepare normalized data for plotting (aligning retrieval/generation metrics)
# #     precision_like = [retrieval_precision, ref_bleu, rag_bleu]
# #     recall_like = [retrieval_recall, ref_rouge, rag_rouge]

# #     plt.figure(figsize=(9, 6))
# #     x = range(len(categories))
# #     plt.bar(x, precision_like, width=0.35, label='Precision / BLEU', align='center')
# #     plt.bar([p + 0.35 for p in x], recall_like, width=0.35, label='Recall / ROUGE-L', align='center')

# #     plt.xticks([p + 0.17 for p in x], categories)
# #     plt.ylabel("Score")
# #     plt.title("Evaluation Metrics Comparison")
# #     plt.ylim(0, 1)
# #     plt.legend()
# #     plt.tight_layout()
# #     plt.show()


# # # --- Query ---
# # query = "Which California state parks allow dogs on the beach?"
# # top_k = 5

# # # --- Task A: Retrieval ---
# # retrieved = retrieve(query, top_k)
# # gold_ids = [
# #     '7585085659962d72-2-2011', '7585085659962d72-1588-3763', '7585085659962d72-3341-5260',
# #     '7585085659962d72-4869-6772', '7585085659962d72-6288-8343', '7585085659962d72-7960-9788',
# #     '7585085659962d72-9393-11229', '7585085659962d72-10768-12728', '7585085659962d72-12303-14246',
# #     '7585085659962d72-13894-15841', '7585085659962d72-15346-17352', '7585085659962d72-16947-18818',
# #     '7585085659962d72-18419-20343', '7585085659962d72-19996-21870', '7585085659962d72-21488-23380',
# #     '7585085659962d72-23020-24857', '7585085659962d72-24424-26283', '7585085659962d72-25922-27831',
# #     '7585085659962d72-27401-29383', '7585085659962d72-28994-30805', '7585085659962d72-30428-32216',
# #     '7585085659962d72-31801-33736', '7585085659962d72-33378-35235', '7585085659962d72-34825-36748',
# #     '7585085659962d72-36398-36895'
# # ]
# # retrieval_metrics = compute_retrieval_metrics(retrieved, gold_ids)

# # # --- Task B: Generation with Reference (Reference-Based Gen) ---
# # with open(os.getenv('datafile'), "r", encoding='utf-8') as f:
# #     data = f.read()

# # reference_answer = summarizer_func(data)
# # generated_using_references = reference_based_generation(query)
# # generation_metrics_ref = compute_generation_metrics(generated_using_references, reference_answer)

# # # --- Task C: Generation with Retrieved Passages (RAG) ---
# # generated_using_retrieval = generator_using_retrieval(query, top_k)
# # generation_metrics_rag = compute_generation_metrics(generated_using_retrieval, reference_answer)

# # # --- Print Results ---
# # print("Task A — Retrieval:\n", retrieved, "\n")
# # print("Retrieval Metrics:", retrieval_metrics, "\n")

# # print("Task B — Generation w/ Reference:\n", generated_using_references, "\n")
# # print("Generation (Reference) Metrics:", generation_metrics_ref, "\n")

# # print("Task C — Generation w/ Retrieval:\n", generated_using_retrieval, "\n")
# # print("Generation (RAG) Metrics:", generation_metrics_rag, "\n")

# # # --- Plot Results ---
# # plot_metrics(retrieval_metrics, generation_metrics_ref, generation_metrics_rag)



# import streamlit as st
# import matplotlib.pyplot as plt
# from src.retrieval import retrieve
# from src.generation_referenced import reference_based_generation
# from src.generation_retrieved import generator_using_retrieval
# from evaluation import compute_retrieval_metrics, compute_generation_metrics
# from src.summarizer import summarizer_func
# from dotenv import load_dotenv
# import os

# load_dotenv()

# # --- Streamlit Page Config ---
# st.set_page_config(page_title="RAG Evaluation Dashboard", layout="wide")

# st.title("📊 RAG Evaluation Dashboard")
# st.markdown(
#     "This app compares **Retrieval**, **Reference-based Generation**, and **RAG** outputs "
#     "with metrics and visualization."
# )

# # --- Query & Gold ID Input ---
# query = st.text_input(
#     "Enter your query:",
#     placeholder="Type your question..."
# )

# top_k = st.slider("Select Top K retrievals:", min_value=1, max_value=10, value=5, step=1)

# gold_ids_input = st.text_area(
#     "Enter Gold Passage IDs (comma or newline separated):",
#     placeholder="e.g. 7585085659962d72-2-2011, 7585085659962d72-1588-3763"
# )

# # Convert input string to list of IDs
# gold_ids = []
# if gold_ids_input.strip():
#     gold_ids = [gid.strip() for gid in gold_ids_input.replace("\n", ",").split(",") if gid.strip()]

# # --- Run Button ---
# if st.button("Run Evaluation"):
#     if not query.strip():
#         st.warning("⚠️ Please enter a query before running the evaluation.")
#     elif not gold_ids:
#         st.warning("⚠️ Please enter at least one Gold Passage ID.")
#     else:
#         with st.spinner("Running RAG evaluation pipeline... please wait ⏳"):
#             # --- Task A: Retrieval ---
#             retrieved = retrieve(query, top_k)
#             retrieval_metrics = compute_retrieval_metrics(retrieved, gold_ids)

#             import json

#             corpus_path = os.getenv('corpus_file')

#             gold_contents = []
#             if os.path.exists(corpus_path):
#                 with open(corpus_path, "r", encoding="utf-8") as f:
#                     for line in f:
#                         line = line.strip()
#                         if not line:
#                             continue  # skip empty lines
#                         try:
#                             record = json.loads(line)
#                             if record.get("id") in gold_ids:
#                                 gold_contents.append(record.get("content", ""))
#                         except json.JSONDecodeError as e:
#                             print(f"⚠️ Skipping invalid JSON line: {line[:80]}... ({e})")

#             if gold_contents:
#                 data = "\n".join(gold_contents)
#                 reference_answer = summarizer_func(data)
#             else:
#                 st.warning("⚠️ No matching IDs found in corpus.jsonl.")
#                 reference_answer = ""


#             # Continue with your usual summarization and generation steps
#             reference_answer = summarizer_func(reference_answer)
#             generated_using_references = reference_based_generation(query, reference_answer)
#             generation_metrics_ref = compute_generation_metrics(generated_using_references, reference_answer)
            
#             # --- Task C: Generation with Retrieved Passages (RAG) ---
#             generated_using_retrieval = generator_using_retrieval(query, top_k)
#             generation_metrics_rag = compute_generation_metrics(
#                 generated_using_retrieval, reference_answer
#             )

#         # --- Display Results ---
#         st.success("✅ Evaluation Completed!")

#         # --- Task A Output ---
#         st.subheader("📘 Task A — Retrieval")
#         st.write("**Retrieved Passages:**")
#         for r in retrieved:
#             with st.expander(f"📄 {r['title']} — (Score: {r['score']:.3f})"):
#                 st.write(r["content"])
#         st.write("**Retrieval Metrics:**", retrieval_metrics)

#         # --- Task B Output ---
#         st.subheader("🧠 Task B — Generation with Reference")
#         st.write("**Generated Answer (Reference-based):**")
#         st.info(generated_using_references)
#         st.write("**Generation (Reference) Metrics:**", generation_metrics_ref)

#         # --- Task C Output ---
#         st.subheader("🔍 Task C — Generation with Retrieved Passages (RAG)")
#         st.write("**Generated Answer (RAG):**")
#         st.info(generated_using_retrieval)
#         st.write("**Generation (RAG) Metrics:**", generation_metrics_rag)

#         # --- Plot Metrics ---
#         st.subheader("📊 Metrics Comparison")

#         categories = ['Retrieval', 'Reference-Based Gen', 'RAG']
#         retrieval_precision = retrieval_metrics.get('precision@k', 0)
#         retrieval_recall = retrieval_metrics.get('recall@k', 0)
#         ref_bleu = generation_metrics_ref.get('BLEU', 0)
#         ref_rouge = generation_metrics_ref.get('ROUGE-L', 0)
#         rag_bleu = generation_metrics_rag.get('BLEU', 0)
#         rag_rouge = generation_metrics_rag.get('ROUGE-L', 0)

#         precision_like = [retrieval_precision, ref_bleu, rag_bleu]
#         recall_like = [retrieval_recall, ref_rouge, rag_rouge]

#         # # Make the graph slightly smaller
#         # fig, ax = plt.subplots(figsize=(2.5, 1.8))
#         # x = range(len(categories))
#         # ax.bar(x, precision_like, width=0.35, label='Precision / BLEU')
#         # ax.bar([p + 0.35 for p in x], recall_like, width=0.35, label='Recall / ROUGE-L')
#         # ax.set_xticks([p + 0.17 for p in x])
#         # ax.set_xticklabels(categories)
#         # ax.set_ylabel("Score")
#         # ax.set_title("Evaluation Metrics Comparison")
#         # ax.set_ylim(0, 1)
#         # ax.legend()
#         # st.pyplot(fig)
#         fig, ax = plt.subplots(figsize=(3.2, 2.2))  # small but readable
#         x = range(len(categories))
#         ax.bar(x, precision_like, width=0.35, label='Precision / BLEU')
#         ax.bar([p + 0.35 for p in x], recall_like, width=0.35, label='Recall / ROUGE-L')

#         ax.set_xticks([p + 0.17 for p in x])
#         ax.set_xticklabels(categories, rotation=20, ha='right', fontsize=8)  # tilt & shrink labels
#         ax.set_ylabel("Score", fontsize=9)
#         ax.set_title("Evaluation Metrics Comparison", fontsize=10)
#         ax.set_ylim(0, 1)
#         ax.legend(fontsize=8)
#         plt.tight_layout(pad=1.0)  # adjusts spacing automatically

#         st.pyplot(fig, use_container_width=False)

