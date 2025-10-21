# Task C

from src.retrieval import retrieve
import google.generativeai as genai
import os
from dotenv import load_dotenv
from src.summarizer import summarizer_func
load_dotenv()

api_key = os.getenv('gemini_api')
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.5-flash')

def generator_using_retrieval(query, top_k):
    retrieved_content = retrieve(query, top_k)
    combined_content = " ".join([doc.get("content", "") for doc in retrieved_content])

    summarized_content = summarizer_func(combined_content)
    prompt = f"""You are an assistant. Use the following retrieve passages to answer.

    Context:
    {summarized_content}

    Question:
    {query}

    Answer:"""

    response = model.generate_content(prompt)
    return response.text
    # return summarized_content

