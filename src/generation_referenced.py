import google.generativeai as genai
import os
from dotenv import load_dotenv
import json
from pathlib import Path
import re
from src.summarizer import summarizer_func
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

load_dotenv()

DATA_FILE = Path(os.getenv('corpus_file'))
OUTPUT_FILE = Path("./outputs/generation_reference.jsonl")


with open(os.getenv('datafile'), "r", encoding='utf-8') as f:
    data = f.read()

# Configure Gemini API
genai.configure(api_key=os.getenv('gemini_api'))
model = genai.GenerativeModel(model_name="gemini-2.5-flash")

def generator_using_reference(question, context):
    prompt = f"""You are an assistant. Use the following reference passages to answer.

Context:
{context}

Question:
{question}

Answer:"""
    response = model.generate_content(prompt)
    return response.text


def reference_based_generation(query):

    summary = summarizer_func(data)
    generated_answer = generator_using_reference(query, summary)
    # print(generated_answer)
    # print(summary)
    return generated_answer

# reference_based_generation(query="Which California state parks allow dogs on the beach?")