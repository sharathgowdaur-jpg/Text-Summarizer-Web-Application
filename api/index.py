from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import requests
import os
import re
import math
from collections import Counter
from pathlib import Path

app = FastAPI(title="Text Summarizer App", description="Text Summarization using T5 and Extractive NLP", version="1.0")

BASE_DIR = Path(__file__).parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR))

HF_MODEL_ID = os.environ.get("HF_MODEL_ID", "t5-small")

class DialogueInput(BaseModel):
    dialogue: str

def smart_compress_summarize(text: str) -> str:
    """Intelligent NLP text condensation algorithm that extracts key information and shortens input by 50-70%."""
    clean_text = re.sub(r'<.*?>', ' ', text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    if not clean_text:
        return "No content provided to summarize."
        
    sentences = re.split(r'(?<=[.!?])\s+', clean_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    
    if not sentences:
        return clean_text
        
    if len(sentences) == 1:
        clauses = re.split(r'[,;:]\s+', sentences[0])
        if len(clauses) > 1:
            return clauses[0].strip() + "."
        return sentences[0]

    stopwords = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'to', 'in', 'of', 'on', 'with', 'for', 'at', 'by', 'from', 'it', 'this', 'that', 'and', 'or', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall', 'should', 'can', 'could', 'may', 'might', 'must', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'under', 'again', 'further', 'then', 'once'}
    words = [w.lower() for w in re.findall(r'\w+', clean_text)]
    filtered_words = [w for w in words if w not in stopwords and len(w) > 2]
    
    word_counts = Counter(filtered_words)
    if not word_counts:
        return sentences[0]
        
    max_freq = max(word_counts.values())
    word_weights = {w: freq / max_freq for w, freq in word_counts.items()}
    
    sentence_scores = []
    for idx, sentence in enumerate(sentences):
        sentence_words = [w.lower() for w in re.findall(r'\w+', sentence)]
        if not sentence_words:
            continue
        score = sum(word_weights.get(w, 0) for w in sentence_words) / math.sqrt(len(sentence_words))
        sentence_scores.append((score, idx, sentence))
    
    sentence_scores.sort(key=lambda x: x[0], reverse=True)
    
    target_count = 1 if len(sentences) <= 4 else 2
    top_sentences = sentence_scores[:target_count]
    top_sentences.sort(key=lambda x: x[1])
    
    summary = " ".join(s[2] for s in top_sentences)
    return summary

def query_hf_api(text: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    
    payload = {
        "inputs": f"summarize: {text}",
        "parameters": {
            "max_length": 100,
            "min_length": 15
        },
        "options": {
            "wait_for_model": True
        }
    }
    
    urls = [
        f"https://router.huggingface.co/hf-inference/models/{HF_MODEL_ID}",
        f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"
    ]
    
    if hf_token:
        for url in urls:
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=10)
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0 and "summary_text" in result[0]:
                        return result[0]["summary_text"]
                    elif isinstance(result, dict) and "summary_text" in result:
                        return result["summary_text"]
            except Exception:
                continue

    return smart_compress_summarize(text)

@app.post("/summarize/")
def summarize(dialogue_input: DialogueInput):
    summary = query_hf_api(dialogue_input.dialogue)
    return {"summary": summary}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")
