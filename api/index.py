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

# Root directory for index.html template
BASE_DIR = Path(__file__).parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR))

HF_MODEL_ID = os.environ.get("HF_MODEL_ID", "t5-small")

class DialogueInput(BaseModel):
    dialogue: str

def extractive_summarize(text: str, max_sentences: int = 3) -> str:
    """Fallback high-accuracy sentence ranking NLP summarizer."""
    clean_text = re.sub(r'<.*?>', ' ', text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    sentences = re.split(r'(?<=[.!?])\s+', clean_text)
    if len(sentences) <= max_sentences:
        return clean_text
    
    stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'to', 'in', 'of', 'on', 'with', 'for', 'at', 'by', 'from', 'it', 'this', 'that', 'you', 'i', 'we', 'they', 'he', 'she', 'my', 'your', 'our', 'in'}
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
        score = sum(word_weights.get(w, 0) for w in sentence_words)
        if len(sentence_words) > 0:
            score = score / math.sqrt(len(sentence_words))
        sentence_scores.append((score, idx, sentence))
    
    sentence_scores.sort(key=lambda x: x[0], reverse=True)
    top_sentences = sentence_scores[:max_sentences]
    top_sentences.sort(key=lambda x: x[1])
    
    return ' '.join(s[2] for s in top_sentences)

def query_hf_api(text: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    
    payload = {
        "inputs": f"summarize: {text}",
        "parameters": {
            "max_length": 150,
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
    
    for url in urls:
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=12)
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0 and "summary_text" in result[0]:
                    return result[0]["summary_text"]
                elif isinstance(result, dict) and "summary_text" in result:
                    return result["summary_text"]
        except Exception:
            continue

    # Fallback to instant Extractive Summarizer if Hugging Face token is absent or rate-limited
    return extractive_summarize(text)

@app.post("/summarize/")
def summarize(dialogue_input: DialogueInput):
    summary = query_hf_api(dialogue_input.dialogue)
    return {"summary": summary}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")
