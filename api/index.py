from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import requests
import os
from pathlib import Path

app = FastAPI(title="Text Summarizer App", description="Text Summarization using T5", version="1.0")

# Root directory for index.html template
BASE_DIR = Path(__file__).parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR))

# Hugging Face Inference API Endpoint (Free Serverless)
# Defaults to t5-small or custom fine-tuned model path
HF_MODEL_ID = os.environ.get("HF_MODEL_ID", "t5-small")
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"

class DialogueInput(BaseModel):
    dialogue: str

def query_hf_api(text: str) -> str:
    headers = {}
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    
    payload = {
        "inputs": f"summarize: {text}",
        "parameters": {
            "max_length": 150,
            "min_length": 15
        }
    }
    
    urls = [
        f"https://router.huggingface.co/hf-inference/models/{HF_MODEL_ID}",
        f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"
    ]
    
    for url in urls:
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=15)
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0 and "summary_text" in result[0]:
                    return result[0]["summary_text"]
                elif isinstance(result, dict) and "summary_text" in result:
                    return result["summary_text"]
        except Exception:
            continue

    return "Model is initializing, please try again in a few seconds."

@app.post("/summarize/")
def summarize(dialogue_input: DialogueInput):
    summary = query_hf_api(dialogue_input.dialogue)
    return {"summary": summary}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")
