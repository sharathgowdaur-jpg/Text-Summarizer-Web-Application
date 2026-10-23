from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import re
import math
import io
from collections import Counter
from pathlib import Path
from contextlib import asynccontextmanager

model = None
tokenizer = None
device = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, device
    model_dir = Path(__file__).parent / "saved_summary_model"
    if model_dir.exists() and any(model_dir.iterdir()):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
                
            model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir), local_files_only=True).to(device)
            tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
            print("Loaded local fine-tuned T5 model successfully.")
        except Exception as e:
            print(f"Local saved_summary_model failed to load ({e}). Using Smart Compress NLP engine.")
            model = None
            tokenizer = None
    else:
        print("Local saved_summary_model directory not found. Using Smart Compress NLP engine.")
        model = None
        tokenizer = None
    yield

app = FastAPI(
    title="Briefly AI - Text Summarizer",
    description="Text Summarization using T5 and Extractive NLP",
    version="2.0",
    lifespan=lifespan
)

# Enable CORS for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR))

class DialogueInput(BaseModel):
    dialogue: str
    style: str = "quick"
    length: str = "medium"

def clean_data(text):
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = text.strip().lower()
    return text

def smart_compress_summarize(text: str, style: str = "quick", length: str = "medium") -> str:
    """Intelligent NLP condensation fallback when local deep learning weights are absent."""
    clean_text = re.sub(r'<.*?>', ' ', text)
    clean_text = re.sub(r'\r\n', ' ', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    if not clean_text:
        return "No content provided to summarize."
        
    sentences = re.split(r'(?<=[.!?])\s+', clean_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    
    if not sentences:
        return clean_text
        
    if len(sentences) == 1:
        if style == "bullet":
            return f"• {sentences[0]}"
        elif style == "takeaways":
            return f"• Core Takeaway: {sentences[0]}"
        return sentences[0]

    stopwords = {
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'to', 'in', 'of', 'on', 'with', 'for', 'at',
        'by', 'from', 'it', 'this', 'that', 'and', 'or', 'be', 'been', 'being', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'shall', 'should', 'can', 'could', 'may', 'might',
        'must', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'under',
        'again', 'further', 'then', 'once'
    }
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
        pos_multiplier = 1.25 if idx == 0 else (1.15 if idx == len(sentences) - 1 else 1.0)
        base_score = sum(word_weights.get(w, 0) for w in sentence_words) / math.sqrt(len(sentence_words))
        score = base_score * pos_multiplier
        sentence_scores.append((score, idx, sentence))
    
    sentence_scores.sort(key=lambda x: x[0], reverse=True)
    
    n_sents = len(sentences)
    if length == "short":
        target_count = max(1, min(2, math.ceil(n_sents * 0.25)))
    elif length == "long":
        target_count = max(3, min(n_sents, math.ceil(n_sents * 0.6)))
    else:
        target_count = max(2, min(n_sents, math.ceil(n_sents * 0.4)))
        
    target_count = min(target_count, n_sents)
    top_sentences = sentence_scores[:target_count]
    top_sentences.sort(key=lambda x: x[1])
    
    if style == "bullet":
        return "\n".join(f"• {s[2]}" for s in top_sentences)
    elif style == "takeaways":
        prefixes = ["Core Finding", "Critical Takeaway", "Key Insight", "Strategic Point", "Main Conclusion"]
        takeaway_items = []
        for i, s in enumerate(top_sentences):
            prefix = prefixes[i] if i < len(prefixes) else f"Key Point {i+1}"
            takeaway_items.append(f"• {prefix}: {s[2]}")
        return "\n".join(takeaway_items)
    elif style == "detailed":
        return " ".join(s[2] for s in top_sentences)
    else:
        return " ".join(s[2] for s in top_sentences)

def summarize_dialogue(dialogue: str, style: str = "quick", length: str = "medium") -> str:
    if model is None or tokenizer is None:
        return smart_compress_summarize(dialogue, style, length)

    cleaned = clean_data(dialogue)

    inputs = tokenizer(
        cleaned,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    max_len = 80 if length == "short" else (240 if length == "long" else 150)

    targets = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_len,
        num_beams=4,
        early_stopping=True
    )
    
    summary = tokenizer.decode(targets[0], skip_special_tokens=True)
    
    if style == "bullet":
        sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', summary) if s.strip()]
        return "\n".join(f"• {s}" for s in sents) if sents else f"• {summary}"
    elif style == "takeaways":
        sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', summary) if s.strip()]
        prefixes = ["Core Finding", "Critical Takeaway", "Key Insight", "Strategic Point", "Main Conclusion"]
        if sents:
            return "\n".join(f"• {prefixes[i] if i < len(prefixes) else f'Key Point {i+1}'}: {s}" for i, s in enumerate(sents))
        return f"• Core Takeaway: {summary}"

    return summary

def extract_text_from_bytes(filename: str, content: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".txt":
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            return content.decode("latin-1", errors="ignore")
    elif ext == ".pdf":
        text_parts = []
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(content))
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text_parts.append(t)
        except Exception:
            try:
                import fitz
                doc = fitz.open(stream=content, filetype="pdf")
                for page in doc:
                    text_parts.append(page.get_text())
            except Exception as e:
                raise ValueError(f"Failed to read PDF document: {str(e)}")
                
        extracted = "\n\n".join(text_parts).strip()
        if not extracted:
            raise ValueError("Unable to extract text from this PDF. It may be a scanned document without an embedded text layer.")
        return extracted
    else:
        raise ValueError("Unsupported file type. Please upload a .txt or .pdf document.")

# API endpoints
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Briefly AI - Text Summarizer",
        "version": "2.0",
        "mode": "local",
        "model_loaded": model is not None,
        "supported_styles": ["quick", "detailed", "bullet", "takeaways"],
        "supported_lengths": ["short", "medium", "long"]
    }

@app.post("/summarize/")
def summarize(dialogue_input: DialogueInput):
    text = dialogue_input.dialogue.strip() if dialogue_input.dialogue else ""
    if not text:
        raise HTTPException(status_code=400, detail="Source text cannot be empty.")
    
    summary = summarize_dialogue(text, dialogue_input.style, dialogue_input.length)
    return {
        "summary": summary,
        "style": dialogue_input.style,
        "length": dialogue_input.length
    }

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file was uploaded.")
    
    ext = Path(file.filename).suffix.lower()
    if ext not in [".txt", ".pdf"]:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a .txt or .pdf file.")

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size exceeds the 10MB limit.")

    try:
        extracted = extract_text_from_bytes(file.filename, content)
        words = len(extracted.split())
        return {
            "filename": file.filename,
            "text": extracted,
            "words": words,
            "chars": len(extracted)
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing error: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting Briefly AI server on http://0.0.0.0:{port}...")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
