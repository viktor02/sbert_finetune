import contextlib
import re
import statistics
from typing import List, Optional

import requests
import torch
import torch.nn.functional as F
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# --- Configuration ---
MODEL_PATH = "../models/final_ai_detector"
BASE_MODEL_NAME = "ai-forever/sbert_large_mt_nlu_ru"
WINDOW_SIZE = 3
STRIDE = 1
MIN_WORDS = 10

# Global storage for model and tokenizer
ml_models = {}


# --- Data Models ---
class AnalyzeRequest(BaseModel):
    url: Optional[str] = Field(None, description="URL статьи для проверки")
    text: Optional[str] = Field(
        None, description="Сырой текст для проверки (если нет URL)"
    )


class ChunkScore(BaseModel):
    text: str
    score: float


class AnalyzeResponse(BaseModel):
    verdict: str
    reason: str
    avg_ai_score: float
    max_ai_score: float
    median_ai_score: float
    total_chunks: int
    suspicious_chunks_count: int
    top_suspicious_chunks: List[ChunkScore]


# --- Helper Functions ---
def clean_html(html_content):
    if not html_content:
        return ""
    soup = BeautifulSoup(html_content, "html.parser")
    for element in soup(
        ["script", "style", "meta", "noscript", "iframe", "svg", "path"]
    ):
        element.decompose()
    for element in soup(["pre", "code"]):
        element.decompose()
    text = soup.get_text(separator=" ", strip=True)
    return " ".join(text.split())


def split_into_sentences(text):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text_sliding_window(text):
    sentences = split_into_sentences(text)
    chunks = []
    if len(sentences) < WINDOW_SIZE:
        full_text = " ".join(sentences)
        if len(full_text.split()) >= MIN_WORDS:
            return [full_text]
        return []
    for i in range(0, len(sentences) - WINDOW_SIZE + 1, STRIDE):
        group = sentences[i : i + WINDOW_SIZE]
        chunk = " ".join(group)
        if len(chunk.split()) >= MIN_WORDS:
            chunks.append(chunk)
    return chunks


def get_html_from_url(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error fetching URL: {str(e)}")


# --- Lifespan (Startup/Shutdown) ---
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Load Model
    print("Loading model and tokenizer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        except:
            print(f"Warning: Local tokenizer not found, downloading {BASE_MODEL_NAME}")
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(device)
        model.eval()

        ml_models["model"] = model
        ml_models["tokenizer"] = tokenizer
        ml_models["device"] = device
        print(f"Model loaded on {device}")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load model. {e}")
        raise e

    yield

    # Clean up resources
    ml_models.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(title="Habr AI Detector API", lifespan=lifespan)

# --- Endpoints ---


@app.post("/predict", response_model=AnalyzeResponse)
def predict(request: AnalyzeRequest):
    # 1. Get Content
    if request.url:
        raw_html = get_html_from_url(request.url)
        cleaned_text = clean_html(raw_html)
    elif request.text:
        cleaned_text = request.text
    else:
        raise HTTPException(status_code=400, detail="Provide either 'url' or 'text'")

    # 2. Chunking
    chunks = chunk_text_sliding_window(cleaned_text)
    if not chunks:
        raise HTTPException(
            status_code=422, detail="Text is too short or empty after cleaning"
        )

    # 3. Inference
    model = ml_models["model"]
    tokenizer = ml_models["tokenizer"]
    device = ml_models["device"]

    ai_probs = []
    batch_size = 16  # Slightly larger batch size for API

    with torch.no_grad():
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]
            inputs = tokenizer(
                batch_chunks,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)

            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            # Class 1 is AI
            ai_probs.extend(probs[:, 1].cpu().numpy().tolist())

    # 4. Statistics & Logic
    avg_ai_prob = statistics.mean(ai_probs)
    max_ai_prob = max(ai_probs)
    median_ai_prob = statistics.median(ai_probs)
    suspicious_chunks = sum(1 for p in ai_probs if p > 0.5)

    # Prepare top chunks
    scored_chunks = list(zip(chunks, ai_probs))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    top_chunks = [ChunkScore(text=c, score=s) for c, s in scored_chunks[:3]]

    # 5. Verdict Logic
    if avg_ai_prob > 0.5:
        verdict = "AI-GENERATED"
        reason = "High average AI probability across the text."
    elif suspicious_chunks / len(chunks) > 0.25:
        verdict = "LIKELY AI (MIXED)"
        reason = f"Significant portion ({suspicious_chunks}/{len(chunks)}) of chunks look like AI."
    elif avg_ai_prob > 0.25 and max_ai_prob > 0.8:
        verdict = "SUSPICIOUS"
        reason = "Elevated average score with high-confidence AI spikes."
    else:
        verdict = "HUMAN-WRITTEN"
        reason = "Consistent low AI probability."

    return AnalyzeResponse(
        verdict=verdict,
        reason=reason,
        avg_ai_score=avg_ai_prob,
        max_ai_score=max_ai_prob,
        median_ai_score=median_ai_prob,
        total_chunks=len(chunks),
        suspicious_chunks_count=suspicious_chunks,
        top_suspicious_chunks=top_chunks,
    )


@app.get("/health")
def health_check():
    return {"status": "ok", "device": str(ml_models.get("device", "unknown"))}
