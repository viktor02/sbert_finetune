import contextlib
import logging
import re
import statistics
from typing import Any, Dict, List, Optional

import httpx
import torch
import torch.nn.functional as F
from bs4 import BeautifulSoup
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# --- 1. Configuration (Cloud Ready) ---
class Settings(BaseSettings):
    # Model Config
    hf_model_id: str = "viktor02/sbert_classification_ru_ai_texts"
    model_path: str = Field(
        default="viktor02/sbert_classification_ru_ai_texts",
        validation_alias="MODEL_PATH",
    )

    # Inference Config
    window_size: int = 3
    stride: int = 1
    min_words: int = 10
    batch_size: int = 16
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # App Config
    log_level: str = "INFO"


settings = Settings()

# --- 2. Logging Setup ---
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("api")


# --- 3. Global State ---
class MLResources:
    model = None
    tokenizer = None
    device = None


ml_resources = MLResources()


# --- 4. Data Models ---
class AnalyzeRequest(BaseModel):
    url: Optional[str] = Field(None, description="URL of the article to check")
    text: Optional[str] = Field(None, description="Raw text to check (if no URL)")


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
    processing_time_ms: Optional[float] = None


# --- 5. Helper Functions ---


def clean_html_sync(html_content: str) -> str:
    """CPU-bound HTML cleaning."""
    if not html_content:
        return ""
    soup = BeautifulSoup(html_content, "html.parser")

    # Remove unwanted tags
    for element in soup(
        ["script", "style", "meta", "noscript", "iframe", "svg", "path"]
    ):
        element.decompose()

    # Remove code blocks (optional, based on your logic)
    for element in soup(["pre", "code"]):
        element.decompose()

    text = soup.get_text(separator=" ", strip=True)
    return " ".join(text.split())


def split_into_sentences(text: str) -> List[str]:
    # Improved regex to handle common abbreviations could go here,
    # but keeping it simple for performance.
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text_sliding_window(text: str) -> List[str]:
    sentences = split_into_sentences(text)
    chunks = []

    if len(sentences) < settings.window_size:
        full_text = " ".join(sentences)
        if len(full_text.split()) >= settings.min_words:
            return [full_text]
        return []

    for i in range(0, len(sentences) - settings.window_size + 1, settings.stride):
        group = sentences[i : i + settings.window_size]
        chunk = " ".join(group)
        if len(chunk.split()) >= settings.min_words:
            chunks.append(chunk)
    return chunks


async def fetch_url_async(url: str) -> str:
    """Async HTTP fetcher using httpx."""
    headers = {"User-Agent": "HabrFilterBot/1.0"}
    async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
        try:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            return response.text
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching {url}: {e}")
            raise HTTPException(
                status_code=e.response.status_code, detail="Could not fetch URL"
            )
        except httpx.RequestError as e:
            logger.error(f"Connection error fetching {url}: {e}")
            raise HTTPException(status_code=400, detail=f"Network error: {str(e)}")


def run_inference_sync(chunks: List[str]) -> List[float]:
    """
    Runs the actual PyTorch inference.
    This is blocking, so it will be run in a threadpool.
    """
    if not chunks:
        return []

    model = ml_resources.model
    tokenizer = ml_resources.tokenizer
    device = ml_resources.device

    ai_probs = []

    # Batch processing
    with torch.no_grad():
        for i in range(0, len(chunks), settings.batch_size):
            batch_chunks = chunks[i : i + settings.batch_size]
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

    return ai_probs


# --- 6. Lifespan ---
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Startup: Loading model and tokenizer...")
    try:
        # We load these in the main thread during startup (blocking is fine here)
        ml_resources.device = torch.device(settings.device)
        ml_resources.tokenizer = AutoTokenizer.from_pretrained(settings.model_path)
        ml_resources.model = AutoModelForSequenceClassification.from_pretrained(
            settings.model_path
        )
        ml_resources.model.to(ml_resources.device)
        ml_resources.model.eval()

        logger.info(f"Model loaded successfully on {settings.device}")
    except Exception as e:
        logger.critical(f"Failed to load model: {e}")
        raise e

    yield

    logger.info("Shutdown: Clearing resources...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    ml_resources.model = None
    ml_resources.tokenizer = None


# --- 7. App Definition ---
app = FastAPI(title="Habr AI Detector API", lifespan=lifespan)

# --- 8. Endpoints ---


@app.post("/predict", response_model=AnalyzeResponse)
async def predict(request: AnalyzeRequest):
    """
    Analyzes text or URL for AI-generated content.
    Uses async/await to handle network I/O and threadpools for CPU tasks.
    """

    # 1. Get Content (Async I/O)
    if request.url:
        logger.info(f"Processing URL: {request.url}")
        raw_html = await fetch_url_async(request.url)
        # Offload HTML cleaning to threadpool (CPU bound)
        cleaned_text = await run_in_threadpool(clean_html_sync, raw_html)
    elif request.text:
        logger.info("Processing raw text input")
        cleaned_text = request.text
    else:
        raise HTTPException(status_code=400, detail="Provide either 'url' or 'text'")

    # 2. Chunking (CPU bound - fast enough to run here, or offload if huge)
    chunks = chunk_text_sliding_window(cleaned_text)

    if not chunks:
        logger.warning("Text too short after cleaning")
        raise HTTPException(
            status_code=422, detail="Text is too short or empty after cleaning"
        )

    # 3. Inference (Heavy CPU/GPU bound - MUST offload)
    try:
        ai_probs = await run_in_threadpool(run_inference_sync, chunks)
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail="Internal model error")

    # 4. Statistics
    if not ai_probs:
        raise HTTPException(status_code=422, detail="Could not generate probabilities")

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

    logger.info(f"Verdict: {verdict} (Avg: {avg_ai_prob:.2f})")

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
async def health_check():
    """
    K8s/Docker health check.
    Verifies model is actually loaded in memory.
    """
    if ml_resources.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "status": "ok",
        "device": str(ml_resources.device),
        "model": settings.model_path,
    }
