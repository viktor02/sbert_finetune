import asyncio
import contextlib
import logging
import re
import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx
import torch
import torch.nn.functional as F
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# --- 1. Configuration ---
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
    batch_size: int = 16  # Layout on GPU
    max_queue_size: int = 100  # Backpressure limit
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

# --- 3. Data Models ---


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
    queue_wait_time_ms: Optional[float] = None


@dataclass
class QueueItem:
    request_id: str
    chunks: List[str]
    future: asyncio.Future
    arrival_time: float


# --- 4. Global Resources (State Management) ---
class GlobalResources:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.http_client: Optional[httpx.AsyncClient] = None
        self.inference_queue: Optional[asyncio.Queue] = None
        self.worker_task: Optional[asyncio.Task] = None


resources = GlobalResources()

# --- 5. Helper Functions ---


def clean_html_sync(html_content: str) -> str:
    """CPU-bound HTML cleaning. Removes tags and scripts."""
    if not html_content:
        return ""
    soup = BeautifulSoup(html_content, "html.parser")

    for element in soup(
        ["script", "style", "meta", "noscript", "iframe", "svg", "path"]
    ):
        element.decompose()

    # Also remove code blocks to avoid false positives on code syntax
    for element in soup(["pre", "code"]):
        element.decompose()

    text = soup.get_text(separator=" ", strip=True)
    return " ".join(text.split())


def chunk_text_sliding_window(text: str) -> List[str]:
    """Splits text into sliding window chunks."""
    # Simple regex for sentence splitting
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

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


def _run_torch_inference(chunks: List[str]) -> List[float]:
    """
    Actual GPU/CPU inference function.
    This MUST be run in a threadpool or strictly serialized to ensure
    no conflicts with the async event loop.
    """
    if not chunks:
        return []

    ai_probs = []
    model = resources.model
    tokenizer = resources.tokenizer
    device = resources.device

    # Mathematical: $P(AI) = Softmax(Logits)_{class=1}$
    # We use torch.no_grad() to strictly avoid building a computation graph (Memory Leak prevention)
    with torch.no_grad():
        for i in range(0, len(chunks), settings.batch_size):
            batch_chunks = chunks[i : i + settings.batch_size]

            # Move inputs to device
            inputs = tokenizer(
                batch_chunks,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)

            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)

            # Detach immediately and ensure moving to CPU to free VRAM
            ai_probs.extend(probs[:, 1].cpu().numpy().tolist())

            # Explicitly delete tensors to aid GC inside the loop if batch is huge
            del inputs, outputs, probs

    return ai_probs


# --- 6. Queue Worker ---


async def inference_worker():
    """
    Consumer: continuously pulls requests from the queue.
    Processes requests serially to prevent GPU OOM.
    """
    logger.info("Inference worker started.")
    while True:
        item: QueueItem = await resources.inference_queue.get()

        try:
            # Check formatting or empty chunks
            if not item.chunks:
                item.future.set_result([])
                continue

            # Run inference in a separate thread to prevent blocking the asyncio loop
            # Even though we are in a worker, the inference is CPU blocking until GPU dispatch
            logger.debug(
                f"Processing request {item.request_id} with {len(item.chunks)} chunks"
            )

            probs = await run_in_threadpool(_run_torch_inference, item.chunks)

            if not item.future.done():
                item.future.set_result(probs)

        except Exception as e:
            logger.error(f"Error processing request {item.request_id}: {e}")
            if not item.future.done():
                item.future.set_exception(e)
        finally:
            resources.inference_queue.task_done()


# --- 7. Lifespan (Startup/Shutdown) ---


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    logger.info("Startup: Initializing resources...")

    # 1. Load Model
    resources.device = torch.device(settings.device)
    try:
        resources.tokenizer = AutoTokenizer.from_pretrained(settings.model_path)
        resources.model = AutoModelForSequenceClassification.from_pretrained(
            settings.model_path
        )
        resources.model.to(resources.device)
        resources.model.eval()
        logger.info(f"Model loaded on {settings.device}")
    except Exception as e:
        logger.critical(f"Failed to load model: {e}")
        raise e

    resources.http_client = httpx.AsyncClient(
        follow_redirects=True, timeout=10.0, headers={"User-Agent": "HabrFilterBot/1.0"}
    )

    # 3. Initialize & Start Queue
    resources.inference_queue = asyncio.Queue(maxsize=settings.max_queue_size)
    resources.worker_task = asyncio.create_task(inference_worker())

    yield

    # --- Shutdown ---
    logger.info("Shutdown: Cleaning up resources...")

    # 1. Cancel Worker
    if resources.worker_task:
        resources.worker_task.cancel()
        try:
            await resources.worker_task
        except asyncio.CancelledError:
            pass

    # 2. Close HTTP Client
    await resources.http_client.aclose()

    # 3. Clear GPU Memory
    resources.model = None
    resources.tokenizer = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --- 8. App Definition ---
app = FastAPI(title="Habr AI Detector API", lifespan=lifespan)

# --- 9. Endpoints ---


@app.post("/predict", response_model=AnalyzeResponse)
async def predict(request: AnalyzeRequest):
    start_time = time.time()

    # 1. Fetch / Clean Data
    if request.url:
        # Use the shared client
        try:
            resp = await resources.http_client.get(request.url)
            resp.raise_for_status()
            raw_text = resp.text
            cleaned_text = await run_in_threadpool(clean_html_sync, raw_text)
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code, detail="Upstream Fetch Error"
            )
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Network error: {e}")
    elif request.text:
        cleaned_text = request.text
    else:
        raise HTTPException(status_code=400, detail="Provide 'url' or 'text'")

    # 2. Chunking
    chunks = chunk_text_sliding_window(cleaned_text)
    if not chunks:
        raise HTTPException(status_code=422, detail="Text too short to analyze")

    # 3. Enqueue for Inference
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    item = QueueItem(
        request_id=str(time.time()),
        chunks=chunks,
        future=future,
        arrival_time=time.time(),
    )

    try:
        # Push to queue (with timeout for backpressure)
        await asyncio.wait_for(resources.inference_queue.put(item), timeout=2.0)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="Server busy (Queue full)")

    # 4. Await Result
    queue_ep_time = time.time()
    try:
        ai_probs = await future
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    # 5. Post-Processing Stats
    if not ai_probs:
        raise HTTPException(status_code=422, detail="Model returned no predictions")

    avg_ai_prob = statistics.mean(ai_probs)
    max_ai_prob = max(ai_probs)
    median_ai_prob = statistics.median(ai_probs)
    suspicious_chunks = sum(1 for p in ai_probs if p > 0.5)

    scored_chunks = sorted(zip(chunks, ai_probs), key=lambda x: x[1], reverse=True)
    top_chunks = [ChunkScore(text=c, score=s) for c, s in scored_chunks[:3]]

    # Verdict Logic
    # If $Avg(P) > 0.5$, we flag as AI
    if avg_ai_prob > 0.5:
        verdict = "AI-GENERATED"
        reason = "High average AI probability."
    elif suspicious_chunks / len(chunks) > 0.25:
        verdict = "LIKELY AI (MIXED)"
        reason = (
            f"Significant mixed content ({suspicious_chunks}/{len(chunks)} chunks)."
        )
    elif avg_ai_prob > 0.25 and max_ai_prob > 0.8:
        verdict = "SUSPICIOUS"
        reason = "Generally human, but contains high-confidence AI spikes."
    else:
        verdict = "HUMAN-WRITTEN"
        reason = "Consistent human probability patterns."

    total_time = (time.time() - start_time) * 1000
    queue_wait = (queue_ep_time - start_time) * 1000

    return AnalyzeResponse(
        verdict=verdict,
        reason=reason,
        avg_ai_score=avg_ai_prob,
        max_ai_score=max_ai_prob,
        median_ai_score=median_ai_prob,
        total_chunks=len(chunks),
        suspicious_chunks_count=suspicious_chunks,
        top_suspicious_chunks=top_chunks,
        processing_time_ms=round(total_time, 2),
        queue_wait_time_ms=round(queue_wait, 2),
    )


@app.get("/health")
async def health_check():
    q_size = resources.inference_queue.qsize() if resources.inference_queue else 0
    return {"status": "ok", "device": str(resources.device), "queue_depth": q_size}
