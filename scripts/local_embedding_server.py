"""Local embedding server — OpenAI-compatible API for askme memory.

Usage::

    pip install "sentence-transformers>=2.7.0" fastapi uvicorn
    python scripts/local_embedding_server.py

The server exposes ``POST /v1/embeddings`` and ``GET /v1/models``.
"""

import argparse
import logging
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")

app = FastAPI(title="Askme Embedding Server")

model: SentenceTransformer | None = None
model_name: str = DEFAULT_MODEL


def load_model(name: str) -> SentenceTransformer:
    import torch

    logger.info("Loading embedding model: %s (this may take a moment)...", name)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    m = SentenceTransformer(name, trust_remote_code=True, model_kwargs={"torch_dtype": dtype})
    device = next(m.parameters()).device
    logger.info(
        "Model loaded. Dimension=%d, max_seq_length=%s, dtype=%s, device=%s",
        m.get_sentence_embedding_dimension(), m.max_seq_length, dtype, device,
    )
    return m


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

class EmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str = ""


@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        inputs = request.input
        if isinstance(inputs, str):
            inputs = [inputs]

        embeddings = model.encode(inputs, normalize_embeddings=True)

        data = []
        for i, emb in enumerate(embeddings):
            data.append({
                "object": "embedding",
                "index": i,
                "embedding": emb.tolist(),
            })

        return {
            "object": "list",
            "data": data,
            "model": model_name,
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }
    except Exception as e:
        logger.error("Embedding error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def list_models():
    import torch

    dim = model.get_sentence_embedding_dimension() if model else 0
    dtype = str(next(model.parameters()).dtype) if model else "N/A"
    device = str(next(model.parameters()).device) if model else "N/A"
    return {
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "owned_by": "local",
                "embedding_dimension": dim,
                "dtype": dtype,
                "device": device,
            }
        ],
    }


# ------------------------------------------------------------------
# Startup
# ------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Local embedding server")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace model name")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    model_name = args.model
    model = load_model(model_name)
    uvicorn.run(app, host=args.host, port=args.port)
