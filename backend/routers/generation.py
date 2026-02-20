"""
Generation API router.

Endpoints:
    POST /api/generate     — generate text token-by-token (SSE stream)
    POST /api/generate-sync — generate text synchronously (for simpler clients)
"""

import json
import asyncio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from typing import Optional

import torch

router = APIRouter(prefix="/api", tags=["generation"])

_model = None
_tokenizer = None


def set_deps(model, tokenizer):
    global _model, _tokenizer
    _model = model
    _tokenizer = tokenizer


class GenerateRequest(BaseModel):
    prompt: str = "ROMEO:"
    max_tokens: int = 200
    temperature: float = 0.8
    top_k: Optional[int] = 40
    top_p: Optional[float] = None
    strategy: str = "temperature"  # greedy | temperature | top_k | nucleus


@router.post("/generate")
async def generate_stream(req: GenerateRequest):
    """Generate text token-by-token, streaming each token with its probability distribution."""
    if _model is None or _tokenizer is None:
        raise HTTPException(503, "Model not initialized")

    async def event_generator():
        tok = _tokenizer
        model = _model
        model.eval()

        ids = tok.encode(req.prompt)
        input_ids = torch.tensor([ids])

        # Determine generation params from strategy
        temperature = 0.0 if req.strategy == "greedy" else req.temperature
        top_k = req.top_k if req.strategy in ("top_k", "temperature") else None
        top_p = req.top_p if req.strategy == "nucleus" else None

        # Generate tokens — run in thread pool to avoid blocking the event loop
        def _run_generate():
            with torch.no_grad():
                return model.generate(
                    input_ids,
                    max_new_tokens=req.max_tokens,
                    temperature=max(temperature, 1e-8),
                    top_k=top_k,
                    top_p=top_p,
                )

        steps = await asyncio.to_thread(_run_generate)

        # Stream each token
        generated_ids = list(ids)
        for i, step in enumerate(steps):
            token_id = step["token_id"]
            generated_ids.append(token_id)
            token_text = tok.vocab[token_id].decode("utf-8", errors="replace") if token_id < tok.vocab_size else f"[{token_id}]"

            top_tokens = []
            for tid, prob in zip(step["top_ids"], step["top_probs"]):
                t = tok.vocab[tid].decode("utf-8", errors="replace") if tid < tok.vocab_size else f"[{tid}]"
                top_tokens.append({"id": tid, "text": t, "probability": round(prob, 4)})

            event = {
                "type": "token",
                "step": i,
                "token_id": token_id,
                "token_text": token_text,
                "cumulative_text": tok.decode(generated_ids),
                "top_predictions": top_tokens,
            }

            yield {"event": "token", "data": json.dumps(event)}
            await asyncio.sleep(0.05)  # Small delay for visual effect

        # Final event
        yield {
            "event": "done",
            "data": json.dumps({
                "type": "done",
                "full_text": tok.decode(generated_ids),
                "total_tokens": len(steps),
            }),
        }

    return EventSourceResponse(event_generator())


@router.post("/generate-sync")
async def generate_sync(req: GenerateRequest):
    """Generate text synchronously — returns full result at once."""
    if _model is None or _tokenizer is None:
        raise HTTPException(503, "Model not initialized")

    tok = _tokenizer
    model = _model
    model.eval()

    ids = tok.encode(req.prompt)
    input_ids = torch.tensor([ids])

    temperature = 0.0 if req.strategy == "greedy" else req.temperature
    top_k = req.top_k if req.strategy in ("top_k", "temperature") else None
    top_p = req.top_p if req.strategy == "nucleus" else None

    # Run in thread pool to avoid blocking the event loop
    def _run_generate():
        with torch.no_grad():
            return model.generate(
                input_ids,
                max_new_tokens=req.max_tokens,
                temperature=max(temperature, 1e-8),
                top_k=top_k,
                top_p=top_p,
            )

    steps = await asyncio.to_thread(_run_generate)

    generated_ids = list(ids)
    token_details = []
    for step in steps:
        token_id = step["token_id"]
        generated_ids.append(token_id)
        token_text = tok.vocab[token_id].decode("utf-8", errors="replace") if token_id < tok.vocab_size else f"[{token_id}]"

        top_tokens = []
        for tid, prob in zip(step["top_ids"], step["top_probs"]):
            t = tok.vocab[tid].decode("utf-8", errors="replace") if tid < tok.vocab_size else f"[{tid}]"
            top_tokens.append({"id": tid, "text": t, "probability": round(prob, 4)})

        token_details.append({
            "token_id": token_id,
            "token_text": token_text,
            "top_predictions": top_tokens,
        })

    return {
        "prompt": req.prompt,
        "generated_text": tok.decode(generated_ids),
        "total_tokens": len(steps),
        "tokens": token_details,
        "settings": {
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "strategy": req.strategy,
        },
    }
