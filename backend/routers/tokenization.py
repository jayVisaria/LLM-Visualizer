"""
Tokenization API router.

Endpoints:
    POST /api/tokenize          — tokenize input text, return segments
    GET  /api/bpe-merges        — return BPE merge history
    GET  /api/vocab-stats       — vocabulary statistics
    POST /api/train-tokenizer   — train BPE tokenizer on corpus
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/api", tags=["tokenization"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class TokenizeRequest(BaseModel):
    text: str


class TrainTokenizerRequest(BaseModel):
    num_merges: int = 500


# ---------------------------------------------------------------------------
# Module-level state (populated by main.py on startup)
# ---------------------------------------------------------------------------

_tokenizer = None
_training_text = None
_cached_vocab_stats = None


def set_tokenizer(tok, text):
    global _tokenizer, _training_text, _cached_vocab_stats
    _tokenizer = tok
    _training_text = text
    # Pre-compute vocab stats so /vocab-stats is instant
    _cached_vocab_stats = tok.get_vocab_stats(text)


def get_tokenizer():
    if _tokenizer is None:
        raise HTTPException(503, "Tokenizer not initialized")
    return _tokenizer


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/tokenize")
def tokenize(req: TokenizeRequest):
    """Tokenize input text and return per-token segments with metadata."""
    tok = get_tokenizer()
    segments = tok.encode_with_segments(req.text)
    ids = [s["id"] for s in segments]
    return {
        "input_text": req.text,
        "tokens": segments,
        "token_ids": ids,
        "num_tokens": len(ids),
        "input_bytes": len(req.text.encode("utf-8")),
        "compression_ratio": round(
            len(req.text.encode("utf-8")) / max(len(ids), 1), 2
        ),
    }


@router.get("/bpe-merges")
def bpe_merges(limit: int = 100, offset: int = 0):
    """Return the BPE merge history (paginated)."""
    tok = get_tokenizer()
    history = tok.merge_history[offset : offset + limit]
    return {
        "total_merges": len(tok.merge_history),
        "offset": offset,
        "limit": limit,
        "merges": history,
    }


@router.get("/vocab-stats")
def vocab_stats():
    """Return vocabulary statistics and token frequency distribution (cached)."""
    if _cached_vocab_stats is not None:
        return _cached_vocab_stats
    tok = get_tokenizer()
    return tok.get_vocab_stats(_training_text)


@router.post("/train-tokenizer")
def train_tokenizer(req: TrainTokenizerRequest):
    """Re-train BPE tokenizer with a different number of merges."""
    from core.tokenizer import BPETokenizer, get_training_text

    text = get_training_text()
    tok = BPETokenizer()
    tok.train(text, num_merges=req.num_merges, verbose=False)
    set_tokenizer(tok, text)  # also refreshes cached vocab stats
    return {
        "status": "ok",
        "vocab_size": tok.vocab_size,
        "num_merges": len(tok.merges),
    }
