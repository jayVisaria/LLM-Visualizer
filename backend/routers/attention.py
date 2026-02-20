"""
Attention API router.

Endpoints:
    POST /api/self-attention         — scaled dot-product attention (NumPy demo)
    POST /api/multi-head-attention   — multi-head attention (NumPy demo)
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np

router = APIRouter(prefix="/api", tags=["attention"])

_tokenizer = None
_model = None


def set_deps(model, tokenizer):
    global _model, _tokenizer
    _model = model
    _tokenizer = tokenizer


class AttentionRequest(BaseModel):
    text: str
    show_masking: bool = True
    n_heads: int = 4


@router.post("/self-attention")
def self_attention(req: AttentionRequest):
    """
    Compute scaled dot-product attention step-by-step and return
    all intermediate matrices for visualization.
    """
    from core.attention import scaled_dot_product_attention_numpy

    tok = _tokenizer
    if tok is None:
        raise HTTPException(503, "Not initialized")

    # Tokenize and get labels
    segments = tok.encode_with_segments(req.text)
    token_labels = [s["text"] for s in segments]
    token_ids = [s["id"] for s in segments]

    # Limit sequence length for visualization clarity
    max_seq = 32
    if len(token_ids) > max_seq:
        token_ids = token_ids[:max_seq]
        token_labels = token_labels[:max_seq]
        segments = segments[:max_seq]

    seq_len = len(token_ids)
    d_model = 64
    d_k = 16

    # Create random embeddings and project to Q, K, V
    rng = np.random.RandomState(42)
    X = rng.randn(seq_len, d_model).astype(np.float32) * 0.1
    W_q = rng.randn(d_model, d_k).astype(np.float32) * 0.1
    W_k = rng.randn(d_model, d_k).astype(np.float32) * 0.1
    W_v = rng.randn(d_model, d_k).astype(np.float32) * 0.1

    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v

    # Causal mask
    mask = np.tril(np.ones((seq_len, seq_len))) if req.show_masking else None

    result = scaled_dot_product_attention_numpy(Q, K, V, mask)
    result["token_labels"] = token_labels
    result["token_ids"] = token_ids
    result["Q"] = Q.tolist()
    result["K"] = K.tolist()
    result["V"] = V.tolist()
    result["seq_len"] = seq_len
    result["causal_masking"] = req.show_masking

    result["math"] = {
        "step1": "Q·Kᵀ — raw attention scores (dot product of queries and keys)",
        "step2": f"Q·Kᵀ / √{result['d_k']} — scale to prevent vanishing gradients in softmax",
        "step3": "Apply causal mask (set future positions to -∞)" if req.show_masking else "No masking",
        "step4": "softmax(scores) — normalize each row to probability distribution",
        "step5": "attention_weights · V — weighted sum of value vectors",
    }

    # If model is loaded, also compute with PyTorch for comparison
    if _model is not None:
        import torch
        with torch.no_grad():
            ids_tensor = torch.tensor([token_ids])
            _model.eval()
            _model(ids_tensor)  # forward pass to populate attention weights
            # Get attention weights from first block
            attn_w = _model.blocks[0].attn.get_attention_weights()
            if attn_w is not None:
                result["pytorch_attention"] = {
                    "weights": attn_w[0].numpy().tolist(),  # (n_heads, T, T)
                    "n_heads": attn_w.shape[1],
                }

    return result


@router.post("/multi-head-attention")
def multi_head_attention(req: AttentionRequest):
    """
    Compute multi-head attention and return per-head attention matrices.
    """
    from core.attention import multi_head_attention_numpy

    tok = _tokenizer
    if tok is None:
        raise HTTPException(503, "Not initialized")

    segments = tok.encode_with_segments(req.text)
    token_labels = [s["text"] for s in segments]
    token_ids = [s["id"] for s in segments]

    max_seq = 32
    if len(token_ids) > max_seq:
        token_ids = token_ids[:max_seq]
        token_labels = token_labels[:max_seq]

    seq_len = len(token_ids)

    # Use the live model's d_model so 8 heads → d_k = d_model/8, not always 8.
    # Fall back to 64 only if no model is loaded yet.
    d_model = _model.config.d_model if _model is not None else 64

    # Derive X from token IDs so different input text produces different patterns.
    # (A fixed seed=42 matrix meant every sentence gave identical heatmaps.)
    token_seed = int(sum(tid * (i + 1) for i, tid in enumerate(token_ids))) % (2 ** 31)
    rng_x = np.random.RandomState(token_seed)
    # N(0,1) embeddings — no tiny scaling, so after Xavier-init projections
    # the scores land in a reasonable range and heads diverge visually.
    X = rng_x.randn(seq_len, d_model).astype(np.float32)

    result = multi_head_attention_numpy(X, n_heads=req.n_heads, causal=req.show_masking)
    result["token_labels"] = token_labels
    result["token_ids"] = token_ids

    result["explanation"] = (
        f"Multi-head attention splits the {d_model}-dim embedding into "
        f"{req.n_heads} heads of {d_model // req.n_heads} dims each. "
        "Each head can learn different attention patterns (e.g., one head "
        "might focus on adjacent tokens, another on syntactic dependencies). "
        "The outputs are concatenated and projected back to d_model dimensions."
    )

    return result
