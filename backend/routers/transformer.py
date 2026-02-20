"""
Transformer & Full Model API router.

Endpoints:
    POST /api/transformer-block  — trace through a single transformer block
    POST /api/layernorm-demo     — LayerNorm visualization data
    POST /api/ffn-demo           — Feed-forward network visualization
    GET  /api/model-architecture — model config, parameter counts, layer info
    POST /api/forward-pass       — full forward pass trace with shapes/stats
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np

router = APIRouter(prefix="/api", tags=["transformer"])

_model = None
_tokenizer = None


def set_deps(model, tokenizer):
    global _model, _tokenizer
    _model = model
    _tokenizer = tokenizer


class TextRequest(BaseModel):
    text: str


@router.post("/transformer-block")
def transformer_block(req: TextRequest):
    """Trace data through a single transformer block showing all sublayers."""
    import torch
    from core.numpy_demos import layer_norm_demo, ffn_demo

    tok = _tokenizer
    if tok is None:
        raise HTTPException(503, "Not initialized")

    segments = tok.encode_with_segments(req.text)
    token_labels = [s["text"] for s in segments][:32]
    token_ids = [s["id"] for s in segments][:32]

    # NumPy demonstrations
    ln_demo = layer_norm_demo(seq_len=len(token_ids), d_model=16)
    ff_demo = ffn_demo(d_model=16, d_ff=64)

    result = {
        "token_labels": token_labels,
        "layernorm_demo": ln_demo,
        "ffn_demo": ff_demo,
    }

    # If model loaded, trace through actual first block
    if _model is not None:
        with torch.no_grad():
            ids_t = torch.tensor([token_ids])
            _model.eval()

            # Get embeddings
            tok_emb = _model.token_embedding(ids_t)
            pos_emb = _model.position_embedding(_model.position_ids[:, :len(token_ids)])
            x = tok_emb + pos_emb

            block = _model.blocks[0]

            # Pre-norm
            x_ln1 = block.ln1(x)
            # Attention
            x_attn = block.attn(x_ln1)
            # Residual
            x_res1 = x + x_attn
            # Pre-norm 2
            x_ln2 = block.ln2(x_res1)
            # FFN
            x_ffn = block.ffn(x_ln2)
            # Residual 2
            x_res2 = x_res1 + x_ffn

            def stats(t):
                return {
                    "mean": float(t.mean()),
                    "std": float(t.std()),
                    "min": float(t.min()),
                    "max": float(t.max()),
                    "shape": list(t.shape),
                }

            result["pytorch_trace"] = {
                "input": stats(x),
                "after_ln1": stats(x_ln1),
                "after_attention": stats(x_attn),
                "after_residual_1": stats(x_res1),
                "after_ln2": stats(x_ln2),
                "after_ffn": stats(x_ffn),
                "after_residual_2": stats(x_res2),
            }

            # Activation distributions for histograms
            result["distributions"] = {
                "input": x[0].numpy().flatten().tolist()[:200],
                "after_ln1": x_ln1[0].numpy().flatten().tolist()[:200],
                "after_residual_2": x_res2[0].numpy().flatten().tolist()[:200],
            }

    result["explanation"] = (
        "A transformer block has two sub-layers, each with a residual connection:\n"
        "1. LayerNorm → Multi-Head Self-Attention → Add (residual)\n"
        "2. LayerNorm → Feed-Forward Network → Add (residual)\n\n"
        "The residual connections (x = x + sublayer(x)) allow gradients to flow "
        "directly through the network, preventing vanishing gradients. "
        "LayerNorm stabilizes the activations before each sub-layer."
    )

    return result


@router.post("/layernorm-demo")
def layernorm_demo():
    """Return LayerNorm step-by-step computation for visualization."""
    from core.numpy_demos import layer_norm_demo
    return layer_norm_demo()


@router.post("/ffn-demo")
def ffn_demo_endpoint():
    """Return FFN computation steps for visualization."""
    from core.numpy_demos import ffn_demo
    return ffn_demo()


@router.get("/model-architecture")
def model_architecture():
    """Return the full model architecture description."""
    if _model is None:
        raise HTTPException(503, "Model not initialized")
    return _model.get_architecture_info()


@router.post("/forward-pass")
def forward_pass(req: TextRequest):
    """Run a full forward pass and return tensor shapes/stats at each stage."""
    import torch

    tok = _tokenizer
    if tok is None or _model is None:
        raise HTTPException(503, "Not initialized")

    token_ids = tok.encode(req.text)[:_model.config.context_len]
    segments = tok.encode_with_segments(req.text)[:_model.config.context_len]
    token_labels = [s["text"] for s in segments]

    with torch.no_grad():
        _model.eval()
        ids_t = torch.tensor([token_ids])
        trace = _model.forward_trace(ids_t)

        # Also get logits for the last token
        logits, _ = _model(ids_t)
        last_logits = logits[0, -1]  # (vocab_size,)
        probs = torch.softmax(last_logits, dim=-1)
        top_probs, top_ids = torch.topk(probs, min(20, probs.size(-1)))

        top_tokens = []
        for tid, p in zip(top_ids.tolist(), top_probs.tolist()):
            top_tokens.append({
                "id": tid,
                "text": tok.vocab[tid].decode("utf-8", errors="replace") if tid < tok.vocab_size else f"[{tid}]",
                "probability": round(p, 4),
            })

    return {
        "token_labels": token_labels,
        "trace": trace,
        "next_token_predictions": top_tokens,
    }
