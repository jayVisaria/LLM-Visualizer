"""
Attention modules — NumPy (educational) and PyTorch (trainable) implementations.

Covers:
- Scaled Dot-Product Attention
- Causal (autoregressive) masking
- Multi-Head Attention

The NumPy versions exist to expose every intermediate computation for
visualization. The PyTorch versions are used in the actual model.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# NumPy implementations (educational — full intermediate outputs)
# ═══════════════════════════════════════════════════════════════════════════

def softmax_numpy(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically stable softmax.

        softmax(x_i) = exp(x_i - max(x)) / Σ exp(x_j - max(x))

    Subtracting the max prevents overflow in exp().
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def scaled_dot_product_attention_numpy(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> dict:
    """
    Compute scaled dot-product attention with full intermediate outputs.

    Parameters
    ----------
    Q : (seq_len, d_k)  — query vectors
    K : (seq_len, d_k)  — key vectors
    V : (seq_len, d_v)  — value vectors
    mask : optional (seq_len, seq_len) — 1 = keep, 0 = mask out

    Returns
    -------
    dict with keys:
        raw_scores      — Q·Kᵀ                          (seq, seq)
        scaled_scores   — Q·Kᵀ / √d_k                   (seq, seq)
        masked_scores   — after applying causal mask      (seq, seq) or None
        attention_weights — softmax(scores)               (seq, seq)
        output          — attention_weights · V           (seq, d_v)

    Math
    ----
        Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k ) · V
    """
    d_k = Q.shape[-1]

    # Step 1: Q·Kᵀ
    raw_scores = Q @ K.T  # (seq, seq)

    # Step 2: Scale by √d_k
    scaled_scores = raw_scores / math.sqrt(d_k)

    # Step 3: Apply causal mask (optional)
    masked_scores = None
    if mask is not None:
        masked_scores = scaled_scores.copy()
        masked_scores[mask == 0] = -1e9
        scores_for_softmax = masked_scores
    else:
        scores_for_softmax = scaled_scores

    # Step 4: Softmax
    attention_weights = softmax_numpy(scores_for_softmax, axis=-1)

    # Step 5: Weighted sum of values
    output = attention_weights @ V  # (seq, d_v)

    return {
        "raw_scores": raw_scores.tolist(),
        "scaled_scores": scaled_scores.tolist(),
        "masked_scores": masked_scores.tolist() if masked_scores is not None else None,
        "attention_weights": attention_weights.tolist(),
        "output": output.tolist(),
        "d_k": d_k,
        "scale_factor": math.sqrt(d_k),
    }


def multi_head_attention_numpy(
    X: np.ndarray,
    n_heads: int = 4,
    causal: bool = True,
    seed: int = 42,
) -> dict:
    """
    Multi-head attention in NumPy with full intermediates.

    Parameters
    ----------
    X : (seq_len, d_model) — input embeddings
    n_heads : number of attention heads
    causal : whether to apply causal (look-ahead) mask

    Returns
    -------
    dict with per-head attention weights, concatenated output, etc.
    """
    rng = np.random.RandomState(seed)
    seq_len, d_model = X.shape
    assert d_model % n_heads == 0
    d_k = d_model // n_heads

    # Random projection weights — Xavier init so scores have real variance.
    # (0.02 was far too small: scores collapsed to ~0 and softmax became uniform
    #  across all heads, making every heatmap look identical.)
    scale = 1.0 / np.sqrt(d_model)
    W_q = rng.randn(d_model, d_model).astype(np.float32) * scale
    W_k = rng.randn(d_model, d_model).astype(np.float32) * scale
    W_v = rng.randn(d_model, d_model).astype(np.float32) * scale
    W_o = rng.randn(d_model, d_model).astype(np.float32) * scale

    # Project to Q, K, V
    Q_full = X @ W_q  # (seq, d_model)
    K_full = X @ W_k
    V_full = X @ W_v

    # Causal mask
    mask = np.tril(np.ones((seq_len, seq_len))) if causal else None

    # Split into heads and compute attention for each
    head_outputs = []
    head_attention_weights = []
    for h in range(n_heads):
        start = h * d_k
        end = start + d_k
        Q_h = Q_full[:, start:end]
        K_h = K_full[:, start:end]
        V_h = V_full[:, start:end]

        result = scaled_dot_product_attention_numpy(Q_h, K_h, V_h, mask)
        head_outputs.append(np.array(result["output"]))
        head_attention_weights.append(result["attention_weights"])

    # Concatenate heads
    concatenated = np.concatenate(head_outputs, axis=-1)  # (seq, d_model)

    # Output projection
    output = concatenated @ W_o  # (seq, d_model)

    return {
        "head_attention_weights": head_attention_weights,
        "concatenated_shape": list(concatenated.shape),
        "output": output.tolist(),
        "n_heads": n_heads,
        "d_k": d_k,
        "d_model": d_model,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PyTorch implementations (used in the actual model)
# ═══════════════════════════════════════════════════════════════════════════

class CausalSelfAttention(nn.Module):
    """
    Single causal (masked) self-attention head.

    This module implements:
        Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k + M ) · V

    where M is a causal mask that prevents attending to future tokens.
    """

    def __init__(self, d_model: int, n_heads: int, context_len: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Combined QKV projection for efficiency
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask: lower-triangular matrix
        # Registered as buffer so it moves to GPU with the model but isn't a parameter
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(context_len, context_len)).view(
                1, 1, context_len, context_len
            ),
        )

        # Storage for attention weights (for visualization)
        self._attn_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, d_model)

        Returns
        -------
        (batch, seq_len, d_model)
        """
        B, T, C = x.shape

        # Project to Q, K, V and split into heads
        qkv = self.qkv_proj(x)  # (B, T, 3*C)
        q, k, v = qkv.split(self.d_model, dim=2)

        # Reshape: (B, T, C) → (B, n_heads, T, d_k)
        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        # (B, n_heads, T, d_k) @ (B, n_heads, d_k, T) → (B, n_heads, T, T)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply causal mask
        scores = scores.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0, float("-inf")
        )

        # Softmax + dropout
        attn_weights = F.softmax(scores, dim=-1)
        self._attn_weights = attn_weights.detach()  # save for visualization
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum: (B, n_heads, T, T) @ (B, n_heads, T, d_k) → (B, n_heads, T, d_k)
        out = attn_weights @ v

        # Reshape back: (B, n_heads, T, d_k) → (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection + dropout
        out = self.resid_dropout(self.out_proj(out))
        return out

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Return last-computed attention weights (B, n_heads, T, T)."""
        return self._attn_weights
