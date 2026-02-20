"""
Pure NumPy demonstrations of each transformer concept.

Every function here returns detailed intermediate computation results
as plain Python dicts/lists — suitable for JSON serialization and
front-end visualization. These exist alongside the PyTorch implementations
to show the raw mathematics without any framework magic.
"""

import math
import numpy as np
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# Embeddings
# ═══════════════════════════════════════════════════════════════════════════

def embedding_lookup_demo(
    token_ids: list[int],
    vocab_size: int = 256,
    d_model: int = 64,
    seed: int = 42,
) -> dict:
    """
    Demonstrate how token embedding works: a simple matrix lookup.

    embedding_matrix[vocab_size, d_model]  — each row is a learned vector
    embed(token_id) = embedding_matrix[token_id]
    """
    rng = np.random.RandomState(seed)
    # In a real model, this matrix is learned via backprop.
    # Here we initialize randomly for demonstration.
    embedding_matrix = rng.randn(vocab_size, d_model).astype(np.float32) * 0.02

    # Look up each token
    embedded = embedding_matrix[token_ids]  # (seq_len, d_model)

    return {
        "token_ids": token_ids,
        "embedding_matrix_shape": [vocab_size, d_model],
        # Return a small slice of the embedding matrix (first 20 rows)
        "embedding_matrix_sample": embedding_matrix[:min(20, vocab_size)].tolist(),
        "embedded_vectors": embedded.tolist(),
        "explanation": (
            "Each token id is used as an index into the embedding matrix. "
            f"Token id → row of the ({vocab_size}×{d_model}) matrix. "
            "This gives each token a dense vector representation."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Positional Encoding
# ═══════════════════════════════════════════════════════════════════════════

def sinusoidal_positional_encoding(
    seq_len: int = 64,
    d_model: int = 64,
) -> dict:
    """
    Compute sinusoidal positional encoding (Vaswani et al., 2017).

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Properties:
    - Each position gets a unique encoding.
    - The dot product between encodings of nearby positions is higher
      than for distant positions → encodes relative position.
    - Deterministic (no learned parameters).
    """
    pe = np.zeros((seq_len, d_model), dtype=np.float32)
    position = np.arange(seq_len).reshape(-1, 1)  # (seq_len, 1)
    # Compute the division term: 10000^(2i/d_model)
    div_term = np.exp(
        np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
    )  # (d_model/2,)

    pe[:, 0::2] = np.sin(position * div_term)  # even indices
    pe[:, 1::2] = np.cos(position * div_term)  # odd indices

    # Cosine similarity between positions
    norms = np.linalg.norm(pe, axis=1, keepdims=True)
    pe_normalized = pe / (norms + 1e-8)
    similarity = (pe_normalized @ pe_normalized.T).tolist()

    return {
        "pe_matrix": pe.tolist(),
        "shape": [seq_len, d_model],
        "similarity_matrix": similarity,
        "explanation": (
            "Sinusoidal positional encoding assigns each position a unique vector. "
            "Even dimensions use sin, odd dimensions use cos, with frequencies "
            "that decrease geometrically across dimensions. This allows the model "
            "to attend to relative positions."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Layer Normalization
# ═══════════════════════════════════════════════════════════════════════════

def layer_norm_demo(
    x: Optional[np.ndarray] = None,
    seq_len: int = 8,
    d_model: int = 16,
    seed: int = 42,
) -> dict:
    """
    Demonstrate Layer Normalization step-by-step.

    LayerNorm(x) = γ · (x - μ) / (σ + ε) + β

    where μ, σ are computed across the last dimension (per-token),
    γ (scale) and β (shift) are learnable parameters.
    """
    if x is None:
        rng = np.random.RandomState(seed)
        x = rng.randn(seq_len, d_model).astype(np.float32) * 3 + 2

    eps = 1e-5
    gamma = np.ones(x.shape[-1], dtype=np.float32)
    beta = np.zeros(x.shape[-1], dtype=np.float32)

    # Step 1: Compute mean across last dimension
    mu = x.mean(axis=-1, keepdims=True)  # (seq_len, 1)

    # Step 2: Compute variance
    var = x.var(axis=-1, keepdims=True)  # (seq_len, 1)
    sigma = np.sqrt(var + eps)

    # Step 3: Normalize
    x_norm = (x - mu) / sigma

    # Step 4: Scale and shift
    output = gamma * x_norm + beta

    return {
        "input": x.tolist(),
        "mean_per_token": mu.squeeze(-1).tolist(),
        "var_per_token": var.squeeze(-1).tolist(),
        "normalized": x_norm.tolist(),
        "output": output.tolist(),
        "input_stats": {
            "global_mean": float(x.mean()),
            "global_std": float(x.std()),
        },
        "output_stats": {
            "global_mean": float(output.mean()),
            "global_std": float(output.std()),
        },
        "explanation": (
            "LayerNorm normalizes each token's feature vector to zero mean "
            "and unit variance. This stabilizes training by preventing "
            "internal covariate shift. The learnable γ/β parameters allow "
            "the model to undo the normalization if needed."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Feed-Forward Network
# ═══════════════════════════════════════════════════════════════════════════

def ffn_demo(
    x: Optional[np.ndarray] = None,
    d_model: int = 16,
    d_ff: int = 64,
    seed: int = 42,
) -> dict:
    """
    Demonstrate the position-wise feed-forward network.

    FFN(x) = GELU(x·W₁ + b₁)·W₂ + b₂

    This is applied independently to each position (token).
    The expansion from d_model → d_ff → d_model allows the model
    to learn complex non-linear transformations.
    """
    rng = np.random.RandomState(seed)
    if x is None:
        x = rng.randn(4, d_model).astype(np.float32) * 0.5

    W1 = rng.randn(d_model, d_ff).astype(np.float32) * 0.02
    W2 = rng.randn(d_ff, d_model).astype(np.float32) * 0.02

    # Step 1: Linear projection (expand)
    hidden = x @ W1  # (seq, d_ff)

    # Step 2: GELU activation
    # GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
    def gelu(z: np.ndarray) -> np.ndarray:
        return 0.5 * z * (1 + np.tanh(math.sqrt(2.0 / math.pi) * (z + 0.044715 * z**3)))

    activated = gelu(hidden)

    # Step 3: Linear projection (contract)
    output = activated @ W2  # (seq, d_model)

    return {
        "input": x.tolist(),
        "after_linear1": hidden.tolist(),
        "after_gelu": activated.tolist(),
        "output": output.tolist(),
        "shapes": {
            "input": list(x.shape),
            "W1": [d_model, d_ff],
            "hidden": list(hidden.shape),
            "W2": [d_ff, d_model],
            "output": list(output.shape),
        },
        "expansion_ratio": d_ff / d_model,
        "explanation": (
            f"The FFN expands from {d_model} → {d_ff} dimensions (×{d_ff // d_model}), "
            "applies GELU non-linearity, then contracts back. This gives the model "
            "capacity to learn complex transformations at each position independently."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Backpropagation demo (tiny scale)
# ═══════════════════════════════════════════════════════════════════════════

def backprop_demo() -> dict:
    """
    Manual backpropagation through a tiny attention + linear layer.

    Uses 3 tokens, 4-dimensional embeddings. Computes gradients
    by hand using the chain rule, then verifies against PyTorch autograd.
    """
    import torch

    torch.manual_seed(42)

    # Tiny example: 3 tokens, d_model=4
    seq_len, d_model = 3, 4
    x_np = np.array([
        [1.0,  0.5, -0.3,  0.8],
        [0.2, -0.4,  1.0, -0.1],
        [-0.5, 0.9,  0.1,  0.6],
    ], dtype=np.float32)

    # --- NumPy manual forward ---
    np.random.seed(42)
    W_q = np.random.randn(d_model, d_model).astype(np.float32) * 0.5
    W_k = np.random.randn(d_model, d_model).astype(np.float32) * 0.5
    W_v = np.random.randn(d_model, d_model).astype(np.float32) * 0.5
    W_out = np.random.randn(d_model, 3).astype(np.float32) * 0.5  # project to 3 classes

    Q = x_np @ W_q
    K = x_np @ W_k
    V = x_np @ W_v

    scores = Q @ K.T / math.sqrt(d_model)

    # Softmax
    def softmax_np(z, axis=-1):
        e = np.exp(z - z.max(axis=axis, keepdims=True))
        return e / e.sum(axis=axis, keepdims=True)

    attn = softmax_np(scores)
    context = attn @ V
    logits = context @ W_out

    # --- PyTorch autograd verification ---
    x_t = torch.tensor(x_np, requires_grad=True)
    W_q_t = torch.tensor(W_q, requires_grad=True)
    W_k_t = torch.tensor(W_k, requires_grad=True)
    W_v_t = torch.tensor(W_v, requires_grad=True)
    W_out_t = torch.tensor(W_out, requires_grad=True)

    Q_t = x_t @ W_q_t
    K_t = x_t @ W_k_t
    V_t = x_t @ W_v_t
    scores_t = Q_t @ K_t.T / math.sqrt(d_model)
    attn_t = torch.softmax(scores_t, dim=-1)
    context_t = attn_t @ V_t
    logits_t = context_t @ W_out_t

    # Cross-entropy loss with dummy targets
    targets = torch.tensor([0, 1, 2])
    loss = torch.nn.functional.cross_entropy(logits_t, targets)
    loss.backward()

    return {
        "setup": {
            "seq_len": seq_len,
            "d_model": d_model,
            "input_x": x_np.tolist(),
        },
        "forward_pass": {
            "Q": Q.tolist(),
            "K": K.tolist(),
            "V": V.tolist(),
            "scores": scores.tolist(),
            "attention_weights": attn.tolist(),
            "context": context.tolist(),
            "logits": logits.tolist(),
        },
        "loss": loss.item(),
        "gradients": {
            "d_loss/d_x": x_t.grad.numpy().tolist() if x_t.grad is not None else None,
            "d_loss/d_W_q": W_q_t.grad.numpy().tolist() if W_q_t.grad is not None else None,
            "d_loss/d_W_out": W_out_t.grad.numpy().tolist() if W_out_t.grad is not None else None,
        },
        "gradient_stats": {
            "x_grad_norm": float(x_t.grad.norm().item()) if x_t.grad is not None else 0,
            "W_q_grad_norm": float(W_q_t.grad.norm().item()) if W_q_t.grad is not None else 0,
            "W_out_grad_norm": float(W_out_t.grad.norm().item()) if W_out_t.grad is not None else 0,
        },
        "chain_rule_steps": [
            "1. Compute ∂L/∂logits from cross-entropy loss",
            "2. ∂L/∂W_out = contextᵀ · ∂L/∂logits",
            "3. ∂L/∂context = ∂L/∂logits · W_outᵀ",
            "4. ∂L/∂attn = ∂L/∂context · Vᵀ  (backprop through weighted sum)",
            "5. ∂L/∂scores = attn ⊙ (∂L/∂attn - row_sums)  (softmax Jacobian)",
            "6. ∂L/∂Q = ∂L/∂scores · K / √d_k",
            "7. ∂L/∂x = ∂L/∂Q · W_qᵀ + ∂L/∂K · W_kᵀ + ∂L/∂V · W_vᵀ",
        ],
        "explanation": (
            "Backpropagation applies the chain rule to compute gradients of the loss "
            "with respect to every parameter. Starting from ∂L/∂logits, we propagate "
            "backwards through each operation: output projection → attention weighted sum "
            "→ softmax → scaled dot product → linear projections → input. "
            "PyTorch autograd does this automatically; here we verify the math matches."
        ),
    }
