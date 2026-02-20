"""
GPT-style decoder-only transformer model — built from scratch in PyTorch.

Architecture (matches GPT-2 design, scaled down for education):
    Token Embeddings  →  Positional Embeddings  →  Dropout
        ↓
    N × Transformer Block:
        ├─ LayerNorm → Multi-Head Causal Self-Attention → Residual Add
        └─ LayerNorm → Feed-Forward Network (GELU)     → Residual Add
        ↓
    Final LayerNorm → Linear Head (→ logits over vocab)

Config defaults (tiny, CPU-trainable):
    d_model=64, n_heads=4, n_layers=4, context_len=128, vocab_size=~756
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CausalSelfAttention


# ═══════════════════════════════════════════════════════════════════════════
# Model configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class GPTConfig:
    vocab_size: int = 756       # set from tokenizer
    context_len: int = 128      # max sequence length
    d_model: int = 64           # embedding dimension
    n_heads: int = 4            # attention heads
    n_layers: int = 4           # transformer blocks
    dropout: float = 0.1
    bias: bool = False          # no bias in linear layers (GPT-2 style)

    def to_dict(self) -> dict:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════════
# Feed-Forward Network
# ═══════════════════════════════════════════════════════════════════════════

class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.

        FFN(x) = GELU(x·W₁ + b₁)·W₂ + b₂

    Expands from d_model → 4*d_model → d_model (standard 4× expansion).
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model, bias=False)
        self.fc2 = nn.Linear(4 * d_model, d_model, bias=False)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# ═══════════════════════════════════════════════════════════════════════════
# Transformer Block
# ═══════════════════════════════════════════════════════════════════════════

class TransformerBlock(nn.Module):
    """
    Single transformer block with pre-norm architecture (GPT-2 style).

        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))

    Pre-norm (normalize before attention/FFN) is more stable than post-norm
    and is used by GPT-2, GPT-3, LLaMA, etc.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(
            config.d_model, config.n_heads, config.context_len, config.dropout
        )
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config.d_model, config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm + attention + residual
        x = x + self.attn(self.ln1(x))
        # Pre-norm + FFN + residual
        x = x + self.ffn(self.ln2(x))
        return x


# ═══════════════════════════════════════════════════════════════════════════
# Full GPT Model
# ═══════════════════════════════════════════════════════════════════════════

class GPT(nn.Module):
    """
    GPT-style decoder-only transformer language model.

    Forward pass:
        token_ids → token_embeds + pos_embeds → N × TransformerBlock → LayerNorm → logits
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_len, config.d_model)
        self.embed_dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        # Final layer norm + output head
        self.ln_final = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: share token embedding weights with the output head
        # This is standard practice (reduces parameters, improves performance)
        self.token_embedding.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Position indices buffer
        self.register_buffer(
            "position_ids",
            torch.arange(config.context_len).unsqueeze(0),  # (1, context_len)
        )

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights following GPT-2 conventions."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Parameters
        ----------
        input_ids : (batch, seq_len) — token ids
        targets   : (batch, seq_len) — target token ids (for loss computation)

        Returns
        -------
        logits : (batch, seq_len, vocab_size)
        loss   : scalar tensor if targets provided, else None
        """
        B, T = input_ids.shape
        assert T <= self.config.context_len, (
            f"Sequence length {T} exceeds context_len {self.config.context_len}"
        )

        # Token + positional embeddings
        tok_emb = self.token_embedding(input_ids)           # (B, T, d_model)
        pos_emb = self.position_embedding(self.position_ids[:, :T])  # (1, T, d_model)
        x = self.embed_dropout(tok_emb + pos_emb)           # (B, T, d_model)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final norm + project to vocab
        x = self.ln_final(x)                                # (B, T, d_model)
        logits = self.lm_head(x)                             # (B, T, vocab_size)

        # Loss (next-token prediction)
        loss = None
        if targets is not None:
            # Flatten: (B*T, vocab_size) vs (B*T,)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> list[dict]:
        """
        Autoregressive text generation with detailed per-step info.

        Returns a list of dicts, one per generated token:
            {token_id, token_probs_top20, cumulative_ids}
        """
        self.eval()
        steps = []

        for _ in range(max_new_tokens):
            # Crop to context_len
            idx_cond = input_ids[:, -self.config.context_len:]

            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # (B, vocab_size) — last position only

            # Temperature scaling
            if temperature != 1.0:
                logits = logits / temperature

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens with cumulative probability above threshold
                remove_mask = cum_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[remove_mask] = float("-inf")
                # Scatter back
                logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            # Probabilities
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)

            # Sample
            if temperature == 0:
                next_id = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                next_id = torch.multinomial(probs, num_samples=1)

            # Top-20 probs for visualization
            top_probs, top_ids = torch.topk(probs[0], min(20, probs.size(-1)))

            steps.append(
                {
                    "token_id": next_id[0, 0].item(),
                    "top_ids": top_ids.tolist(),
                    "top_probs": top_probs.tolist(),
                }
            )

            # Append to sequence
            input_ids = torch.cat([input_ids, next_id], dim=1)

        return steps

    # ------------------------------------------------------------------
    # Model info
    # ------------------------------------------------------------------
    def count_parameters(self) -> dict:
        """Return parameter count breakdown by component."""
        breakdown = {}
        total = 0
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            n = param.numel()
            # Group by top-level component
            component = name.split(".")[0]
            breakdown[component] = breakdown.get(component, 0) + n
            total += n
        return {"breakdown": breakdown, "total": total}

    def get_architecture_info(self) -> dict:
        """Return full architecture description for visualization."""
        param_info = self.count_parameters()
        return {
            "config": self.config.to_dict(),
            "parameters": param_info,
            "layers": [
                {
                    "name": "token_embedding",
                    "type": "Embedding",
                    "shape": f"({self.config.vocab_size}, {self.config.d_model})",
                    "params": self.token_embedding.weight.numel(),
                },
                {
                    "name": "position_embedding",
                    "type": "Embedding",
                    "shape": f"({self.config.context_len}, {self.config.d_model})",
                    "params": self.position_embedding.weight.numel(),
                },
            ]
            + [
                {
                    "name": f"block_{i}",
                    "type": "TransformerBlock",
                    "sublayers": [
                        "LayerNorm",
                        "CausalSelfAttention",
                        "LayerNorm",
                        "FeedForward",
                    ],
                    "params": sum(
                        p.numel() for p in self.blocks[i].parameters()
                    ),
                }
                for i in range(self.config.n_layers)
            ]
            + [
                {
                    "name": "ln_final",
                    "type": "LayerNorm",
                    "params": sum(
                        p.numel() for p in self.ln_final.parameters()
                    ),
                },
                {
                    "name": "lm_head",
                    "type": "Linear",
                    "shape": f"({self.config.d_model}, {self.config.vocab_size})",
                    "params": "(tied with token_embedding)",
                },
            ],
        }

    # ------------------------------------------------------------------
    # Forward pass trace (for visualization)
    # ------------------------------------------------------------------
    def forward_trace(self, input_ids: torch.Tensor) -> list[dict]:
        """
        Run forward pass and record tensor shapes / stats at each stage.
        """
        B, T = input_ids.shape
        trace = []

        tok_emb = self.token_embedding(input_ids)
        trace.append({"stage": "token_embedding", "shape": list(tok_emb.shape),
                       "mean": tok_emb.mean().item(), "std": tok_emb.std().item()})

        pos_emb = self.position_embedding(self.position_ids[:, :T])
        trace.append({"stage": "position_embedding", "shape": list(pos_emb.shape),
                       "mean": pos_emb.mean().item(), "std": pos_emb.std().item()})

        x = self.embed_dropout(tok_emb + pos_emb)
        trace.append({"stage": "embed_sum+dropout", "shape": list(x.shape),
                       "mean": x.mean().item(), "std": x.std().item()})

        for i, block in enumerate(self.blocks):
            x = block(x)
            trace.append({
                "stage": f"block_{i}",
                "shape": list(x.shape),
                "mean": x.mean().item(),
                "std": x.std().item(),
            })

        x = self.ln_final(x)
        trace.append({"stage": "ln_final", "shape": list(x.shape),
                       "mean": x.mean().item(), "std": x.std().item()})

        logits = self.lm_head(x)
        trace.append({"stage": "lm_head (logits)", "shape": list(logits.shape),
                       "mean": logits.mean().item(), "std": logits.std().item()})

        return trace
