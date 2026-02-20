"""
Training loop for the GPT model.

Handles:
- Data preparation (sliding-window batching of tokenized text)
- AdamW optimizer with cosine learning rate schedule + warmup
- Training loop with loss logging and periodic sample generation
- Checkpoint save/load
"""

from __future__ import annotations

import os
import math
import time
import json
from dataclasses import dataclass, asdict
from typing import Optional, Generator

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .model import GPT, GPTConfig
from .tokenizer import BPETokenizer


# ═══════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════

class TextDataset(Dataset):
    """
    Simple sliding-window dataset for language modelling.

    Given a long sequence of token ids [t₁, t₂, …, tₙ]:
        input  = [tᵢ, tᵢ₊₁, …, tᵢ₊ₗ₋₁]
        target = [tᵢ₊₁, tᵢ₊₂, …, tᵢ₊ₗ]

    where l = context_len.
    """

    def __init__(self, token_ids: list[int], context_len: int):
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.context_len = context_len

    def __len__(self) -> int:
        return max(0, len(self.data) - self.context_len)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.context_len]
        y = self.data[idx + 1 : idx + self.context_len + 1]
        return x, y


# ═══════════════════════════════════════════════════════════════════════════
# Training configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TrainConfig:
    max_steps: int = 2000
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 100
    log_interval: int = 50
    sample_interval: int = 500
    sample_length: int = 100
    checkpoint_dir: str = "checkpoints"
    grad_clip: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════════
# Learning rate schedule
# ═══════════════════════════════════════════════════════════════════════════

def cosine_lr_schedule(step: int, warmup: int, max_steps: int, lr: float) -> float:
    """
    Cosine learning rate schedule with linear warmup.

        if step < warmup:
            lr_t = lr × (step / warmup)             # linear warmup
        else:
            progress = (step - warmup) / (max_steps - warmup)
            lr_t = lr × 0.5 × (1 + cos(π × progress))  # cosine decay
    """
    if step < warmup:
        return lr * step / max(warmup, 1)
    if step >= max_steps:
        return lr * 0.1  # minimum lr
    progress = (step - warmup) / (max_steps - warmup)
    return lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ═══════════════════════════════════════════════════════════════════════════
# Trainer
# ═══════════════════════════════════════════════════════════════════════════

class Trainer:
    """Manages the training loop for the GPT model."""

    def __init__(
        self,
        model: GPT,
        tokenizer: BPETokenizer,
        train_config: TrainConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = train_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Optimizer: AdamW with weight decay
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
            betas=(0.9, 0.95),
        )

        # State
        self.step = 0
        self.loss_history: list[dict] = []
        self.lr_history: list[dict] = []
        self.samples: list[dict] = []
        self._stop_requested = False

    def prepare_data(self, text: str) -> DataLoader:
        """Tokenize text and create DataLoader."""
        token_ids = self.tokenizer.encode(text)
        dataset = TextDataset(token_ids, self.model.config.context_len)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
        )

    def train(self, text: str) -> Generator[dict, None, None]:
        """
        Train the model, yielding progress events (for SSE streaming).

        Yields
        ------
        dict with one of:
            {"type": "step", "step": int, "loss": float, "lr": float, "perplexity": float}
            {"type": "sample", "step": int, "text": str}
            {"type": "done", "final_loss": float, "total_steps": int}
        """
        dataloader = self.prepare_data(text)
        data_iter = iter(dataloader)
        self.model.train()
        self._stop_requested = False

        start_time = time.time()

        for step in range(self.step, self.config.max_steps):
            if self._stop_requested:
                yield {"type": "stopped", "step": step}
                return

            # Get batch (restart iterator when exhausted)
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                x, y = next(data_iter)

            x, y = x.to(self.device), y.to(self.device)

            # Learning rate schedule
            lr = cosine_lr_schedule(
                step, self.config.warmup_steps, self.config.max_steps,
                self.config.learning_rate
            )
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            # Forward pass
            logits, loss = self.model(x, y)

            # Backward pass
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Gradient clipping
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )

            # Update weights
            self.optimizer.step()

            self.step = step + 1
            loss_val = loss.item()
            perplexity = math.exp(min(loss_val, 20))  # cap to avoid inf

            # Record history
            self.loss_history.append({
                "step": self.step,
                "loss": round(loss_val, 4),
                "perplexity": round(perplexity, 2),
                "lr": lr,
            })
            self.lr_history.append({"step": self.step, "lr": lr})

            # Log progress
            if self.step % self.config.log_interval == 0:
                elapsed = time.time() - start_time
                yield {
                    "type": "step",
                    "step": self.step,
                    "loss": round(loss_val, 4),
                    "lr": round(lr, 6),
                    "perplexity": round(perplexity, 2),
                    "elapsed": round(elapsed, 1),
                    "steps_per_sec": round(self.step / elapsed, 1),
                }

            # Generate sample
            if self.step % self.config.sample_interval == 0:
                sample_text = self._generate_sample()
                self.samples.append({"step": self.step, "text": sample_text})
                yield {
                    "type": "sample",
                    "step": self.step,
                    "text": sample_text,
                }

        # Done
        self.save_checkpoint()
        final_loss = self.loss_history[-1]["loss"] if self.loss_history else 0
        yield {
            "type": "done",
            "final_loss": final_loss,
            "total_steps": self.step,
            "elapsed": round(time.time() - start_time, 1),
        }

    def _generate_sample(self, prompt: str = "\n") -> str:
        """Generate a text sample from the current model."""
        self.model.eval()
        ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        steps = self.model.generate(
            input_ids,
            max_new_tokens=self.config.sample_length,
            temperature=0.8,
            top_k=40,
        )
        all_ids = ids + [s["token_id"] for s in steps]
        self.model.train()
        return self.tokenizer.decode(all_ids)

    def stop(self) -> None:
        """Request graceful stop of training."""
        self._stop_requested = True

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------
    def save_checkpoint(self, path: Optional[str] = None) -> str:
        """Save model, optimizer, and training state."""
        ckpt_dir = path or self.config.checkpoint_dir
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, "model.pt")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "step": self.step,
                "config": self.model.config.to_dict(),
                "loss_history": self.loss_history,
            },
            ckpt_path,
        )
        return ckpt_path

    def load_checkpoint(self, path: Optional[str] = None) -> None:
        """Load model and training state from checkpoint."""
        ckpt_dir = path or self.config.checkpoint_dir
        ckpt_path = os.path.join(ckpt_dir, "model.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"No checkpoint at {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.step = ckpt.get("step", 0)
        self.loss_history = ckpt.get("loss_history", [])

    # ------------------------------------------------------------------
    # Gradient analysis (for backpropagation visualization)
    # ------------------------------------------------------------------
    def get_gradient_flow(self) -> list[dict]:
        """
        Return gradient magnitudes per named parameter group.
        Must be called after a backward pass (before optimizer.zero_grad).
        """
        flow = []
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                flow.append({
                    "name": name,
                    "mean_grad": param.grad.abs().mean().item(),
                    "max_grad": param.grad.abs().max().item(),
                    "shape": list(param.shape),
                })
        return flow
