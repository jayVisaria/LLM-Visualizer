"""
FastAPI application — Interactive LLM Visualizer backend.

Initializes the BPE tokenizer, GPT model, and trainer on startup,
then serves API endpoints for each transformer concept.

Run:
    cd backend
    uvicorn main:app --reload --port 8000
"""

import os
import sys

# Ensure the backend package is importable
sys.path.insert(0, os.path.dirname(__file__))

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.tokenizer import BPETokenizer, get_training_text
from core.model import GPT, GPTConfig
from core.trainer import Trainer, TrainConfig

from routers import tokenization, embeddings, positional, attention, transformer, training, generation
from routers import config as config_router


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
app_state: dict = {}

TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "data", "tokenizer.json")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------
def _heavy_init() -> None:
    """Run in a thread — loads tokenizer, model, trainer into app_state."""
    try:
        _do_init()
    except Exception as e:
        print(f"\n*** INIT FAILED: {e} ***\n")
        app_state["init_error"] = str(e)


def _do_init() -> None:
    print("=" * 60)
    print("  LLM Visualizer — Initializing (background)...")
    print("=" * 60)

    # 1. Load / train tokenizer
    print("\n[1/3] Preparing BPE tokenizer...")
    text = get_training_text()
    tok = BPETokenizer()

    if os.path.exists(TOKENIZER_PATH):
        print(f"  Loading saved tokenizer from {TOKENIZER_PATH}")
        tok.load(TOKENIZER_PATH)
    else:
        num_merges = int(os.environ.get("BPE_MERGES", "200"))
        print(f"  Training BPE tokenizer ({num_merges} merges)...")
        tok.train(text, num_merges=num_merges, verbose=False)
        tok.save(TOKENIZER_PATH)
        print(f"  Saved tokenizer to {TOKENIZER_PATH}")

    print(f"  Vocab size: {tok.vocab_size}")

    # 2. Initialize model
    print("\n[2/3] Initializing GPT model...")
    config = GPTConfig(vocab_size=tok.vocab_size)
    model = GPT(config)
    param_info = model.count_parameters()
    print(f"  Config: d_model={config.d_model}, n_heads={config.n_heads}, "
          f"n_layers={config.n_layers}, context_len={config.context_len}")
    print(f"  Total parameters: {param_info['total']:,}")

    # Load checkpoint if available
    ckpt_path = os.path.join(CHECKPOINT_DIR, "model.pt")
    ckpt = None
    if os.path.exists(ckpt_path):
        try:
            import torch
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"  Loaded checkpoint from {ckpt_path}")
        except Exception as e:
            print(f"  Warning: could not load checkpoint ({e}), using fresh model")
            ckpt = None

    # 3. Initialize trainer
    print("\n[3/3] Initializing trainer...")
    train_config = TrainConfig(checkpoint_dir=CHECKPOINT_DIR)
    trainer = Trainer(model, tok, train_config)

    if ckpt is not None:
        trainer.step = ckpt.get("step", 0)
        trainer.loss_history = ckpt.get("loss_history", [])
        print(f"  Resumed from step {trainer.step}")

    # Wire up router dependencies
    tokenization.set_tokenizer(tok, text)
    embeddings.set_deps(model, tok)
    attention.set_deps(model, tok)
    transformer.set_deps(model, tok)
    training.set_deps(trainer, text)
    generation.set_deps(model, tok)

    app_state["tokenizer"] = tok
    app_state["model"] = model
    app_state["trainer"] = trainer
    app_state["text"] = text
    app_state["dataset_name"] = "tiny_shakespeare"
    app_state["checkpoint_dir"] = CHECKPOINT_DIR

    config_router.set_app_state(app_state)
    app_state["ready"] = True

    print("\n" + "=" * 60)
    print("  Ready! API at http://localhost:8000/docs")
    print("=" * 60 + "\n")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Bind port immediately, then init model in background thread."""
    # Run heavy init in a thread so uvicorn binds the port right away
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, _heavy_init)

    yield  # App is serving (model may still be loading)

    print("\nShutting down...")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="LLM Visualizer API",
    description="Interactive visualization of how Large Language Models work — from tokenization to generation.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "https://jayvisaria.github.io",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(tokenization.router)
app.include_router(embeddings.router)
app.include_router(positional.router)
app.include_router(attention.router)
app.include_router(transformer.router)
app.include_router(training.router)
app.include_router(generation.router)
app.include_router(config_router.router)


@app.get("/api/health")
def health():
    if "init_error" in app_state:
        return {"status": "error", "error": app_state["init_error"]}
    ready = app_state.get("ready", False)
    return {
        "status": "ok" if ready else "initializing",
        "tokenizer_ready": "tokenizer" in app_state,
        "model_ready": "model" in app_state,
    }
