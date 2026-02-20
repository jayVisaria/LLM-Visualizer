"""
Configuration & model management API router.

Endpoints:
    GET  /api/config              — get current model/tokenizer/trainer config
    POST /api/config              — update config (does NOT rebuild, just stores)
    POST /api/rebuild-model       — rebuild model with new config
    GET  /api/status              — model/training status summary
    GET  /api/datasets            — list available datasets
    POST /api/select-dataset      — switch active dataset
    POST /api/upload-data         — upload custom training text
    POST /api/snapshot            — save named model snapshot
    GET  /api/snapshots           — list saved snapshots
    POST /api/load-snapshot       — load a saved snapshot
"""

import os
import json
import time
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

router = APIRouter(prefix="/api", tags=["config"])

# ---------------------------------------------------------------------------
# Module-level state (set by main.py)
# ---------------------------------------------------------------------------
_app_state: dict = {}

PRESETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "presets")
SNAPSHOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "snapshots")
CUSTOM_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "custom.txt")


def set_app_state(state: dict):
    global _app_state
    _app_state = state


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class ModelConfigRequest(BaseModel):
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 4
    context_len: int = 128
    dropout: float = 0.1

class TrainConfigRequest(BaseModel):
    max_steps: int = 2000
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 100
    grad_clip: float = 1.0

class UpdateConfigRequest(BaseModel):
    model_config_data: Optional[ModelConfigRequest] = None
    train_config: Optional[TrainConfigRequest] = None

class RebuildRequest(BaseModel):
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 4
    context_len: int = 128
    dropout: float = 0.1

class SelectDatasetRequest(BaseModel):
    name: str  # "tiny_shakespeare", "custom", or a preset name

class SnapshotRequest(BaseModel):
    name: str
    description: str = ""

class LoadSnapshotRequest(BaseModel):
    name: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/config")
def get_config():
    """Return current model, tokenizer, and trainer configuration."""
    model = _app_state.get("model")
    trainer = _app_state.get("trainer")
    tok = _app_state.get("tokenizer")

    result = {
        "model": model.config.to_dict() if model else None,
        "model_params": model.count_parameters() if model else None,
        "tokenizer": {
            "vocab_size": tok.vocab_size if tok else 0,
            "num_merges": len(tok.merges) if tok else 0,
        },
        "training": trainer.config.to_dict() if trainer else None,
        "dataset": _app_state.get("dataset_name", "tiny_shakespeare"),
    }
    return result


@router.post("/config")
def update_config(req: UpdateConfigRequest):
    """Update stored config (does NOT rebuild model — call /rebuild-model for that)."""
    result = {}
    model_config = req.model_config_data
    train_config = req.train_config
    if model_config:
        result["model"] = model_config.dict()
    if train_config and _app_state.get("trainer"):
        trainer = _app_state["trainer"]
        trainer.config.max_steps = train_config.max_steps
        trainer.config.batch_size = train_config.batch_size
        trainer.config.learning_rate = train_config.learning_rate
        trainer.config.weight_decay = train_config.weight_decay
        trainer.config.warmup_steps = train_config.warmup_steps
        trainer.config.grad_clip = train_config.grad_clip
        result["training"] = trainer.config.to_dict()
    return {"status": "ok", "updated": result}


@router.post("/rebuild-model")
def rebuild_model(req: RebuildRequest):
    """Rebuild the model from scratch with a new architecture config."""
    import torch
    from core.model import GPT, GPTConfig
    from core.trainer import Trainer, TrainConfig
    from routers import embeddings, attention, transformer, training, generation, tokenization

    tok = _app_state.get("tokenizer")
    text = _app_state.get("text")
    if tok is None:
        raise HTTPException(503, "Tokenizer not initialized")

    # Validate config
    if req.d_model % req.n_heads != 0:
        raise HTTPException(400, f"d_model ({req.d_model}) must be divisible by n_heads ({req.n_heads})")

    config = GPTConfig(
        vocab_size=tok.vocab_size,
        d_model=req.d_model,
        n_heads=req.n_heads,
        n_layers=req.n_layers,
        context_len=req.context_len,
        dropout=req.dropout,
    )
    model = GPT(config)
    param_info = model.count_parameters()

    # Reset trainer
    checkpoint_dir = _app_state.get("checkpoint_dir", "checkpoints")
    train_config = TrainConfig(checkpoint_dir=checkpoint_dir)
    trainer = Trainer(model, tok, train_config)

    # Update global state
    _app_state["model"] = model
    _app_state["trainer"] = trainer

    # Re-wire all router dependencies
    tokenization.set_tokenizer(tok, text)
    embeddings.set_deps(model, tok)
    attention.set_deps(model, tok)
    transformer.set_deps(model, tok)
    training.set_deps(trainer, text)
    generation.set_deps(model, tok)

    return {
        "status": "ok",
        "config": config.to_dict(),
        "parameters": param_info,
        "message": f"Model rebuilt: {param_info['total']:,} parameters",
    }


@router.get("/status")
def get_status():
    """Return current model/training status for the dashboard."""
    model = _app_state.get("model")
    trainer = _app_state.get("trainer")
    tok = _app_state.get("tokenizer")

    is_training = False
    if trainer and hasattr(trainer, '_stop_event'):
        is_training = not trainer._stop_event.is_set() if trainer._stop_event else False

    return {
        "model_ready": model is not None,
        "tokenizer_ready": tok is not None,
        "config": model.config.to_dict() if model else None,
        "parameters": model.count_parameters() if model else None,
        "vocab_size": tok.vocab_size if tok else 0,
        "dataset": _app_state.get("dataset_name", "tiny_shakespeare"),
        "training": {
            "is_training": is_training,
            "steps_completed": trainer.step if trainer else 0,
            "current_loss": trainer.loss_history[-1]["loss"] if trainer and trainer.loss_history else None,
            "total_history": len(trainer.loss_history) if trainer else 0,
        },
    }


# ---------------------------------------------------------------------------
# Dataset management
# ---------------------------------------------------------------------------

@router.get("/datasets")
def list_datasets():
    """List available datasets with metadata."""
    datasets = []

    # Tiny Shakespeare (always available)
    from core.tokenizer import get_training_text
    shakespeare_text = get_training_text()
    datasets.append({
        "name": "tiny_shakespeare",
        "display_name": "Tiny Shakespeare",
        "size_bytes": len(shakespeare_text.encode("utf-8")),
        "char_count": len(shakespeare_text),
        "preview": shakespeare_text[:300],
        "active": _app_state.get("dataset_name", "tiny_shakespeare") == "tiny_shakespeare",
    })

    # Preset datasets
    os.makedirs(PRESETS_DIR, exist_ok=True)
    for fname in sorted(os.listdir(PRESETS_DIR)):
        if fname.endswith(".txt"):
            path = os.path.join(PRESETS_DIR, fname)
            name = fname[:-4]
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            datasets.append({
                "name": name,
                "display_name": name.replace("_", " ").title(),
                "size_bytes": len(text.encode("utf-8")),
                "char_count": len(text),
                "preview": text[:300],
                "active": _app_state.get("dataset_name") == name,
            })

    # Custom upload
    if os.path.exists(CUSTOM_DATA_PATH):
        with open(CUSTOM_DATA_PATH, "r", encoding="utf-8") as f:
            text = f.read()
        datasets.append({
            "name": "custom",
            "display_name": "Custom Upload",
            "size_bytes": len(text.encode("utf-8")),
            "char_count": len(text),
            "preview": text[:300],
            "active": _app_state.get("dataset_name") == "custom",
        })

    return {"datasets": datasets}


@router.post("/select-dataset")
def select_dataset(req: SelectDatasetRequest):
    """Switch the active training dataset and re-tokenize."""
    from core.tokenizer import get_training_text
    from routers import tokenization, training

    if req.name == "tiny_shakespeare":
        text = get_training_text()
    elif req.name == "custom":
        if not os.path.exists(CUSTOM_DATA_PATH):
            raise HTTPException(404, "No custom dataset uploaded")
        with open(CUSTOM_DATA_PATH, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        path = os.path.join(PRESETS_DIR, f"{req.name}.txt")
        if not os.path.exists(path):
            raise HTTPException(404, f"Dataset '{req.name}' not found")
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

    if len(text) < 100:
        raise HTTPException(400, "Dataset too small (< 100 chars)")

    _app_state["text"] = text
    _app_state["dataset_name"] = req.name

    # Update routers that need the text
    tok = _app_state.get("tokenizer")
    if tok:
        tokenization.set_tokenizer(tok, text)
    trainer = _app_state.get("trainer")
    if trainer:
        training.set_deps(trainer, text)

    return {
        "status": "ok",
        "dataset": req.name,
        "char_count": len(text),
        "size_bytes": len(text.encode("utf-8")),
    }


@router.post("/upload-data")
async def upload_data(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
):
    """Upload custom training text (file or raw text)."""
    if file:
        content = await file.read()
        data = content.decode("utf-8", errors="replace")
    elif text:
        data = text
    else:
        raise HTTPException(400, "Provide either a file or text")

    if len(data) < 100:
        raise HTTPException(400, "Data too small (need at least 100 characters)")
    if len(data) > 10_000_000:
        raise HTTPException(400, "Data too large (max 10 MB)")

    os.makedirs(os.path.dirname(CUSTOM_DATA_PATH), exist_ok=True)
    with open(CUSTOM_DATA_PATH, "w", encoding="utf-8") as f:
        f.write(data)

    # Auto-select custom dataset
    _app_state["text"] = data
    _app_state["dataset_name"] = "custom"

    from routers import tokenization, training
    tok = _app_state.get("tokenizer")
    if tok:
        tokenization.set_tokenizer(tok, data)
    trainer = _app_state.get("trainer")
    if trainer:
        training.set_deps(trainer, data)

    return {
        "status": "ok",
        "char_count": len(data),
        "size_bytes": len(data.encode("utf-8")),
        "preview": data[:300],
    }


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------

@router.post("/snapshot")
def save_snapshot(req: SnapshotRequest):
    """Save a named snapshot of the current model state."""
    import torch

    model = _app_state.get("model")
    trainer = _app_state.get("trainer")
    tok = _app_state.get("tokenizer")

    if model is None:
        raise HTTPException(503, "Model not initialized")

    os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
    snap_dir = os.path.join(SNAPSHOTS_DIR, req.name)
    os.makedirs(snap_dir, exist_ok=True)

    # Save model weights
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": model.config.to_dict(),
        "step": trainer.step if trainer else 0,
        "loss_history": trainer.loss_history if trainer else [],
    }, os.path.join(snap_dir, "model.pt"))

    # Save metadata
    meta = {
        "name": req.name,
        "description": req.description,
        "created_at": time.time(),
        "config": model.config.to_dict(),
        "parameters": model.count_parameters(),
        "steps": trainer.step if trainer else 0,
        "loss": trainer.loss_history[-1]["loss"] if trainer and trainer.loss_history else None,
        "dataset": _app_state.get("dataset_name", "tiny_shakespeare"),
        "vocab_size": tok.vocab_size if tok else 0,
    }
    with open(os.path.join(snap_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return {"status": "ok", "snapshot": meta}


@router.get("/snapshots")
def list_snapshots():
    """List all saved snapshots."""
    snapshots = []
    if os.path.exists(SNAPSHOTS_DIR):
        for name in sorted(os.listdir(SNAPSHOTS_DIR)):
            meta_path = os.path.join(SNAPSHOTS_DIR, name, "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                snapshots.append(meta)
    return {"snapshots": snapshots}


@router.post("/load-snapshot")
def load_snapshot(req: LoadSnapshotRequest):
    """Load a saved snapshot into the current model."""
    import torch
    from core.model import GPT, GPTConfig
    from core.trainer import Trainer, TrainConfig
    from routers import embeddings, attention, transformer, training, generation, tokenization

    snap_dir = os.path.join(SNAPSHOTS_DIR, req.name)
    if not os.path.exists(snap_dir):
        raise HTTPException(404, f"Snapshot '{req.name}' not found")

    ckpt = torch.load(os.path.join(snap_dir, "model.pt"), map_location="cpu", weights_only=False)
    saved_config = ckpt["config"]

    tok = _app_state.get("tokenizer")
    text = _app_state.get("text")
    if tok is None:
        raise HTTPException(503, "Tokenizer not initialized")

    # Rebuild model with saved config
    config = GPTConfig(**saved_config)
    model = GPT(config)
    model.load_state_dict(ckpt["model_state_dict"])

    checkpoint_dir = _app_state.get("checkpoint_dir", "checkpoints")
    train_config = TrainConfig(checkpoint_dir=checkpoint_dir)
    trainer = Trainer(model, tok, train_config)
    trainer.step = ckpt.get("step", 0)
    trainer.loss_history = ckpt.get("loss_history", [])

    _app_state["model"] = model
    _app_state["trainer"] = trainer

    tokenization.set_tokenizer(tok, text)
    embeddings.set_deps(model, tok)
    attention.set_deps(model, tok)
    transformer.set_deps(model, tok)
    training.set_deps(trainer, text)
    generation.set_deps(model, tok)

    return {
        "status": "ok",
        "config": config.to_dict(),
        "steps": trainer.step,
        "parameters": model.count_parameters(),
    }
