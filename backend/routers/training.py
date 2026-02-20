"""
Training API router.

Endpoints:
    POST /api/train            — start training (SSE stream)
    POST /api/stop-training    — stop training
    GET  /api/training-status  — current training status
    GET  /api/loss-history     — full loss curve data
    POST /api/gradient-demo    — backpropagation demo
    POST /api/gradient-flow    — gradient magnitudes per layer
"""

import asyncio
import threading
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
import json

router = APIRouter(prefix="/api", tags=["training"])

_trainer = None
_training_text = None
_training_thread = None
_training_events: list[dict] = []
_is_training = False


def set_deps(trainer, training_text):
    global _trainer, _training_text
    _trainer = trainer
    _training_text = training_text


class TrainRequest(BaseModel):
    max_steps: int = 2000
    batch_size: int = 32
    learning_rate: float = 3e-4
    log_interval: int = 50
    sample_interval: int = 500


class GradientRequest(BaseModel):
    text: str = "To be, or not to be"


@router.post("/train")
async def train(req: TrainRequest):
    """Start training and stream progress events via SSE."""
    global _is_training, _training_events, _training_thread

    if _trainer is None or _training_text is None:
        raise HTTPException(503, "Trainer not initialized")

    if _is_training:
        raise HTTPException(409, "Training already in progress")

    # Update config
    _trainer.config.max_steps = req.max_steps
    _trainer.config.batch_size = req.batch_size
    _trainer.config.learning_rate = req.learning_rate
    _trainer.config.log_interval = req.log_interval
    _trainer.config.sample_interval = req.sample_interval

    _training_events = []
    _is_training = True

    async def event_generator():
        global _is_training

        # Run training in a thread to not block the event loop
        loop = asyncio.get_event_loop()
        event_queue = asyncio.Queue()

        def train_worker():
            global _is_training
            try:
                for event in _trainer.train(_training_text):
                    _training_events.append(event)
                    loop.call_soon_threadsafe(event_queue.put_nowait, event)
            except Exception as e:
                loop.call_soon_threadsafe(
                    event_queue.put_nowait,
                    {"type": "error", "message": str(e)}
                )
            finally:
                _is_training = False
                loop.call_soon_threadsafe(event_queue.put_nowait, None)

        _training_thread = threading.Thread(target=train_worker, daemon=True)
        _training_thread.start()

        while True:
            event = await event_queue.get()
            if event is None:
                break
            yield {"event": event["type"], "data": json.dumps(event)}

    return EventSourceResponse(event_generator())


@router.post("/stop-training")
def stop_training():
    """Request graceful stop of training."""
    if _trainer is None:
        raise HTTPException(503, "Trainer not initialized")
    _trainer.stop()
    return {"status": "stop_requested"}


@router.get("/training-status")
def training_status():
    """Return current training status."""
    return {
        "is_training": _is_training,
        "step": _trainer.step if _trainer else 0,
        "total_events": len(_training_events),
        "loss_history_length": len(_trainer.loss_history) if _trainer else 0,
    }


@router.get("/loss-history")
def loss_history():
    """Return the complete loss/LR history."""
    if _trainer is None:
        return {"loss_history": [], "lr_history": [], "samples": []}
    return {
        "loss_history": _trainer.loss_history,
        "lr_history": _trainer.lr_history,
        "samples": _trainer.samples,
    }


@router.post("/gradient-demo")
def gradient_demo():
    """
    Run the manual backpropagation demo:
    tiny attention + linear layer with hand-computed gradients
    vs PyTorch autograd.
    """
    from core.numpy_demos import backprop_demo
    return backprop_demo()


@router.post("/gradient-flow")
def gradient_flow(req: GradientRequest):
    """
    Run a forward+backward pass on the given text and return
    gradient magnitudes per layer.
    """
    import torch

    if _trainer is None or _trainer.model is None:
        raise HTTPException(503, "Model not initialized")

    tok = _trainer.tokenizer
    model = _trainer.model

    token_ids = tok.encode(req.text)[:model.config.context_len - 1]

    # Need at least 2 tokens for input/target
    if len(token_ids) < 2:
        raise HTTPException(400, "Text too short")

    # Create input/target pair
    x = torch.tensor([token_ids[:-1]])
    y = torch.tensor([token_ids[1:]])

    model.train()
    model.zero_grad()

    logits, loss = model(x, y)
    loss.backward()

    flow = _trainer.get_gradient_flow()

    model.zero_grad()
    model.eval()

    return {
        "loss": loss.item(),
        "gradient_flow": flow,
        "explanation": (
            "Each bar shows the average absolute gradient for a parameter group. "
            "Very small gradients → vanishing gradient problem (early layers learn slowly). "
            "Very large gradients → exploding gradients (training instability). "
            "The residual connections and LayerNorm in transformers help keep gradients "
            "in a healthy range across all layers."
        ),
    }
