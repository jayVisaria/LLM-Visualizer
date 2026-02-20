"""
Embeddings API router.

Endpoints:
    POST /api/embeddings       — get embedding vectors for tokens
    POST /api/embed-pca        — get 2D PCA projection of embeddings
    GET  /api/embedding-matrix — get embedding matrix heatmap data
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np

router = APIRouter(prefix="/api", tags=["embeddings"])

_model = None
_tokenizer = None


def set_deps(model, tokenizer):
    global _model, _tokenizer
    _model = model
    _tokenizer = tokenizer


class EmbedRequest(BaseModel):
    text: str


@router.post("/embeddings")
def get_embeddings(req: EmbedRequest):
    """Return embedding vectors for each token in the input text."""
    from core.numpy_demos import embedding_lookup_demo

    tok = _tokenizer
    if tok is None:
        raise HTTPException(503, "Not initialized")

    token_ids = tok.encode(req.text)
    segments = tok.encode_with_segments(req.text)
    token_labels = [s["text"] for s in segments]

    # NumPy demo with the actual vocab size
    demo = embedding_lookup_demo(
        token_ids, vocab_size=tok.vocab_size, d_model=64
    )

    # If model is loaded, get real learned embeddings
    learned_embeddings = None
    if _model is not None:
        import torch
        with torch.no_grad():
            ids_tensor = torch.tensor([token_ids])
            emb = _model.token_embedding(ids_tensor)[0]  # (seq, d_model)
            learned_embeddings = emb.numpy().tolist()

    return {
        "token_ids": token_ids,
        "token_labels": token_labels,
        "numpy_demo": demo,
        "learned_embeddings": learned_embeddings,
    }


@router.post("/embed-pca")
def embed_pca(req: EmbedRequest):
    """Return 2D PCA projection of token embeddings for scatter plot."""
    from sklearn.decomposition import PCA

    tok = _tokenizer
    if tok is None:
        raise HTTPException(503, "Not initialized")

    token_ids = tok.encode(req.text)
    segments = tok.encode_with_segments(req.text)
    token_labels = [s["text"] for s in segments]

    if _model is not None:
        import torch
        with torch.no_grad():
            ids_tensor = torch.tensor([token_ids])
            emb = _model.token_embedding(ids_tensor)[0].numpy()
    else:
        rng = np.random.RandomState(42)
        emb = rng.randn(len(token_ids), 64).astype(np.float32) * 0.02

    # PCA to 2D
    if emb.shape[0] >= 2:
        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(emb)
        explained_var = pca.explained_variance_ratio_.tolist()
    else:
        coords_2d = emb[:, :2]
        explained_var = [1.0, 0.0]

    return {
        "points": [
            {
                "x": float(coords_2d[i, 0]),
                "y": float(coords_2d[i, 1]),
                "label": token_labels[i],
                "id": token_ids[i],
            }
            for i in range(len(token_ids))
        ],
        "explained_variance": explained_var,
        "num_tokens": len(token_ids),
    }


@router.get("/embedding-matrix")
def embedding_matrix(rows: int = 50):
    """Return a slice of the embedding matrix for heatmap visualization."""
    if _model is not None:
        import torch
        with torch.no_grad():
            mat = _model.token_embedding.weight[:rows].numpy()
    else:
        rng = np.random.RandomState(42)
        mat = rng.randn(rows, 64).astype(np.float32) * 0.02

    # Token labels for the rows
    labels = []
    if _tokenizer is not None:
        for i in range(min(rows, _tokenizer.vocab_size)):
            labels.append(
                _tokenizer.vocab[i].decode("utf-8", errors="replace")
            )
    else:
        labels = [str(i) for i in range(rows)]

    return {
        "matrix": mat.tolist(),
        "shape": list(mat.shape),
        "row_labels": labels,
    }
