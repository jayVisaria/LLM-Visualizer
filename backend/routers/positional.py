"""
Positional Encoding API router.

Endpoints:
    GET  /api/positional-encoding   — sinusoidal PE matrix
    GET  /api/position-similarity   — cosine similarity between positions
"""

from fastapi import APIRouter

router = APIRouter(prefix="/api", tags=["positional"])


@router.get("/positional-encoding")
def positional_encoding(seq_len: int = 64, d_model: int = 64, encoding_type: str = "sinusoidal"):
    """Return the positional encoding matrix and related data."""
    from core.numpy_demos import sinusoidal_positional_encoding

    result = sinusoidal_positional_encoding(seq_len=seq_len, d_model=d_model)
    result["encoding_type"] = encoding_type
    result["formula"] = {
        "even": "PE(pos, 2i) = sin(pos / 10000^(2i/d_model))",
        "odd": "PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))",
    }
    return result


@router.get("/position-similarity")
def position_similarity(seq_len: int = 32, d_model: int = 64):
    """Return cosine similarity matrix between positional encodings."""
    from core.numpy_demos import sinusoidal_positional_encoding

    result = sinusoidal_positional_encoding(seq_len=seq_len, d_model=d_model)
    return {
        "similarity_matrix": result["similarity_matrix"],
        "seq_len": seq_len,
        "explanation": (
            "Each cell (i, j) shows the cosine similarity between the positional "
            "encoding of position i and position j. Nearby positions have higher "
            "similarity — this is how the model understands token order."
        ),
    }
