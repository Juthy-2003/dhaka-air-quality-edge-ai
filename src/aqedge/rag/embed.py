"""bge-small-en-v1.5 embeddings (384 dimensions) via fastembed (ONNX, CPU,
no PyTorch). BGE expects queries — not passages — to carry an instruction
prefix; forgetting it measurably hurts retrieval."""

from functools import lru_cache

import numpy as np

MODEL = "BAAI/bge-small-en-v1.5"
DIM = 384
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def _model():
    from fastembed import TextEmbedding

    return TextEmbedding(MODEL)


def embed_passages(texts: list[str]) -> np.ndarray:
    return np.array(list(_model().embed(texts)), dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    return np.array(next(iter(_model().embed([QUERY_PREFIX + text]))), dtype=np.float32)
