"""Create normalized embeddings, with a dependency-free local fallback."""

import hashlib
import re

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
FALLBACK_BACKEND = "local-hash-v1"

_model = None
_model_error: Exception | None = None


def _load_model():
    """Load the optional transformer once; remember failures for this process."""
    global _model, _model_error
    if _model is not None:
        return _model
    if _model_error is not None:
        raise RuntimeError("Sentence Transformers is unavailable") from _model_error
    try:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(MODEL_NAME)
        return _model
    except Exception as error:
        _model_error = error
        raise RuntimeError("Sentence Transformers is unavailable") from error


def get_embedding_backend(preferred: str | None = None) -> str:
    """Return the backend that will be used for embeddings."""
    if preferred == FALLBACK_BACKEND:
        return FALLBACK_BACKEND
    if preferred not in (None, "sentence-transformers"):
        raise ValueError(f"Unsupported embedding backend: {preferred}")
    try:
        _load_model()
        return "sentence-transformers"
    except RuntimeError:
        if preferred == "sentence-transformers":
            raise
        return FALLBACK_BACKEND


def _hash_embeddings(texts: list[str]) -> np.ndarray:
    """Build stable token and character n-gram vectors using only NumPy."""
    vectors = np.zeros((len(texts), EMBEDDING_DIMENSION), dtype=np.float32)
    for row, text in enumerate(texts):
        words = re.findall(r"[\w']+", text.casefold(), flags=re.UNICODE)
        features: list[tuple[str, float]] = [(word, 1.0) for word in words]
        features.extend(
            (f"{left}_{right}", 1.3) for left, right in zip(words, words[1:])
        )
        for word in words:
            padded = f"^{word}$"
            features.extend(
                (padded[index : index + size], 0.25)
                for size in (3, 4)
                for index in range(max(0, len(padded) - size + 1))
            )

        for feature, weight in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            index = value % EMBEDDING_DIMENSION
            sign = 1.0 if value & (1 << 63) else -1.0
            vectors[row, index] += sign * weight

        norm = np.linalg.norm(vectors[row])
        if norm:
            vectors[row] /= norm
    return vectors


def create_embeddings(
    texts: list[str], backend: str | None = None
) -> np.ndarray:
    """Create normalized vectors with the selected or best available backend."""
    if not texts:
        return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)

    selected_backend = get_embedding_backend(backend)
    if selected_backend == FALLBACK_BACKEND:
        return _hash_embeddings(texts)

    model = _load_model()
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype=np.float32)


if __name__ == "__main__":
    sentences = [
        "Machine learning helps computers learn from data.",
        "Python is a popular programming language.",
        "Artificial intelligence enables machines to perform intelligent tasks.",
    ]
    vectors = create_embeddings(sentences)
    print("Embedding backend:", get_embedding_backend())
    print("Number of sentences:", len(sentences))
    print("Embedding shape:", vectors.shape)
    print("\nSimilarity matrix:")
    print(np.round(vectors @ vectors.T, 3))
