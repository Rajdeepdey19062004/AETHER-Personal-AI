"""Optional cross-encoder reranking, loaded only when explicitly requested."""


class LocalReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str | None = None):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as error:
            raise RuntimeError("Install sentence-transformers to use the optional reranker.") from error
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, documents: list[dict], top_k: int = 5) -> list[dict]:
        if not documents:
            return []
        scores = self.model.predict([(query, item.get("text", "")) for item in documents])
        ranked = [{**item, "rerank_score": float(score)} for item, score in zip(documents, scores)]
        return sorted(ranked, key=lambda item: item["rerank_score"], reverse=True)[:top_k]
