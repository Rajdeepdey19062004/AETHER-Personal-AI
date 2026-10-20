"""Check AETHER index structure and local Ollama availability."""

import json
import os

import requests

from document_store import load_index
from logger_config import get_logger

logger = get_logger("health_check")


def check_index_health():
    try:
        chunks, embeddings, documents = load_index()
        if embeddings.ndim != 2:
            return {"status": "error", "message": "Embeddings are not a 2D array."}
        if len(chunks) != embeddings.shape[0]:
            return {"status": "error", "message": "Chunk count does not match embedding count."}
        if not isinstance(documents, list):
            return {"status": "error", "message": "Document metadata is invalid."}
        return {"status": "ok", "chunks": len(chunks),
                "embeddings": int(embeddings.shape[0]), "documents": len(documents)}
    except Exception:
        logger.exception("Index health check failed.")
        return {"status": "error", "message": "Unable to load the index."}


def check_ollama_health(base_url=None):
    base_url = (base_url or os.getenv("AETHER_OLLAMA_URL", "http://localhost:11434")).rstrip("/")
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        response.raise_for_status()
        data = response.json()
        return {"status": "ok", "models": [item.get("name", "") for item in data.get("models", [])]}
    except (requests.RequestException, ValueError):
        logger.exception("Ollama health check failed.")
        return {"status": "error", "message": "Ollama is not reachable. Check whether the local service is running."}


def run_health_check():
    report = {"index": check_index_health(), "ollama": check_ollama_health()}
    report["overall_status"] = "ok" if all(item["status"] == "ok" for item in report.values()) else "degraded"
    return report


if __name__ == "__main__":
    print(json.dumps(run_health_check(), indent=2))
