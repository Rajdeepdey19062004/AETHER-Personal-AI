"""Run a small labelled retrieval benchmark and save its JSON report."""

import json
from pathlib import Path

from document_store import list_documents
from evaluation import evaluate_aether_queries
from semantic_search import search_documents

BASE_DIR = Path(__file__).resolve().parent
REPORT_DIR = BASE_DIR / "data" / "logs"


def build_test_cases():
    """Use only labels that exist in the current library, avoiding fake IDs."""
    cases = []
    for document in list_documents():
        document_id = document.get("document_id", "")
        filename = document.get("filename", document_id).casefold()
        if "machine" in filename or "learning" in filename:
            query = "What is machine learning?"
        elif "python" in filename:
            query = "What is Python?"
        elif "cyber" in filename or "security" in filename:
            query = "Explain cybersecurity."
        else:
            query = f"Summarize {Path(filename).stem.replace('_', ' ')}."
        cases.append({"query": query, "relevant_ids": [document_id]})
    return cases


def main():
    cases = build_test_cases()
    report = evaluate_aether_queries(cases, search_documents, k=5)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
