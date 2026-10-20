"""Bound retrieved context before sending it to a language model."""


def filter_context(documents: list[dict], max_chunks: int = 5, max_chars: int = 6000) -> list[dict]:
    filtered, total = [], 0
    for document in documents:
        text = str(document.get("text", "")).strip()
        remaining = max_chars - total
        if not text or remaining <= 0:
            break
        item = dict(document)
        item["text"] = text[:remaining]
        filtered.append(item)
        total += len(item["text"])
        if len(filtered) >= max_chunks:
            break
    return filtered
