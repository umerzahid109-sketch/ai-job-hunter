
from src.rag.schemas import TextChunk


def chunk_text(
    text: str,
    source_id: str,
    source_type: str = "document",
    chunk_size: int = 100,
    overlap: int = 20,
) -> list[TextChunk]:
    """Split text into overlapping word-based chunks."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if not source_id.strip():
        raise ValueError("source_id must not be empty")

    if not source_type.strip():
        raise ValueError("source_type must not be empty")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size"
        )

    words = text.split()
    if not words:
        return []

    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        selected_words = words[start : start + chunk_size]

        if not selected_words:
            break

        chunk_index = len(chunks)

        chunks.append(
            TextChunk(
                chunk_id=f"{source_id}::chunk_{chunk_index}",
                source_id=source_id,
                source_type=source_type,
                text=" ".join(selected_words),
                chunk_index=chunk_index,
            )
        )

        if start + chunk_size >= len(words):
            break

    return chunks
