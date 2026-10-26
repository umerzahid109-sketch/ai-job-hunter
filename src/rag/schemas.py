
from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    source_id: str
    source_type: str
    text: str
    chunk_index: int


@dataclass(frozen=True)
class RetrievalResult:
    chunk: TextChunk
    score: float
