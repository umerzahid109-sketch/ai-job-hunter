from dataclasses import dataclass

from src.rag.chunking import chunk_text
from src.rag.schemas import RetrievalResult
from src.rag.semantic_index import SemanticIndex


@dataclass(frozen=True)
class CVEvidenceRetriever:
    """Retrieve relevant evidence passages from a candidate CV."""

    index: SemanticIndex

    @classmethod
    def from_cv(
        cls,
        cv_text: str,
        source_id: str = "candidate_cv",
        chunk_size: int = 100,
        overlap: int = 20,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> "CVEvidenceRetriever":
        if not isinstance(cv_text, str):
            raise TypeError("cv_text must be a string")

        if not cv_text.strip():
            raise ValueError("cv_text must not be empty")

        chunks = chunk_text(
            text=cv_text,
            source_id=source_id,
            source_type="cv",
            chunk_size=chunk_size,
            overlap=overlap,
        )

        index = SemanticIndex(model_name=model_name)
        index.fit(chunks)

        return cls(index=index)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.20,
    ) -> list[RetrievalResult]:
        """Return the most relevant CV passages for a query."""
        return self.index.search(
            query=query,
            top_k=top_k,
            min_score=min_score,
        )
