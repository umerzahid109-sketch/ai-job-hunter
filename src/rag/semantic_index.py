from sentence_transformers import SentenceTransformer

from .schemas import RetrievalResult, TextChunk


class SemanticIndex:
    """Retrieve text chunks using semantic similarity."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.model = SentenceTransformer(model_name)
        self.chunks: list[TextChunk] = []
        self.embeddings = None

    def fit(self, chunks: list[TextChunk]) -> None:
        if not chunks:
            raise ValueError("Cannot build an index with no chunks.")

        self.chunks = list(chunks)
        texts = [chunk.text for chunk in self.chunks]

        self.embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    
    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[RetrievalResult]:
        if self.embeddings is None:
            raise RuntimeError("Call fit() before search().")
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if not -1.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between -1 and 1.")

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        scores = self.embeddings @ query_embedding
        ranked_indices = scores.argsort()[::-1]
        results = []

        for index in ranked_indices:
            score = float(scores[index])

            if score < min_score:
                break

            results.append(
                RetrievalResult(
                    chunk=self.chunks[int(index)],
                    score=score,
                )
            )

            if len(results) >= top_k:
                break

        return results
