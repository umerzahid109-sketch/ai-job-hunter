
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.rag.schemas import RetrievalResult, TextChunk


class RetrievalIndex:
    """A simple TF-IDF index for searching text chunks."""

    def __init__(self):
        self._vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=8000,
        )
        self._chunks: list[TextChunk] = []
        self._matrix = None

    def fit(self, chunks: list[TextChunk]) -> None:
        """Build the search index from text chunks."""

        if not chunks:
            raise ValueError("At least one chunk is required")

        if not any(chunk.text.strip() for chunk in chunks):
            raise ValueError("Chunks must contain searchable text")

        try:
            matrix = self._vectorizer.fit_transform(
                [chunk.text for chunk in chunks]
            )
        except ValueError as error:
            raise ValueError(
                "Could not build index. Check that chunks contain "
                "meaningful searchable words."
            ) from error

        self._chunks = list(chunks)
        self._matrix = matrix

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Return the top matching chunks, highest score first."""

        if self._matrix is None:
            raise RuntimeError("Call fit() before search()")

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be empty")

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_vector = self._vectorizer.transform([query])
        scores = cosine_similarity(
            query_vector,
            self._matrix,
        ).ravel()

        ranked_indices = sorted(
            range(len(self._chunks)),
            key=lambda index: (-scores[index], index),
        )

        results = []

        for index in ranked_indices[:top_k]:
            score = float(scores[index])

            # Exclude chunks with no matching terms.
            if score <= 0:
                continue

            results.append(
                RetrievalResult(
                    chunk=self._chunks[index],
                    score=score,
                )
            )

        return results
