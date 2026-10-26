import unittest
from unittest.mock import patch

import numpy as np

from src.rag.chunking import chunk_text
from src.rag.semantic_index import SemanticIndex


class FakeEmbeddingModel:
    """A predictable fake model for fast, offline tests."""

    def encode(self, texts, **kwargs):
        vectors = []

        for text in texts:
            value = text.lower()

            if "python" in value or "automation" in value:
                vector = [1.0, 0.0, 0.0]
            elif "react" in value or "frontend" in value:
                vector = [0.0, 1.0, 0.0]
            elif "sql" in value or "database" in value:
                vector = [0.0, 0.0, 1.0]
            else:
                vector = [-1.0, 0.0, 0.0]

            vectors.append(vector)

        return np.array(vectors, dtype=float)


class TestSemanticIndex(unittest.TestCase):
    def setUp(self):
        self.model_patcher = patch(
            "src.rag.semantic_index.SentenceTransformer",
            return_value=FakeEmbeddingModel(),
        )
        self.model_patcher.start()
        self.addCleanup(self.model_patcher.stop)

        self.index = SemanticIndex()

        self.chunks = [
            *chunk_text(
                "Python API integration and automation",
                "cv-python",
                source_type="cv",
                chunk_size=1000,
                overlap=0,
            ),
            *chunk_text(
                "React frontend components and interfaces",
                "cv-frontend",
                source_type="cv",
                chunk_size=1000,
                overlap=0,
            ),
            *chunk_text(
                "SQL databases and relational queries",
                "cv-sql",
                source_type="cv",
                chunk_size=1000,
                overlap=0,
            ),
        ]

        self.index.fit(self.chunks)

    def test_empty_chunk_list_is_rejected(self):
        with self.assertRaises(ValueError):
            self.index.fit([])

    def test_search_before_fit_fails(self):
        empty_index = SemanticIndex()
        empty_index.embeddings = None

        with self.assertRaises(RuntimeError):
            empty_index.search("Python")

    def test_empty_query_is_rejected(self):
        with self.assertRaises(ValueError):
            self.index.search(" ")

    def test_top_k_must_be_positive(self):
        with self.assertRaises(ValueError):
            self.index.search("Python", top_k=0)

    def test_min_score_must_be_between_minus_one_and_one(self):
        with self.assertRaises(ValueError):
            self.index.search("Python", min_score=1.1)

        with self.assertRaises(ValueError):
            self.index.search("Python", min_score=-1.1)

    def test_most_similar_chunk_ranks_first(self):
        results = self.index.search("Python", top_k=3)

        self.assertEqual(results[0].chunk.source_id, "cv-python")
        self.assertAlmostEqual(results[0].score, 1.0)

    def test_min_score_filters_less_similar_chunks(self):
        results = self.index.search(
            "Python",
            top_k=3,
            min_score=0.5,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk.source_id, "cv-python")

    def test_top_k_limits_result_count(self):
        results = self.index.search("Python", top_k=2)

        self.assertLessEqual(len(results), 2)


if __name__ == "__main__":
    unittest.main()
