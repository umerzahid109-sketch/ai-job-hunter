
import unittest

from src.rag.chunking import chunk_text
from src.rag.index import RetrievalIndex


class TestRetrievalIndex(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            *chunk_text(
                "Python programming API integration and automation",
                "cv-python",
                source_type="cv",
                chunk_size=20,
                overlap=0,
            ),
            *chunk_text(
                "React frontend components and JavaScript interfaces",
                "cv-frontend",
                source_type="cv",
                chunk_size=20,
                overlap=0,
            ),
            *chunk_text(
                "SQL databases and relational data queries",
                "cv-sql",
                source_type="cv",
                chunk_size=20,
                overlap=0,
            ),
        ]

        self.index = RetrievalIndex()
        self.index.fit(self.chunks)

    def test_relevant_chunk_ranks_first(self):
        results = self.index.search(
            "Python API integration",
            top_k=3,
        )

        self.assertTrue(results)
        self.assertEqual(results[0].chunk.source_id, "cv-python")

    def test_results_are_ranked_highest_first(self):
        results = self.index.search("Python automation", top_k=3)
        scores = [result.score for result in results]

        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_top_k_limits_result_count(self):
        results = self.index.search("Python SQL React", top_k=2)
        self.assertLessEqual(len(results), 2)

    def test_unrelated_query_returns_no_results(self):
        results = self.index.search("underwater basket weaving")
        self.assertEqual(results, [])

    def test_search_before_fit_fails(self):
        empty_index = RetrievalIndex()

        with self.assertRaises(RuntimeError):
            empty_index.search("Python")

    def test_empty_query_is_rejected(self):
        with self.assertRaises(ValueError):
            self.index.search(" ")

    def test_source_metadata_is_available(self):
        results = self.index.search("Python API integration")

        self.assertEqual(results[0].chunk.source_type, "cv")
        self.assertGreater(results[0].score, 0)


if __name__ == "__main__":
    unittest.main()
