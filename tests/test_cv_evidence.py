import unittest
from unittest.mock import MagicMock, patch

from src.rag.cv_evidence import CVEvidenceRetriever
from src.rag.schemas import RetrievalResult, TextChunk


class TestCVEvidenceRetriever(unittest.TestCase):

    @patch("src.rag.cv_evidence.SemanticIndex")
    def test_from_cv_chunks_and_indexes_text(self, mock_index_class):
        mock_index = mock_index_class.return_value

        retriever = CVEvidenceRetriever.from_cv(
            "Python developer with machine learning experience.",
            source_id="test_cv",
        )

        mock_index.fit.assert_called_once()
        chunks = mock_index.fit.call_args.args[0]

        self.assertTrue(chunks)
        self.assertEqual(chunks[0].source_id, "test_cv")
        self.assertEqual(chunks[0].source_type, "cv")
        self.assertIs(retriever.index, mock_index)

    @patch("src.rag.cv_evidence.SemanticIndex")
    def test_retrieve_passes_search_parameters(self, mock_index_class):
        mock_index = mock_index_class.return_value
        expected = [
            RetrievalResult(
                chunk=TextChunk(
                    chunk_id="cv::chunk_0",
                    source_id="cv",
                    source_type="cv",
                    text="Python and machine learning experience.",
                    chunk_index=0,
                ),
                score=0.82,
            )
        ]
        mock_index.search.return_value = expected

        retriever = CVEvidenceRetriever.from_cv("Python experience.")
        results = retriever.retrieve(
            "machine learning",
            top_k=3,
            min_score=0.25,
        )

        self.assertEqual(results, expected)
        mock_index.search.assert_called_once_with(
            query="machine learning",
            top_k=3,
            min_score=0.25,
        )

    @patch("src.rag.cv_evidence.SemanticIndex")
    def test_empty_cv_is_rejected(self, mock_index_class):
        with self.assertRaises(ValueError):
            CVEvidenceRetriever.from_cv("   ")

        mock_index_class.assert_not_called()

    @patch("src.rag.cv_evidence.SemanticIndex")
    def test_non_string_cv_is_rejected(self, mock_index_class):
        with self.assertRaises(TypeError):
            CVEvidenceRetriever.from_cv(None)

        mock_index_class.assert_not_called()


if __name__ == "__main__":
    unittest.main()
