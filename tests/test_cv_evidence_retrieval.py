import unittest
from unittest.mock import patch

from src.rag.cv_evidence import CVEvidenceRetriever


SAMPLE_CV = """
Skills: Python, JavaScript, React, SQL.

Machine Learning Project:
Built a machine learning model in Python to predict late deliveries.
Used pandas for data preparation and scikit-learn for model training.

Frontend Project:
Developed a responsive user interface using React, JavaScript, HTML, and CSS.
"""


class TestRealCVEvidenceRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.rag.semantic_index import SemanticIndex

        cls.retriever = CVEvidenceRetriever.from_cv(
            SAMPLE_CV,
            source_id="test_cv",
            chunk_size=20,
            overlap=5,
        )

    def test_machine_learning_query_retrieves_relevant_evidence(self):
        results = self.retriever.retrieve(
            "What machine learning project did the candidate build?",
            top_k=2,
            min_score=0.35,
        )

        self.assertTrue(results)
        combined_text = " ".join(r.chunk.text.lower() for r in results)
        self.assertTrue(
            "machine learning" in combined_text
            or "late deliveries" in combined_text
        )

    def test_frontend_query_retrieves_frontend_evidence(self):
        results = self.retriever.retrieve(
            "Which frontend technologies has the candidate used?",
            top_k=2,
            min_score=0.35,
        )

        self.assertTrue(results)
        combined_text = " ".join(r.chunk.text.lower() for r in results)
        self.assertTrue(
            "react" in combined_text or "responsive user interface" in combined_text
        )

    def test_unrelated_cloud_query_returns_no_results(self):
        results = self.retriever.retrieve(
            "Does the CV establish experience with cloud infrastructure?",
            top_k=2,
            min_score=0.35,
        )

        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
