import json
import unittest
from unittest.mock import patch

from src.ai_application_assistant import verify_generated_text


class TestEvidenceAuditorRegression(unittest.TestCase):
    @patch("src.ai_application_assistant._generate_text")
    def test_unsupported_javascript_gmail_claim_is_flagged(self, mock_generate):
        cv_text = """
        Skills: Python, JavaScript, React

        Personal AI Gmail Assistant
        Built with Google Apps Script and an AI API.
        Classifies important emails and generates a daily digest.
        """
        generated_text = (
            "I built the Personal AI Gmail Assistant using JavaScript."
        )

        mock_generate.return_value = json.dumps({
            "claims": [{
                "claim": generated_text,
                "evidence": "",
                "status": "supported",
                "reason": "Test response.",
            }]
        })

        results = verify_generated_text(cv_text, generated_text)

        self.assertEqual(len(results), 1)
        self.assertEqual(
            results[0]["status"],
            "FLAGGED — DO NOT USE WITHOUT REVIEW",
        )
        self.assertIn(
            "does not establish JavaScript usage",
            results[0]["reason"],
        )
        mock_generate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
