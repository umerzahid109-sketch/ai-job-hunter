
import unittest

from src.rag.chunking import chunk_text


class TestChunking(unittest.TestCase):
    def test_empty_text_returns_no_chunks(self):
        self.assertEqual(chunk_text("", "cv-1"), [])

    def test_short_text_returns_one_chunk(self):
        chunks = chunk_text("Python and API integration", "cv-1")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "Python and API integration")

    def test_chunks_overlap(self):
        words = [f"word{i}" for i in range(10)]
        chunks = chunk_text(
            " ".join(words),
            "cv-1",
            chunk_size=4,
            overlap=1,
        )

        self.assertEqual(len(chunks), 3)
        self.assertEqual(
            chunks[0].text.split()[-1],
            chunks[1].text.split()[0],
        )

    def test_source_metadata_is_preserved(self):
        chunks = chunk_text(
            "Python automation project",
            "my-cv",
            source_type="cv",
        )

        self.assertEqual(chunks[0].source_id, "my-cv")
        self.assertEqual(chunks[0].source_type, "cv")
        self.assertEqual(chunks[0].chunk_index, 0)

    def test_invalid_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            chunk_text("some text", "cv-1", chunk_size=5, overlap=5)

    def test_chunk_ids_are_unique(self):
        text = " ".join(f"word{i}" for i in range(20))
        chunks = chunk_text(
            text,
            "cv-1",
            chunk_size=5,
            overlap=1,
        )

        ids = [chunk.chunk_id for chunk in chunks]
        self.assertEqual(len(ids), len(set(ids)))


if __name__ == "__main__":
    unittest.main()
