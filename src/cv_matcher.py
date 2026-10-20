import re
from pathlib import Path

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def extract_cv_text(pdf_path):
    """Extract text from a PDF CV."""
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"CV file not found: {path}")

    reader = PdfReader(str(path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        raise ValueError(
            "No selectable text was found in this PDF. "
            "Try a text-based PDF rather than a scanned image."
        )

    return text


def calculate_cv_match(cv_text, job_description):
    """Calculate text similarity between a CV and a job description."""
    if not cv_text or not job_description:
        return {
            "match_score": 0,
            "matched_terms": [],
            "missing_terms": [],
        }

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=8000,
    )

    try:
        matrix = vectorizer.fit_transform([cv_text, job_description])
        score = float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])
        features = vectorizer.get_feature_names_out()
    except ValueError:
        return {
            "match_score": 0,
            "matched_terms": [],
            "missing_terms": [],
        }

    cv_words = set(re.findall(r"[a-zA-Z][a-zA-Z+#.]*", cv_text.lower()))
    job_words = set(
        re.findall(r"[a-zA-Z][a-zA-Z+#.]*", job_description.lower())
    )

    common_terms = sorted(
        term for term in cv_words.intersection(job_words)
        if len(term) > 2
    )

    missing_terms = sorted(
        term for term in job_words.difference(cv_words)
        if len(term) > 3
    )

    return {
        "match_score": round(score * 100),
        "matched_terms": common_terms[:15],
        "missing_terms": missing_terms[:15],
    }