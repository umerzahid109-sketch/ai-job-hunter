

from src.ai_application_assistant import (
    tailor_cv,
    generate_cover_letter,
    verify_generated_text,
)

sample_cv = """
Software Engineering graduate.
Skills: Python, JavaScript, React, SQL.
Project: Personal AI Gmail Assistant built with Google Apps Script
and an AI API to classify important emails and generate a daily digest.
"""

sample_job = """
AI Automation Intern.
Requirements: Python, API integration, automation,
problem-solving, and clear communication.
"""


def print_audit(title, cv_text, generated_text):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    claims = verify_generated_text(cv_text, generated_text)

    if not claims:
        print("No candidate-specific claims were returned for auditing.")
        return

    for number, item in enumerate(claims, start=1):
        print(f"\nClaim {number}: {item['claim']}")
        print(f"Status: {item['status']}")
        print(
            "Exact quotation exists in CV:",
            item["evidence_exists_verbatim"],
        )
        print(f"Evidence: {item['evidence'] or '(none provided)'}")
        print(f"Reason: {item['reason'] or '(not provided)'}")


try:
    print("\nTesting CV tailoring...\n")
    cv_suggestions = tailor_cv(sample_cv, sample_job)
    print(cv_suggestions)

    print_audit(
        "CV SUGGESTIONS EVIDENCE AUDIT",
        sample_cv,
        cv_suggestions,
    )

    print("\n" + "=" * 60)
    print("Testing cover letter generation...")
    print("=" * 60 + "\n")

    cover_letter = generate_cover_letter(sample_cv, sample_job)
    print(cover_letter)

    print_audit(
        "COVER LETTER EVIDENCE AUDIT",
        sample_cv,
        cover_letter,
    )

except (RuntimeError, ValueError) as error:
    print("TEST FAILED:", error)
