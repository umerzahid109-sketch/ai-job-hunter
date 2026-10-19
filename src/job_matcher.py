

import re

# Career priorities based on the roles we want to target.
CAREER_KEYWORDS = {
    "Frontend / Web": [
        "frontend developer",
        "front-end developer",
        "frontend engineer",
        "react developer",
        "web developer",
        "javascript developer",
        "ui developer",
        "web development intern",
    ],
    "AI / ML / Automation": [
        "ai engineer",
        "artificial intelligence",
        "machine learning",
        "ml engineer",
        "ai automation",
        "automation engineer",
        "llm engineer",
        "generative ai",
        "data science intern",
        "ai intern",
    ],
    "Cybersecurity": [
        "cybersecurity",
        "security analyst",
        "soc analyst",
        "information security",
        "security engineer",
        "penetration tester",
        "security intern",
    ],
    "SQA / Testing": [
        "software quality assurance",
        "quality assurance engineer",
        "qa engineer",
        "sqa engineer",
        "software tester",
        "test automation",
        "manual tester",
        "qa tester",
        "testing intern",
    ],
    "Software Engineering": [
        "software engineer",
        "graduate software engineer",
        "junior software engineer",
        "associate software engineer",
        "software engineering intern",
        "graduate developer",
        "junior developer",
    ],
}

# These are strong indicators of a role suitable for an early-career applicant.
ENTRY_LEVEL_KEYWORDS = [
    "intern",
    "internship",
    "entry level",
    "entry-level",
    "junior",
    "graduate",
    "new grad",
    "associate",
    "trainee",
    "apprentice",
]

# Roles we generally want to avoid.
SENIOR_KEYWORDS = [
    "senior",
    "staff engineer",
    "principal engineer",
    "lead engineer",
    "tech lead",
    "technical lead",
    "engineering manager",
    "director",
    "head of engineering",
    "chief technology",
    "architect",
]

# Additional role preferences.
EXTRA_PREFERENCES = [
    "remote",
    "work from home",
    "worldwide",
    "anywhere",
]


def contains_phrase(text, phrase):
    """Match a phrase without accidentally matching inside another word."""
    pattern = r"(?<!\w)" + re.escape(phrase.lower()) + r"(?!\w)"
    return re.search(pattern, text.lower()) is not None


def match_job(job):
    title = job.get("title", "")
    description = job.get("description", "")
    location = job.get("location", "")

    title_lower = title.lower()
    description_lower = description.lower()
    location_lower = location.lower()

    # Titles matter more than long descriptions.
    category_scores = {}

    for category, keywords in CAREER_KEYWORDS.items():
        score = 0

        for keyword in keywords:
            if contains_phrase(title_lower, keyword):
                score += 8
            elif contains_phrase(description_lower, keyword):
                score += 2

        category_scores[category] = score

    category = max(category_scores, key=category_scores.get)
    best_category_score = category_scores[category]

    matched_categories = [
        name
        for name, score in category_scores.items()
        if score > 0
    ]

    # Start with relevance to one of the target career fields.
    score = min(60, best_category_score * 5)

    # Reward clear early-career indicators.
    entry_matches = [
        keyword
        for keyword in ENTRY_LEVEL_KEYWORDS
        if contains_phrase(title_lower, keyword)
    ]

    if entry_matches:
        score += 25

    # Penalize clearly senior or leadership positions.
    senior_matches = [
        keyword
        for keyword in SENIOR_KEYWORDS
        if contains_phrase(title_lower, keyword)
    ]

    if senior_matches:
        score -= 50

    # Small preference for remote-friendly roles.
    if any(
        contains_phrase(location_lower, keyword)
        for keyword in ["remote", "worldwide", "anywhere"]
    ):
        score += 10

    # Flag geographic eligibility for manual review.
    # This is only a warning, not proof of eligibility or ineligibility.
    location_restrictions = [
        "united states only",
        "us only",
        "usa only",
        "canada only",
        "uk only",
        "united kingdom only",
        "must be based in the us",
        "must reside in the us",
    ]

    eligibility_warning = any(
        contains_phrase(location_lower, phrase)
        for phrase in location_restrictions
    )

    if eligibility_warning:
        score -= 15

    score = max(0, min(100, score))

    if best_category_score == 0:
        category = "Other"
        score = 0

    return {
        "category": category,
        "score": score,
        "seniority_warning": bool(senior_matches),
        "entry_level_match": bool(entry_matches),
        "eligibility_warning": eligibility_warning,
        "matched_categories": matched_categories,
    }
