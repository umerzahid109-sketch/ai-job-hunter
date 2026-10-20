from urllib.parse import urlencode


ROLE_SEARCHES = {
    "AI / ML / Automation": (
        '"AI automation" OR "machine learning" OR "AI intern"'
    ),
    "Frontend / Web": (
        '"frontend developer" OR "web developer" OR "React developer"'
    ),
    "Cybersecurity": (
        '"cybersecurity intern" OR "security analyst" OR "SOC analyst"'
    ),
    "SQA / Testing": (
        '"QA engineer" OR "software tester" OR "SQA intern"'
    ),
    "Graduate Software Engineering": (
        '"junior software engineer" OR "graduate software engineer"'
    ),
}


def build_linkedin_search_url(category, location="Karachi, Pakistan"):
    """Build a LinkedIn Jobs search URL; does not scrape LinkedIn."""
    keywords = ROLE_SEARCHES.get(category)

    if not keywords:
        raise ValueError(f"Unknown search category: {category}")

    params = {
        "keywords": keywords,
        "location": location,
    }

    return (
        "https://www.linkedin.com/jobs/search/?"
        + urlencode(params)
    )


def get_search_categories():
    return list(ROLE_SEARCHES.keys())