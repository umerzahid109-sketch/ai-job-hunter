


import re
from datetime import datetime, timezone
from html.parser import HTMLParser

import requests


JOBICY_URL = "https://jobicy.com/api/v2/remote-jobs"
REMOTIVE_URL = "https://remotive.com/api/remote-jobs"

HEADERS = {
    "User-Agent": "AIJobHunter/0.1 (personal job discovery tool)",
    "Accept": "application/json",
}
TIMEOUT = 25


class HTMLTextExtractor(HTMLParser):
    """Convert HTML job descriptions to readable text."""

    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        text = data.strip()
        if text:
            self.parts.append(text)

    def get_text(self):
        return " ".join(self.parts)


def clean_description(value):
    if not value:
        return ""

    parser = HTMLTextExtractor()
    try:
        parser.feed(value)
        return re.sub(r"\s+", " ", parser.get_text()).strip()
    except Exception:
        return re.sub(r"<[^>]*>", " ", value).strip()


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def fetch_jobicy():
    """Fetch recent public remote listings from Jobicy."""
    response = requests.get(
        JOBICY_URL,
        params={"count": 100, "geo": "anywhere"},
        headers=HEADERS,
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()

    jobs = []

    for item in payload.get("jobs", []):
        url = item.get("url")
        title = item.get("jobTitle")
        company = item.get("companyName")

        if not url or not title or not company:
            continue

        description = clean_description(
            item.get("jobDescription") or item.get("jobExcerpt", "")
        )

        jobs.append({
            "title": title.strip(),
            "company": company.strip(),
            "location": item.get("jobGeo") or "Remote; eligibility unspecified",
            "description": description,
            "url": url,
            "source": "Jobicy",
            "posted_at": item.get("pubDate"),
            "discovered_at": now_utc(),
        })

    return jobs


def fetch_remotive():
    """Fetch public remote listings from Remotive."""
    response = requests.get(
        REMOTIVE_URL,
        params={"limit": 100},
        headers=HEADERS,
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()

    jobs = []

    for item in payload.get("jobs", []):
        url = item.get("url")
        title = item.get("title")
        company = item.get("company_name")

        if not url or not title or not company:
            continue

        description = clean_description(item.get("description", ""))

        jobs.append({
            "title": title.strip(),
            "company": company.strip(),
            "location": (
                item.get("candidate_required_location")
                or "Remote; eligibility unspecified"
            ),
            "description": description,
            "url": url,
            "source": "Remotive",
            "posted_at": item.get("publication_date"),
            "discovered_at": now_utc(),
        })

    return jobs


def get_live_jobs():
    """
    Query both sources independently.
    A failure at one source does not stop the other.
    """
    all_jobs = []
    errors = []

    for source_name, fetcher in [
        ("Jobicy", fetch_jobicy),
        ("Remotive", fetch_remotive),
    ]:
        try:
            results = fetcher()
            all_jobs.extend(results)
            print(f"{source_name}: retrieved {len(results)} listings")
        except (requests.RequestException, ValueError, KeyError) as exc:
            errors.append(f"{source_name}: {type(exc).__name__}: {exc}")
            print(f"{source_name}: source unavailable ({type(exc).__name__})")

    return all_jobs, errors
