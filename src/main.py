

from .database import (
    initialize_database,
    add_job,
    get_all_jobs,
    remove_demo_jobs,
    update_job_match,
)
from .job_sources import get_live_jobs
from .job_matcher import match_job


def main():
    print("\n=== AI JOB HUNTER ===")
    print("Mode: LIVE PUBLIC REMOTE JOB SOURCES\n")

    initialize_database()

    removed = remove_demo_jobs()
    if removed:
        print(f"Removed {removed} fictional demo listings.")

    incoming_jobs, errors = get_live_jobs()

    new_jobs = 0
    updated_jobs = 0

    for job in incoming_jobs:
        match = match_job(job)

        job["category"] = match["category"]
        job["score"] = match["score"]

        # Insert new listings; the database prevents duplicate URLs.
        if add_job(job):
            new_jobs += 1
        else:
            # Refresh the category and score of an existing listing.
            updated = update_job_match(
                job["url"],
                job["category"],
                job["score"],
            )
            if updated:
                updated_jobs += 1

    jobs = [
        job for job in get_all_jobs()
        if job["source"] != "DEMO"
    ]

    print("\n--- RUN SUMMARY ---")
    print(f"Listings retrieved: {len(incoming_jobs)}")
    print(f"New listings saved: {new_jobs}")
    print(f"Existing listings updated: {updated_jobs}")
    print(f"Total unique listings stored: {len(jobs)}")
    print(f"Sources with errors: {len(errors)}")

    if errors:
        print("\n--- SOURCE ERRORS ---")
        for error in errors:
            print(error)

    print("\n--- TOP MATCHES ---\n")

    for job in jobs[:30]:
        print(f"Job: {job['title']}")
        print(f"Company: {job['company']}")
        print(f"Location: {job['location']}")
        print(f"Category: {job['category']}")
        print(f"Relevance: {job['score']}/100")
        print(f"Posted: {job['posted_at'] or 'Not provided'}")
        print(f"Discovered: {job['discovered_at']}")
        print(f"Source: {job['source']}")
        print(f"Apply: {job['url']}")
        print("-" * 55)

    if not incoming_jobs and errors:
        print(
            "\nNo source returned listings successfully. "
            "Check your internet connection and source errors."
        )

    print("\nRun complete.")


if __name__ == "__main__":
    main()
