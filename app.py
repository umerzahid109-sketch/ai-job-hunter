
import pandas as pd
import streamlit as st
from datetime import date

from src.database import (
    get_all_jobs,
    initialize_database,
    update_job_status,
    update_job_tracking,
    get_due_follow_ups,
    add_job,
    update_job_match,
)
from src.job_sources import get_live_jobs
from src.job_matcher import match_job

st.set_page_config(
    page_title="AI Job Hunter",
    page_icon="🎯",
    layout="wide",
)

STATUS_OPTIONS = [
    "New",
    "To Apply",
    "Applied",
    "Interview",
    "Rejected",
    "Offer",
    "Skipped",
]

CAREER_CATEGORIES = [
    "Frontend / Web",
    "AI / ML / Automation",
    "Cybersecurity",
    "SQA / Testing",
    "Software Engineering",
    "Other",
]


@st.cache_data(ttl=30)
def load_jobs():
    """Load saved job listings from SQLite."""
    initialize_database()
    rows = get_all_jobs()

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame([dict(row) for row in rows])


def refresh_jobs():
    load_jobs.clear()


def parse_saved_date(value):
    """Convert a saved date to a date object for Streamlit."""
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None

    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


st.title("🎯 AI Job Hunter")
st.caption(
    "Discover remote jobs, track applications, and manage follow-ups "
    "from one workspace."
)

# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.header("Job Filters")

    search_text = st.text_input(
        "Search title, company or description",
        placeholder="e.g. React, Python, cybersecurity",
    )

    selected_categories = st.multiselect(
        "Career categories",
        options=CAREER_CATEGORIES,
        default=[
            "Frontend / Web",
            "AI / ML / Automation",
            "Cybersecurity",
            "SQA / Testing",
            "Software Engineering",
        ],
    )

    minimum_score = st.slider(
        "Minimum relevance score",
        min_value=0,
        max_value=100,
        value=20,
        step=5,
    )

    selected_statuses = st.multiselect(
        "Application status",
        options=STATUS_OPTIONS,
        default=["New", "To Apply", "Applied", "Interview", "Offer"],
    )

    st.divider()
    st.subheader("Job discovery")

    if st.button(
        "🔄 Fetch latest jobs",
        use_container_width=True,
    ):
        with st.spinner("Fetching jobs from public sources..."):
            incoming_jobs, errors = get_live_jobs()

            added = 0
            refreshed = 0

            for job in incoming_jobs:
                match = match_job(job)
                job["category"] = match["category"]
                job["score"] = match["score"]

                if add_job(job):
                    added += 1
                else:
                    if update_job_match(
                        job["url"],
                        job["category"],
                        job["score"],
                    ):
                        refreshed += 1

            refresh_jobs()

        st.session_state["last_fetch_summary"] = {
            "retrieved": len(incoming_jobs),
            "added": added,
            "refreshed": refreshed,
            "errors": errors,
        }
        st.rerun()

    if st.button(
        "↻ Refresh dashboard",
        use_container_width=True,
    ):
        refresh_jobs()
        st.rerun()


# ---------------- REFRESH RESULTS ----------------

summary = st.session_state.get("last_fetch_summary")

if summary:
    st.subheader("Latest refresh")
    st.write(
        f"Retrieved: {summary['retrieved']} · "
        f"New jobs: {summary['added']} · "
        f"Existing jobs refreshed: {summary['refreshed']}"
    )

    for error in summary["errors"]:
        st.warning(str(error))


# ---------------- LOAD DATABASE ----------------

jobs_df = load_jobs()

if jobs_df.empty:
    st.info(
        "Your database is empty. Use 'Fetch latest jobs' or run "
        "`python -m src.main` in your terminal."
    )
    st.stop()

if "source" in jobs_df.columns:
    jobs_df = jobs_df[jobs_df["source"] != "DEMO"].copy()

# Add missing columns for compatibility with an older database.
defaults = [
    ("id", 0),
    ("title", ""),
    ("company", ""),
    ("description", ""),
    ("location", "Not specified"),
    ("category", "Other"),
    ("score", 0),
    ("status", "New"),
    ("url", ""),
    ("source", "Unknown"),
    ("posted_at", None),
    ("discovered_at", ""),
    ("application_date", None),
    ("follow_up_date", None),
    ("notes", ""),
]

for column, default in defaults:
    if column not in jobs_df.columns:
        jobs_df[column] = default

jobs_df["score"] = pd.to_numeric(
    jobs_df["score"],
    errors="coerce",
).fillna(0)

jobs_df["status"] = jobs_df["status"].fillna("New")
jobs_df["notes"] = jobs_df["notes"].fillna("")


# ---------------- OVERVIEW ----------------

total_jobs = len(jobs_df)
new_count = int((jobs_df["status"] == "New").sum())
applied_count = int((jobs_df["status"] == "Applied").sum())
interview_count = int((jobs_df["status"] == "Interview").sum())
offer_count = int((jobs_df["status"] == "Offer").sum())

metric_cols = st.columns(5)
metric_cols[0].metric("Total jobs", total_jobs)
metric_cols[1].metric("New", new_count)
metric_cols[2].metric("Applied", applied_count)
metric_cols[3].metric("Interviews", interview_count)
metric_cols[4].metric("Offers", offer_count)


# ---------------- FOLLOW-UP REMINDERS ----------------

st.divider()
st.subheader("📅 Follow-up reminders")

due_follow_ups = get_due_follow_ups(date.today().isoformat())

if due_follow_ups:
    for item in due_follow_ups:
        st.warning(
            f"Follow up with {item['company']} — {item['title']} "
            f"(due {item['follow_up_date']})"
        )
else:
    st.success("No overdue follow-ups. You're all caught up!")


# ---------------- FILTER JOBS ----------------

st.divider()
st.subheader("Find your next opportunity")

filtered = jobs_df.copy()

if search_text.strip():
    query = search_text.strip().lower()

    searchable = (
        filtered["title"].fillna("").astype(str)
        + " "
        + filtered["company"].fillna("").astype(str)
        + " "
        + filtered["description"].fillna("").astype(str)
    ).str.lower()

    filtered = filtered[searchable.str.contains(query, regex=False)]

if selected_categories:
    filtered = filtered[
        filtered["category"].isin(selected_categories)
    ]
else:
    filtered = filtered.iloc[0:0]

filtered = filtered[filtered["score"] >= minimum_score]

if selected_statuses:
    filtered = filtered[
        filtered["status"].isin(selected_statuses)
    ]
else:
    filtered = filtered.iloc[0:0]

sort_option = st.selectbox(
    "Sort jobs by",
    [
        "Highest relevance",
        "Most recently discovered",
        "Company name",
        "Job title",
    ],
)

if sort_option == "Highest relevance":
    filtered = filtered.sort_values(
        ["score", "discovered_at"],
        ascending=[False, False],
    )
elif sort_option == "Most recently discovered":
    filtered = filtered.sort_values(
        "discovered_at",
        ascending=False,
    )
elif sort_option == "Company name":
    filtered = filtered.sort_values("company")
else:
    filtered = filtered.sort_values("title")

st.subheader(f"Matching jobs ({len(filtered)})")


# ---------------- JOB CARDS ----------------

if filtered.empty:
    st.warning(
        "No jobs match these filters. Lower the relevance score "
        "or select more categories and statuses."
    )
else:
    for _, job in filtered.iterrows():
        title = str(job["title"] or "Untitled role")
        company = str(job["company"] or "Unknown company")
        category = str(job["category"] or "Other")
        location = str(job["location"] or "Not specified")
        url = str(job["url"] or "")
        status = str(job["status"] or "New")
        score = int(job["score"])
        job_id = int(job["id"])

        with st.container(border=True):
            heading_col, score_col = st.columns([5, 1])

            heading_col.markdown(f"### {title}")
            heading_col.write(f"**{company}** · {location}")
            score_col.metric("Match", f"{score}/100")

            st.caption(
                f"Category: {category} · "
                f"Source: {job['source']} · "
                f"Status: {status}"
            )

            if job["posted_at"]:
                st.caption(f"Posted: {job['posted_at']}")

            button_col, status_col = st.columns([1, 2])

            if url.startswith(("https://", "http://")):
                button_col.link_button(
                    "Open job posting",
                    url,
                    use_container_width=True,
                )
            else:
                button_col.caption("No valid application URL")

            new_status = status_col.selectbox(
                "Application status",
                options=STATUS_OPTIONS,
                index=(
                    STATUS_OPTIONS.index(status)
                    if status in STATUS_OPTIONS
                    else 0
                ),
                key=f"status_{job_id}",
            )

            if new_status != status:
                if update_job_status(job_id, new_status):
                    refresh_jobs()
                    st.toast(f"Status updated: {new_status}")
                    st.rerun()

            with st.expander("View job description"):
                description = str(job["description"] or "")

                if description.strip():
                    st.write(description[:5000])

                    if len(description) > 5000:
                        st.caption(
                            "Description shortened for display."
                        )
                else:
                    st.caption(
                        "No description was provided by the source."
                    )

            with st.expander("📌 Application tracking"):
                application_date = st.date_input(
                    "Application date",
                    value=parse_saved_date(
                        job.get("application_date")
                    ),
                    key=f"application_date_{job_id}",
                    format="YYYY-MM-DD",
                )

                follow_up_date = st.date_input(
                    "Follow-up date",
                    value=parse_saved_date(
                        job.get("follow_up_date")
                    ),
                    key=f"follow_up_date_{job_id}",
                    format="YYYY-MM-DD",
                )

                notes = st.text_area(
                    "Notes",
                    value=str(job.get("notes") or ""),
                    key=f"notes_{job_id}",
                    placeholder=(
                        "Recruiter details, application notes, "
                        "interview preparation..."
                    ),
                )

                if st.button(
                    "Save tracking details",
                    key=f"save_tracking_{job_id}",
                ):
                    saved = update_job_tracking(
                        job_id=job_id,
                        status=new_status,
                        application_date=(
                            application_date.isoformat()
                            if application_date
                            else None
                        ),
                        follow_up_date=(
                            follow_up_date.isoformat()
                            if follow_up_date
                            else None
                        ),
                        notes=notes,
                    )

                    if saved:
                        refresh_jobs()
                        st.toast("Application tracking saved.")
                        st.rerun()
                    else:
                        st.error("Could not save tracking details.")


# ---------------- CSV EXPORT ----------------

st.divider()
st.subheader("Export results")

export_columns = [
    "title",
    "company",
    "location",
    "category",
    "score",
    "status",
    "application_date",
    "follow_up_date",
    "notes",
    "source",
    "posted_at",
    "url",
]

available_columns = [
    column
    for column in export_columns
    if column in filtered.columns
]

csv_data = filtered[available_columns].to_csv(index=False)

st.download_button(
    "⬇️ Download filtered jobs as CSV",
    data=csv_data,
    file_name="ai_job_hunter_jobs.csv",
    mime="text/csv",
    disabled=filtered.empty,
)

st.caption(
    "Listings come from public job feeds. Verify eligibility, "
    "requirements and deadlines on the employer's official posting."
)