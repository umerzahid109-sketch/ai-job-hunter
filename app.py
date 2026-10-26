import inspect
from datetime import date
from io import BytesIO

import pandas as pd
import streamlit as st
from pypdf import PdfReader

from src.ai_application_assistant import (
    generate_cover_letter,
    tailor_cv,
    verify_generated_text,
)
from src.cv_matcher import calculate_cv_match
from src.database import (
    add_job,
    get_all_jobs,
    get_due_follow_ups,
    initialize_database,
    update_job_match,
    update_job_status,
    update_job_tracking,
)
from src.job_matcher import match_job
from src.job_sources import get_live_jobs
from src.linkedin_search import (
    build_linkedin_search_url,
    get_search_categories,
)
from src.rag.cv_evidence import CVEvidenceRetriever

st.set_page_config(
    page_title="AI Job Hunter",
    page_icon="🎯",
    layout="wide",
)

# Newer Streamlit versions replace use_container_width=True with
# width="stretch". Detect which one this install supports.
if "width" in inspect.signature(st.button).parameters:
    STRETCH = {"width": "stretch"}
else:
    STRETCH = {"use_container_width": True}

STATUS_OPTIONS = [
    "New",
    "To Apply",
    "Applied",
    "Interview",
    "Rejected",
    "Offer",
    "Skipped",
]

STATUS_ICONS = {
    "New": "⚪",
    "To Apply": "🟣",
    "Applied": "🔵",
    "Interview": "🟡",
    "Rejected": "🔴",
    "Offer": "🟢",
    "Skipped": "⚫",
}

CAREER_CATEGORIES = [
    "Frontend / Web",
    "AI / ML / Automation",
    "Cybersecurity",
    "SQA / Testing",
    "Software Engineering",
    "Other",
]

DEFAULT_CATEGORIES = [
    "Frontend / Web",
    "AI / ML / Automation",
    "Cybersecurity",
    "SQA / Testing",
    "Software Engineering",
]

DEFAULT_STATUSES = ["New", "To Apply", "Applied", "Interview", "Offer"]


# ---------------- HELPERS ----------------


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


def clean_text(value, default=""):
    """Return a display-safe string; treats None/NaN/blank as default."""
    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass

    text = str(value).strip()
    return text if text else default


def match_label_for(score):
    if score >= 80:
        return "🟢", "Strong match"
    if score >= 60:
        return "🔵", "Good match"
    if score >= 40:
        return "🟡", "Possible match"
    return "⚪", "Lower match"


def render_audit_table(claims, flagged_message, clean_message):
    """Show an evidence audit as a table plus a summary message."""
    if not claims:
        st.info("No candidate-specific claims were returned.")
        return

    st.dataframe(
        [
            {
                "Claim": item.get("claim", ""),
                "Status": item.get("status", "Not provided"),
                "Exact quotation found": item.get(
                    "evidence_exists_verbatim", False
                ),
                "Supporting evidence": item.get("evidence", ""),
                "Reason": item.get("reason", ""),
            }
            for item in claims
        ],
        hide_index=True,
        **STRETCH,
    )

    flagged_count = sum(
        "FLAGGED" in str(item.get("status", "")).upper()
        for item in claims
    )

    if flagged_count:
        st.warning(f"{flagged_count} claim(s) flagged for review. {flagged_message}")
    else:
        st.info(f"No claims were flagged by this audit. {clean_message}")


def render_pagination(position, current_page, total_pages):
    """Previous / page indicator / Next controls."""
    previous_col, page_col, next_col = st.columns([1, 2, 1])

    with previous_col:
        if st.button(
            "← Previous",
            disabled=current_page <= 1,
            key=f"jobs_previous_page_{position}",
            **STRETCH,
        ):
            st.session_state["jobs_current_page"] = current_page - 1
            st.rerun()

    with page_col:
        st.markdown(
            f"<div style='text-align:center; padding-top:8px;'>"
            f"Page <strong>{current_page}</strong> of "
            f"<strong>{total_pages}</strong></div>",
            unsafe_allow_html=True,
        )

    with next_col:
        if st.button(
            "Next →",
            disabled=current_page >= total_pages,
            key=f"jobs_next_page_{position}",
            **STRETCH,
        ):
            st.session_state["jobs_current_page"] = current_page + 1
            st.rerun()


# ---------------- HEADER ----------------

st.title("🎯 AI Job Hunter")
st.caption(
    "Discover remote jobs, track applications, and manage follow-ups "
    "from one workspace."
)

# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.header("Job discovery")

    if st.button(
        "🔄 Fetch latest jobs",
        **STRETCH,
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
                elif update_job_match(
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
        **STRETCH,
    ):
        refresh_jobs()
        st.rerun()

    summary = st.session_state.get("last_fetch_summary")

    if summary:
        st.caption(
            f"Last fetch — retrieved {summary['retrieved']}, "
            f"new {summary['added']}, "
            f"refreshed {summary['refreshed']}."
        )

        if summary["errors"]:
            with st.expander(f"⚠️ {len(summary['errors'])} source warning(s)"):
                for error in summary["errors"]:
                    st.warning(str(error))

    st.divider()
    st.header("🔎 LinkedIn search")

    linkedin_category = st.selectbox(
        "Target role",
        options=get_search_categories(),
        key="linkedin_category",
    )

    linkedin_location = st.text_input(
        "LinkedIn job location",
        value="Karachi, Pakistan",
        key="linkedin_location",
    )

    if st.button(
        "Prepare LinkedIn Search",
        **STRETCH,
    ):
        if linkedin_location.strip():
            st.session_state["linkedin_search_url"] = (
                build_linkedin_search_url(
                    linkedin_category,
                    linkedin_location.strip(),
                )
            )
        else:
            st.warning("Enter a location first.")

    if st.session_state.get("linkedin_search_url"):
        st.link_button(
            "Open LinkedIn Jobs",
            st.session_state["linkedin_search_url"],
            **STRETCH,
        )


# ---------------- LOAD DATABASE ----------------

jobs_df = load_jobs()

if jobs_df.empty:
    st.info(
        "Your job list is empty. Use 'Fetch latest jobs' in the sidebar "
        "to get started. The CV Matcher and AI Career Tools tabs work "
        "without any saved jobs."
    )

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


# ---------------- TABS ----------------

dashboard_tab, jobs_tab, cv_tab, ai_tab = st.tabs([
    "📊 Dashboard",
    "🔍 Find Jobs",
    "📄 CV Matcher",
    "✍️ AI Career Tools",
])


# =====================================================================
# DASHBOARD
# =====================================================================

with dashboard_tab:
    st.subheader("Your job search at a glance")
    st.caption(
        "Track opportunities, monitor application progress, "
        "and stay on top of follow-ups."
    )

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
    st.caption(
        "Employers you planned to contact after applying."
    )

    today = date.today()
    reminders = []

    for item in get_due_follow_ups(today.isoformat()) or []:
        due_date = parse_saved_date(item["follow_up_date"])

        if due_date is None:
            continue

        reminders.append((due_date, item))

    reminders.sort(key=lambda entry: entry[0])

    if reminders:
        overdue_count = sum(1 for d, _ in reminders if d < today)
        today_count = sum(1 for d, _ in reminders if d == today)

        summary_cols = st.columns(2)
        summary_cols[0].metric("Overdue follow-ups", overdue_count)
        summary_cols[1].metric("Due today", today_count)

        for due_date, item in reminders:
            days_difference = (due_date - today).days
            heading = f"**{item['company']}** — {item['title']}"

            if days_difference < 0:
                st.warning(
                    f"{heading}\n\n"
                    f"Overdue by {abs(days_difference)} day(s) · "
                    f"Due {due_date.isoformat()}"
                )
            elif days_difference == 0:
                st.info(
                    f"{heading}\n\n"
                    "Due today · Consider sending a follow-up."
                )
            else:
                st.info(
                    f"{heading}\n\n"
                    f"Due in {days_difference} day(s) · "
                    f"{due_date.isoformat()}"
                )
    else:
        st.success("You're all caught up!")
        st.caption(
            "No follow-ups are due right now. Add follow-up dates in "
            "'Application tracking' on a job card to see reminders here."
        )

    # ---------------- PIPELINE + TOP MATCHES ----------------

    if total_jobs > 0:
        st.divider()

        chart_col, top_col = st.columns([1, 1])

        with chart_col:
            st.subheader("Applications by status")
            status_counts = (
                jobs_df["status"]
                .value_counts()
                .reindex(STATUS_OPTIONS, fill_value=0)
            )
            st.bar_chart(status_counts)

        with top_col:
            st.subheader("Top matches to act on")
            st.caption(
                "Highest-scoring jobs you haven't rejected or skipped."
            )

            top_jobs = (
                jobs_df[~jobs_df["status"].isin(["Rejected", "Skipped"])]
                .sort_values("score", ascending=False)
                .head(5)
            )

            if top_jobs.empty:
                st.info("No active jobs to show yet.")
            else:
                for _, top_job in top_jobs.iterrows():
                    top_title = clean_text(top_job["title"], "Untitled role")
                    top_company = clean_text(
                        top_job["company"], "Unknown company"
                    )
                    top_url = clean_text(top_job["url"])
                    top_score = int(top_job["score"])
                    top_status = clean_text(top_job["status"], "New")
                    icon, _ = match_label_for(top_score)

                    if top_url.startswith(("https://", "http://")):
                        title_md = f"[{top_title}]({top_url})"
                    else:
                        title_md = top_title

                    st.markdown(
                        f"{icon} **{top_score}/100** · {title_md}  \n"
                        f"{top_company} · "
                        f"{STATUS_ICONS.get(top_status, '⚪')} {top_status}"
                    )


# =====================================================================
# CV MATCHER
# =====================================================================

with cv_tab:
    st.subheader("📄 CV-to-Job Matcher")

    st.write(
        "Compare your CV with a job description to identify "
        "overlapping terms and potential areas to improve."
    )

    st.caption(
        "Your PDF text is extracted locally. The match score is "
        "a text-similarity measure, not a hiring probability."
    )

    uploaded_cv = st.file_uploader(
        "Upload your CV (PDF)",
        type=["pdf"],
        key="uploaded_cv",
        help="For best results, use a PDF with selectable text.",
    )

    cv_text = ""

    if uploaded_cv is not None:
        try:
            reader = PdfReader(BytesIO(uploaded_cv.getvalue()))
            cv_text = " ".join(
                page.extract_text() or ""
                for page in reader.pages
            ).strip()

            if cv_text:
                st.success("CV text extracted successfully.")

                with st.expander("Preview extracted CV text"):
                    st.text(cv_text[:4000])

                    if len(cv_text) > 4000:
                        st.caption("Preview shortened to 4,000 characters.")
            else:
                st.warning(
                    "No selectable text was found. Try a text-based "
                    "PDF instead of a scanned CV."
                )

        except Exception as exc:
            st.error(f"Unable to read this PDF: {exc}")

    # Share with the AI Career Tools tab.
    st.session_state["cv_text"] = cv_text

    st.markdown("#### Job description")

    job_description = st.text_area(
        "Paste the job description",
        height=220,
        placeholder=(
            "Include the role's responsibilities, qualifications, "
            "required skills, and experience."
        ),
        key="cv_matcher_description",
        label_visibility="collapsed",
    )

    analyze_col, clear_col = st.columns([2, 1])

    with analyze_col:
        analyze_clicked = st.button(
            "🔍 Analyze CV match",
            type="primary",
            **STRETCH,
        )

    with clear_col:
        clear_clicked = st.button(
            "Clear results",
            **STRETCH,
        )

    if clear_clicked:
        st.session_state.pop("cv_match_result", None)
        st.rerun()

    if analyze_clicked:
        if not cv_text:
            st.warning("Upload a readable PDF CV first.")

        elif not job_description.strip():
            st.warning("Paste a job description first.")

        else:
            try:
                st.session_state["cv_match_result"] = calculate_cv_match(
                    cv_text,
                    job_description,
                )
            except Exception as exc:
                st.error(f"Unable to analyze the CV match: {exc}")

    result = st.session_state.get("cv_match_result")

    if result is not None:
        st.divider()
        st.subheader("Your matching results")

        match_score = max(0, min(100, int(result["match_score"])))

        if match_score >= 75:
            score_label = "Higher text similarity"
        elif match_score >= 50:
            score_label = "Moderate text similarity"
        else:
            score_label = "Lower text similarity"

        score_col, summary_col = st.columns([1, 2])

        with score_col:
            st.metric("Text similarity", f"{match_score}%")

        with summary_col:
            st.markdown(f"**{score_label}**")
            st.progress(match_score / 100)
            st.caption(
                "This score reflects text similarity only. "
                "It does not predict hiring outcomes."
            )

        matched_terms = result.get("matched_terms", [])
        missing_terms = result.get("missing_terms", [])

        matched_col, missing_col = st.columns(2)

        with matched_col:
            st.markdown("#### ✅ Overlapping terms")
            st.metric("Terms identified", len(matched_terms))

            if matched_terms:
                st.write(", ".join(matched_terms))
            else:
                st.info("No overlapping terms were identified.")

        with missing_col:
            st.markdown("#### 🔎 Potential gaps")
            st.metric("Terms to review", len(missing_terms))

            if missing_terms:
                st.write(", ".join(missing_terms))
            else:
                st.info("No job-only terms were identified.")

        st.info(
            "Next step: review the potential gaps against your actual "
            "experience. Add a term to your CV only when it accurately "
            "describes your skills or experience. Some terms may be "
            "ordinary words rather than job requirements."
        )


    # ---------------- CV EVIDENCE RETRIEVAL ----------------

    st.divider()
    st.subheader("🔎 Find supporting evidence in your CV")
    st.write(
        "Retrieve passages from your CV that are semantically relevant "
        "to the target job description."
    )
    st.caption(
        "Retrieved passages are references for manual review, not proof "
        "that a generated claim is supported. Similarity scores are not "
        "probabilities."
    )

    if st.button(
        "Find relevant CV evidence",
        key="find_cv_evidence_button",
        **STRETCH,
    ):
        if not cv_text.strip():
            st.warning("Upload a readable PDF CV first.")
        elif not job_description.strip():
            st.warning("Paste a job description first.")
        else:
            try:
                import hashlib

                # Keep the retriever in this session and rebuild it only
                # when the uploaded CV changes.
                cv_hash = hashlib.sha256(
                    cv_text.encode("utf-8")
                ).hexdigest()

                cached_hash = st.session_state.get(
                    "cv_evidence_source_hash"
                )

                if (
                    cached_hash != cv_hash
                    or "cv_evidence_retriever" not in st.session_state
                ):
                    with st.spinner(
                        "Building a semantic index of your CV..."
                    ):
                        st.session_state["cv_evidence_retriever"] = (
                            CVEvidenceRetriever.from_cv(
                                cv_text,
                                source_id="candidate_cv",
                                chunk_size=20,
                                overlap=5,
                            )
                        )
                        st.session_state["cv_evidence_source_hash"] = (
                            cv_hash
                        )

                with st.spinner("Finding relevant CV passages..."):
                    evidence_results = (
                        st.session_state["cv_evidence_retriever"].retrieve(
                            job_description[:3000],
                            top_k=5,
                            min_score=0.35,
                        )
                    )

                st.session_state["cv_evidence_results"] = [
                    {
                        "text": item.chunk.text,
                        "score": item.score,
                    }
                    for item in evidence_results
                ]
                st.session_state["cv_evidence_query_hash"] = hashlib.sha256(
                    (
                        cv_hash
                        + "\0"
                        + job_description
                    ).encode("utf-8")
                ).hexdigest()

            except Exception as exc:
                st.session_state.pop("cv_evidence_results", None)
                st.error(f"Unable to retrieve CV evidence: {exc}")

    # Show results only when they match the current CV and job description.
    evidence_results = st.session_state.get("cv_evidence_results")
    evidence_query_hash = st.session_state.get("cv_evidence_query_hash")

    if cv_text.strip() and job_description.strip():
        import hashlib

        current_cv_hash = hashlib.sha256(
            cv_text.encode("utf-8")
        ).hexdigest()
        current_query_hash = hashlib.sha256(
            (
                current_cv_hash
                + "\0"
                + job_description
            ).encode("utf-8")
        ).hexdigest()

        if evidence_query_hash == current_query_hash:
            if evidence_results:
                st.markdown("#### Retrieved passages")

                for index, item in enumerate(evidence_results, start=1):
                    with st.container(border=True):
                        st.markdown(f"**Passage {index}**")
                        st.caption(
                            f"Semantic similarity: {item['score']:.3f}"
                        )
                        st.write(item["text"])

                st.caption(
                    "Review each passage in context before using it in an "
                    "application. A retrieved passage may be incomplete "
                    "or insufficient to support a particular claim."
                )
            else:
                st.info(
                    "No passages met the current similarity threshold. "
                    "This does not necessarily mean your CV lacks relevant "
                    "experience; review the CV manually or try a more "
                    "specific job description."
                )


# =====================================================================
# AI CAREER TOOLS
# =====================================================================

with ai_tab:
    st.subheader("✍️ AI Application Assistant")
    st.caption(
        "Generate job-specific CV suggestions and cover letters, "
        "then audit their claims against your uploaded CV."
    )

    # CV and job description come from the CV Matcher tab.
    ai_cv_text = st.session_state.get("cv_text", "")
    ai_job_description = st.session_state.get("cv_matcher_description", "")

    if not ai_cv_text or not ai_job_description.strip():
        st.info(
            "Upload your CV and enter a target job description in the "
            "CV Matcher tab to use this feature."
        )
    else:
        st.success("CV and job description detected. Ready to generate.")

        col1, col2 = st.columns(2)

        # ---------------- GENERATE CONTENT ----------------

        with col1:
            if st.button(
                "Tailor My CV",
                key="ai_tailor_cv_button",
                **STRETCH,
            ):
                with st.spinner("Analyzing your CV against the job..."):
                    try:
                        st.session_state["ai_cv_suggestions"] = tailor_cv(
                            ai_cv_text,
                            ai_job_description,
                        )

                        # Clear the previous editor and audit for new content.
                        st.session_state.pop("ai_cv_suggestions_editor", None)
                        st.session_state.pop("ai_cv_audit", None)

                        st.success("CV suggestions generated.")
                    except (RuntimeError, ValueError) as error:
                        st.error(str(error))

        with col2:
            if st.button(
                "Generate Cover Letter",
                key="ai_generate_cover_letter_button",
                **STRETCH,
            ):
                with st.spinner("Writing your tailored cover letter..."):
                    try:
                        st.session_state["ai_cover_letter"] = (
                            generate_cover_letter(
                                ai_cv_text,
                                ai_job_description,
                            )
                        )

                        # Clear the previous editor and audit for new content.
                        st.session_state.pop("ai_cover_letter_editor", None)
                        st.session_state.pop("ai_cover_letter_audit", None)

                        st.success("Cover letter generated.")
                    except (RuntimeError, ValueError) as error:
                        st.error(str(error))

        # ---------------- CV SUGGESTIONS ----------------

        if st.session_state.get("ai_cv_suggestions"):
            st.divider()
            st.subheader("CV Improvement Suggestions")

            st.warning(
                "Review every recommendation. Do not add unsupported "
                "skills, projects, responsibilities, or achievements "
                "to your CV."
            )

            st.text_area(
                "Generated CV review — edit before using",
                value=st.session_state["ai_cv_suggestions"],
                height=350,
                key="ai_cv_suggestions_editor",
            )

            # Read the current editable text, not just the original output.
            current_cv_suggestions = st.session_state.get(
                "ai_cv_suggestions_editor",
                st.session_state["ai_cv_suggestions"],
            )

            if st.button(
                "Audit CV Suggestions",
                key="ai_audit_cv_button",
                **STRETCH,
            ):
                with st.spinner("Checking claims against your CV..."):
                    try:
                        st.session_state["ai_cv_audit"] = (
                            verify_generated_text(
                                ai_cv_text,
                                current_cv_suggestions,
                            )
                        )
                    except (RuntimeError, ValueError) as error:
                        st.error(f"CV audit failed: {error}")

            if "ai_cv_audit" in st.session_state:
                st.markdown("#### CV Evidence Audit")
                st.caption(
                    "Run the audit again after editing your suggestions. "
                    "A matching quotation is not proof that a claim is "
                    "fully accurate."
                )

                render_audit_table(
                    st.session_state["ai_cv_audit"],
                    flagged_message=(
                        "Remove or correct unsupported claims before "
                        "using the suggestions."
                    ),
                    clean_message=(
                        "Manually verify every claim before using it."
                    ),
                )

            st.download_button(
                "Download CV Suggestions",
                data=current_cv_suggestions,
                file_name="cv_improvement_suggestions.txt",
                mime="text/plain",
                key="ai_download_cv_suggestions",
            )

        # ---------------- COVER LETTER ----------------

        if st.session_state.get("ai_cover_letter"):
            st.divider()
            st.subheader("Tailored Cover Letter")

            st.text_area(
                "Review and edit your cover letter",
                value=st.session_state["ai_cover_letter"],
                height=350,
                key="ai_cover_letter_editor",
            )

            current_cover_letter = st.session_state.get(
                "ai_cover_letter_editor",
                st.session_state["ai_cover_letter"],
            )

            if st.button(
                "Audit Cover Letter",
                key="ai_audit_cover_letter_button",
                **STRETCH,
            ):
                with st.spinner(
                    "Checking cover letter claims against your CV..."
                ):
                    try:
                        st.session_state["ai_cover_letter_audit"] = (
                            verify_generated_text(
                                ai_cv_text,
                                current_cover_letter,
                            )
                        )
                    except (RuntimeError, ValueError) as error:
                        st.error(f"Cover letter audit failed: {error}")

            if "ai_cover_letter_audit" in st.session_state:
                st.markdown("#### Cover Letter Evidence Audit")
                st.caption(
                    "Check that every statement describes your actual "
                    "experience. Re-run this audit after editing."
                )

                render_audit_table(
                    st.session_state["ai_cover_letter_audit"],
                    flagged_message=(
                        "Correct or remove unsupported statements."
                    ),
                    clean_message=(
                        "Manually verify every claim before sending."
                    ),
                )

            st.download_button(
                "Download Cover Letter",
                data=current_cover_letter,
                file_name="tailored_cover_letter.txt",
                mime="text/plain",
                key="ai_download_cover_letter",
            )


# =====================================================================
# FIND JOBS
# =====================================================================

with jobs_tab:
    st.subheader("Find your next opportunity")

    # ---------------- FILTERS ----------------

    with st.expander("🔧 Filters and sorting", expanded=True):
        search_text = st.text_input(
            "Search title, company or description",
            placeholder="e.g. React, Python, cybersecurity",
            key="filter_search_text",
        )

        filter_col1, filter_col2 = st.columns(2)

        with filter_col1:
            selected_categories = st.multiselect(
                "Career categories",
                options=CAREER_CATEGORIES,
                default=DEFAULT_CATEGORIES,
                key="filter_categories",
            )

        with filter_col2:
            selected_statuses = st.multiselect(
                "Application status",
                options=STATUS_OPTIONS,
                default=DEFAULT_STATUSES,
                key="filter_statuses",
            )

        filter_col3, filter_col4 = st.columns(2)

        with filter_col3:
            minimum_score = st.slider(
                "Minimum relevance score",
                min_value=0,
                max_value=100,
                value=20,
                step=5,
                key="filter_min_score",
            )

        with filter_col4:
            sort_option = st.selectbox(
                "Sort jobs by",
                [
                    "Highest relevance",
                    "Most recently discovered",
                    "Company name",
                    "Job title",
                ],
                key="filter_sort",
            )

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
        filtered = filtered[filtered["category"].isin(selected_categories)]
    else:
        filtered = filtered.iloc[0:0]

    filtered = filtered[filtered["score"] >= minimum_score]

    if selected_statuses:
        filtered = filtered[filtered["status"].isin(selected_statuses)]
    else:
        filtered = filtered.iloc[0:0]

    if sort_option == "Highest relevance":
        filtered = filtered.sort_values(
            ["score", "discovered_at"],
            ascending=[False, False],
        )
    elif sort_option == "Most recently discovered":
        filtered = filtered.sort_values("discovered_at", ascending=False)
    elif sort_option == "Company name":
        filtered = filtered.sort_values("company")
    else:
        filtered = filtered.sort_values("title")

    filtered_count = len(filtered)

    st.subheader(f"Matching jobs ({filtered_count})")

    # ---------------- PAGINATION + JOB CARDS ----------------

    if filtered.empty:
        st.warning(
            "No jobs match these filters. Try lowering the relevance "
            "score or selecting more categories and statuses."
        )
    else:
        page_size = st.selectbox(
            "Jobs per page",
            options=[10, 20, 50],
            index=0,
            key="jobs_page_size",
        )

        # Go back to page 1 only when the filters or sorting change.
        page_signature = (
            search_text.strip(),
            tuple(sorted(selected_categories)),
            minimum_score,
            tuple(sorted(selected_statuses)),
            sort_option,
            page_size,
        )

        if st.session_state.get("jobs_page_signature") != page_signature:
            st.session_state["jobs_page_signature"] = page_signature
            st.session_state["jobs_current_page"] = 1

        total_pages = max(
            1,
            (filtered_count + page_size - 1) // page_size,
        )

        current_page = max(
            1,
            min(st.session_state.get("jobs_current_page", 1), total_pages),
        )
        st.session_state["jobs_current_page"] = current_page

        start_index = (current_page - 1) * page_size
        end_index = min(start_index + page_size, filtered_count)

        paginated_filtered = filtered.iloc[start_index:end_index]

        st.caption(
            f"Showing jobs {start_index + 1}–{end_index} "
            f"of {filtered_count} matching jobs."
        )

        render_pagination("top", current_page, total_pages)

        st.divider()

        for _, job in paginated_filtered.iterrows():
            title = clean_text(job["title"], "Untitled role")
            company = clean_text(job["company"], "Unknown company")
            category = clean_text(job["category"], "Other")
            location = clean_text(job["location"], "Not specified")
            url = clean_text(job["url"])
            status = clean_text(job["status"], "New")
            score = int(job["score"])
            job_id = int(job["id"])
            source = clean_text(job.get("source"), "Unknown")
            posted_at = clean_text(job.get("posted_at"))

            match_icon, match_label = match_label_for(score)
            status_icon = STATUS_ICONS.get(status, "⚪")

            with st.container(border=True):
                heading_col, score_col = st.columns([4, 1])

                with heading_col:
                    st.markdown(f"### {title}")
                    st.markdown(f"**{company}**")
                    st.caption(
                        f"📍 {location} · {status_icon} {status}"
                    )

                with score_col:
                    st.metric("Match score", f"{score}/100")
                    st.caption(f"{match_icon} {match_label}")

                detail_col1, detail_col2, detail_col3 = st.columns(3)

                detail_col1.caption("CATEGORY")
                detail_col1.write(category)

                detail_col2.caption("SOURCE")
                detail_col2.write(source)

                if posted_at:
                    detail_col3.caption("POSTED")
                    detail_col3.write(posted_at)

                button_col, status_col = st.columns([1, 2])

                with button_col:
                    if url.startswith(("https://", "http://")):
                        st.link_button(
                            "↗ Open job posting",
                            url,
                            **STRETCH,
                        )
                    else:
                        st.caption("No valid application URL available.")

                with status_col:
                    new_status = st.selectbox(
                        "Update application status",
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

                with st.expander("📄 View job description"):
                    description = clean_text(job.get("description"))

                    if description:
                        st.write(description[:5000])

                        if len(description) > 5000:
                            st.caption("Description shortened for display.")
                    else:
                        st.caption(
                            "No description was provided by the source."
                        )

                with st.expander("📌 Application tracking"):
                    application_date = st.date_input(
                        "Application date",
                        value=parse_saved_date(job.get("application_date")),
                        key=f"application_date_{job_id}",
                        format="YYYY-MM-DD",
                    )

                    follow_up_date = st.date_input(
                        "Follow-up date",
                        value=parse_saved_date(job.get("follow_up_date")),
                        key=f"follow_up_date_{job_id}",
                        format="YYYY-MM-DD",
                    )

                    notes = st.text_area(
                        "Notes",
                        value=clean_text(job.get("notes")),
                        key=f"notes_{job_id}",
                        placeholder=(
                            "Recruiter details, application notes, "
                            "interview preparation..."
                        ),
                    )

                    if st.button(
                        "Save tracking details",
                        key=f"save_tracking_{job_id}",
                        type="primary",
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

        st.divider()
        render_pagination("bottom", current_page, total_pages)

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
        column for column in export_columns if column in filtered.columns
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
