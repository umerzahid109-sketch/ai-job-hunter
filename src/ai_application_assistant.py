
import json
import os
import re
import unicodedata

import requests
from dotenv import load_dotenv


load_dotenv()

API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "openai/gpt-oss-20b"


FACTUAL_ACCURACY_RULES = """
STRICT FACTUAL ACCURACY RULES:

1. The candidate CV is the only source of facts about the candidate.
2. Do not invent projects, responsibilities, technologies, employers,
   achievements, implementation details, or numerical results.
3. A general skills list establishes that a skill is listed. It does
   not establish that the skill was used in a particular project.
4. Never infer Python usage from a Google Apps Script project.
5. Never infer React usage in a project merely because React is listed.
6. Do not infer REST API experience from an unrelated API mention.
7. Do not infer expertise, proficiency, seniority, or years of
   experience unless the CV explicitly supports those claims.
8. Do not turn a planned feature or target into a completed achievement.
9. Do not invent communication, leadership, teamwork, documentation,
   testing, troubleshooting, or presentation experience.
10. If evidence is missing, omit the claim or say:
    "Not established by the provided CV."
11. Distinguish recommendations from established candidate facts.
12. Prefer modest, factual wording over inflated claims.
13. When uncertain, leave the claim out.
"""


def _generate_text(system_prompt, user_prompt, max_tokens=1200):
    """Send a prompt to Groq and return the generated text."""

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY was not found. Check your project's .env file."
        )

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
        "max_completion_tokens": max_tokens,
        "reasoning_effort": "low",
    }

    try:
        response = requests.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=90,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Groq API error ({response.status_code}): "
                f"{response.text[:500]}"
            )

        data = response.json()
        content = data["choices"][0]["message"]["content"]

        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("The AI returned an empty response.")

        return content.strip()

    except requests.RequestException as error:
        raise RuntimeError(
            f"Could not connect to Groq: {error}"
        ) from error


def _validate_inputs(cv_text, job_description):
    """Reject missing inputs before calling the AI."""

    if not isinstance(cv_text, str) or not cv_text.strip():
        raise ValueError("Provide the candidate's CV text.")

    if not isinstance(job_description, str) or not job_description.strip():
        raise ValueError("Provide the target job description.")


def _normalize_text(value):
    """Normalize whitespace, Unicode, and quotation marks for comparisons."""

    value = unicodedata.normalize("NFKC", str(value))
    value = value.replace("\u2018", "'").replace("\u2019", "'")
    value = value.replace("\u201c", '"').replace("\u201d", '"')
    value = re.sub(r"\s+", " ", value).strip()

    while len(value) >= 2:
        if (
            value.startswith('"') and value.endswith('"')
        ) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1].strip()
        else:
            break

    return value.casefold()


def _parse_json_response(raw_result):
    """Parse a JSON object, tolerating Markdown code fences."""

    cleaned = raw_result.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned,
            count=1,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        # Recover a JSON object if the model added text around it.
        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start == -1 or end <= start:
            raise RuntimeError(
                "The AI returned invalid JSON. Please run the operation again."
            )

        try:
            result = json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError as error:
            raise RuntimeError(
                "The AI returned invalid JSON. Please run the operation again."
            ) from error

    if not isinstance(result, dict):
        raise RuntimeError("Unexpected AI response format.")

    return result


def tailor_cv(cv_text, job_description):
    """Suggest CV improvements grounded in the candidate's actual CV."""

    _validate_inputs(cv_text, job_description)

    system_prompt = f"""
You are an expert CV reviewer and job application assistant.

{FACTUAL_ACCURACY_RULES}

STRICT OUTPUT REQUIREMENTS:

- Support candidate-specific claims with evidence from the CV.
- Assess meaning rather than requiring every paraphrase to be identical.
- For exact supporting quotations, copy text directly from the CV.
- Never fabricate a quotation or present generated text as a CV quotation.
- Do not attribute general skills to a project without project-specific
  evidence.
- Never generate hypothetical achievements among recommended CV bullets.
- Label a proposed bullet "UNVERIFIED — DO NOT USE" if the CV does not
  establish the full claim.
- Phrase missing information as a recommendation, not a completed fact.
- Do not present job requirements as candidate skills.
- Separate supported keywords from unsupported role requirements.

Additional evidence rules:
- Never claim a programming language was used in a project just because
  it appears in the Skills section.
- Never describe a skill as demonstrated through project development
  unless the CV identifies a project that used that skill.
- Every suggested CV bullet must be factually supported by the CV.
- If the CV lists Python as a skill but does not connect Python to a
  specific project, do not suggest a bullet claiming Python was used
  in that project.
- Do not label a suggestion as supported merely because a quotation
  contains a related keyword. The evidence must support the entire claim.
- If a claim is unsupported, exclude it from the suggested bullets and
  explain the limitation separately.

"""

    user_prompt = f"""
Review the candidate's CV against the target job description.

CANDIDATE CV — ONLY SOURCE OF CANDIDATE FACTS:
<cv>
{cv_text[:18000]}
</cv>

JOB DESCRIPTION — SOURCE OF ROLE REQUIREMENTS:
<job_description>
{job_description[:10000]}
</job_description>

Return these sections:

1. ROLE FIT
List each important requirement, the supporting CV quotation, and
the match status. Distinguish listed skills from project experience.
For unsupported requirements, state:
"Not established by the provided CV."

2. PRIORITY IMPROVEMENTS
Give five practical recommendations. Missing experience may be added
only if the candidate can confirm it is true. Do not present suggestions
as achievements already completed.


3. SUGGESTED CV BULLETS

Only suggest complete, useful CV bullet points that describe
documented project work, employment, or achievements.

For every proposed bullet:
- Provide the complete proposed bullet.
- Provide one exact supporting quotation copied from the CV.
- Ensure the quotation supports every factual detail in the bullet.
- Do not combine separate CV statements into one quotation.
- Do not invent quotations, responsibilities, technologies, or results.

Never use an isolated skill name as a proposed CV bullet.
For example, do not output "Python", "React", or "SQL" as bullets.
Skills belong in the Skills section, not the project-experience section.

Do not rewrite the Skills section as project experience.
A skill listed in the CV does not establish that it was used in a
particular project.

If the CV does not provide enough evidence for another useful bullet,
return this exact message:
"No additional evidence-backed CV bullets can be suggested from the
provided CV."

Do not include unsupported bullets with warnings. Exclude them.


4. SKILL GAPS
List relevant requirements not demonstrated by the CV.
State "Not established by the provided CV" for each unsupported item.

5. KEYWORDS
Separate into:
- Supported by explicit CV evidence
- Relevant but not established by the provided CV

A keyword in the job description is not evidence that the candidate
has that skill.

STRICT RULES FOR SUGGESTED CV BULLETS:
- Suggest only bullets supported by the candidate's CV.
- A skill listed in the Skills section does not prove that the skill
  was used in a specific project.
- Never claim Python, JavaScript, React, or SQL was used in the
  Gmail Assistant unless the CV explicitly says so.
- Never invent project features, interfaces, database operations,
  achievements, metrics, testing, or performance improvements.
- If evidence supports only a general skill, do not turn it into a
  project achievement.
- Exclude unsupported bullets instead of generating them and relying
  on the evidence auditor to flag them.
- For each suggested bullet, ensure the supporting quotation proves
  the entire claim, not merely a related keyword.
-Do not suggest duplicate or near-duplicate bullets.
 Each bullet must add a distinct, evidence-supported detail.
 If only one useful bullet is supported, return only one bullet.
"""

    return _generate_text(
        system_prompt,
        user_prompt,
        max_tokens=2400,
    )



def generate_cover_letter(cv_text, job_description):
    """
    Generate a professional cover letter using only CV-supported facts.
    """
    _validate_inputs(cv_text, job_description)

    system_prompt = """
You are a careful professional cover-letter writer.

Rules:
1. Use only factual information supported by the candidate's CV.
2. Do not invent experience, achievements, proficiency levels, or project details.
3. Do not claim a technology was used in a project unless the CV confirms it.
4. Distinguish skills listed on the CV from skills demonstrated in projects.
5. Do not present problem-solving, communication, teamwork, or other
   personal strengths as proven facts unless the CV supports them.
6. You may express interest in the role and willingness to contribute.
7. Use the job description to tailor the letter, but never invent a match.
8. Write a concise, natural cover letter.
9. Do not include explanations, evidence audits, or Markdown tables.
10. Use [Your Name] as the signature placeholder.
11. Never claim that Python or any other specific programming language
    was used in a project unless the CV explicitly confirms it.
    Listing a language in the Skills section does not prove it was
    used in a particular project.

12. A Skills section proves only that a skill is listed on the CV.
    Never describe the candidate as proficient, expert, highly skilled,
    or having a strong foundation based solely on that listing.
    Prefer omitting broad skill lists from the letter unless relevant.
    If a listed skill is mentioned, describe it only as a skill listed
    on the CV, not as proven project experience.

13. Do not claim project activities or outcomes beyond what the CV
    explicitly describes.
14. Avoid promises such as "contribute effectively from day one."
    Use modest, professional language instead.
"""


    user_prompt = f"""
CANDIDATE CV — THE ONLY SOURCE OF CANDIDATE FACTS:
<cv>
{cv_text[:18000]}
</cv>

JOB DESCRIPTION — USE ONLY TO UNDERSTAND THE ROLE:
<job_description>
{job_description[:10000]}
</job_description>

Write a concise, professional cover letter tailored to this role.

STRICT FACTUAL REQUIREMENTS:

1. Do not describe a listed skill as "experience in" that skill.
   If relevant, say it is listed in the candidate's Skills section.
   Prefer discussing documented projects instead of listing skills.
2. Do not say or imply that these skills were used in the Gmail
   Assistant project unless the CV explicitly connects them to it.
3. Describe the Gmail Assistant only with these documented facts:
   it was built with Google Apps Script and an AI API, classifies
   important emails, and generates a daily digest.

   Do not add claims about user-friendly output, testing, debugging,
   time saved, accuracy, or other implementation details unless the
   CV explicitly supports them.

   You may explain that the documented project is relevant to AI
   automation, but do not claim additional responsibilities.
4. Do not invent API endpoints, data processing methods, debugging,
   testing, user feedback, results, or technical responsibilities.
5. Do not claim a "strong foundation", proficiency, or expertise
   without supporting evidence.
6. Do not turn job requirements into candidate experience.
7. Do not make unsupported claims about problem-solving,
   communication, teamwork, or leadership.
8. Use the job description to explain why the documented experience
   is relevant, without claiming unverified experience.
9. Do not repeat the entire Skills section merely to make the letter
   sound more technical.
10. Use [Your Name] as the signature placeholder.
11. Return only the cover letter, with no audit or explanation.

Before responding, check each sentence:
- Is every factual claim supported by the CV?
- Does it incorrectly connect a listed skill to a project?
- Does it add any project details absent from the CV?

Remove any sentence that fails these checks.
12. Do not call the candidate a "recent graduate" unless the CV
    explicitly supports that timing. "Software Engineering graduate"
    is sufficient.
"""

    return _generate_text(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=700,
    )


def verify_generated_text(cv_text, generated_text):
    """
    Audit AI-generated claims against CV evidence.

    Exact quotations are recorded as evidence metadata. Their absence
    does not automatically mean a paraphrased claim is unsupported.

    This is an evidence-assistance tool, not a guarantee of truth.
    """

    _validate_inputs(cv_text, generated_text)

    system_prompt = """
You are a careful CV evidence auditor.

Return ONLY valid JSON:
{
  "claims": [
    {
      "claim": "One candidate-specific factual claim",
      "evidence": "Exact supporting quotation from the CV, or empty string",
      "status": "supported, unsupported, or needs_review",
      "reason": "Brief explanation"
    }
  ]
}

RULES:

1. Split compound statements into distinct factual claims where practical.
2. Assess meaning, not just identical wording. A paraphrase can be supported.
3. Evidence must support the claim, not merely mention a related topic.
4. A general skills list proves that a skill is listed, not that it was
   used in a specific project.
5. Do not infer Python or React usage in the Gmail Assistant from a
   general skills list.
6. Do not infer REST API experience from an unrelated API mention.
7. Do not infer proficiency, expertise, or years of experience without
   direct evidence.
8. Check dates, numbers, percentages, technologies, responsibilities,
   outcomes, and deployment claims carefully.
9. Do not turn a target or planned feature into an achieved result.
10. Use "unsupported" when evidence is absent or contradicts the claim.
11. Use "needs_review" when evidence is ambiguous or incomplete.
12. Do not invent quotations. Copy the evidence exactly from the CV,
    or return an empty string.
13. Do not mark a claim unsupported solely because it is paraphrased.
14. The job description and generated text are not evidence about the
    candidate. Only the CV is evidence.
15. Audit candidate-specific factual claims, not general advice.
"""

    user_prompt = f"""
CANDIDATE CV — ONLY SOURCE OF EVIDENCE:
<cv>
{cv_text[:18000]}
</cv>

TEXT TO AUDIT:
<generated_text>
{generated_text[:12000]}
</generated_text>

Audit every meaningful candidate-specific factual claim.
Return valid JSON only.
"""

    raw_result = _generate_text(
        system_prompt,
        user_prompt,
        max_tokens=3000,
    )

    result = _parse_json_response(raw_result)

    if not isinstance(result.get("claims"), list):
        raise RuntimeError("Unexpected evidence auditor response.")

    normalized_cv = _normalize_text(cv_text)

    # Find the CV's skills section, if it follows a conventional heading.
    skills_lines = []
    in_skills_section = False

    section_headings = re.compile(
        r"^(education|experience|work experience|employment|"
        r"projects?|project experience|certifications?|"
        r"virtual job simulations|training|summary|profile|"
        r"technical skills|core skills|skills)\s*:?\s*$",
        flags=re.IGNORECASE,
    )

    for line in cv_text.splitlines():
        stripped = line.strip()

        if re.match(
            r"^(skills|technical skills|core skills)\s*:",
            stripped,
            flags=re.IGNORECASE,
        ):
            in_skills_section = True
            skills_lines.append(stripped.split(":", 1)[1])
            continue

        if in_skills_section:
            if section_headings.match(stripped):
                in_skills_section = False
                continue

            if stripped:
                skills_lines.append(stripped)

    normalized_skills = _normalize_text(" ".join(skills_lines))

    # Extract nearby Gmail Assistant project text to avoid confusing
    # general skills with project-specific technology usage.
    gmail_project_text = ""
    lines = cv_text.splitlines()

    for index, line in enumerate(lines):
        if re.search(
            r"personal\s+ai\s+gmail\s+assistant",
            line,
            flags=re.IGNORECASE,
        ):
            project_lines = [line]

            for following_line in lines[index + 1:index + 9]:
                if section_headings.match(following_line.strip()):
                    break
                project_lines.append(following_line)

            gmail_project_text = _normalize_text(" ".join(project_lines))
            break

    # A documented AI API used for email classification supports a
    # modest API-integration claim, but does not prove REST API usage
    # or Python usage in the project.
    documented_email_api_use = (
        bool(re.search(r"\bapi\b", normalized_cv))
        and bool(
            re.search(
                r"\b(classif\w*|categor\w*|process\w*)\b",
                normalized_cv,
            )
        )
        and bool(
            re.search(
                r"\b(gmail|email|e-mail)\b",
                normalized_cv,
            )
        )
    )

    audited_claims = []

    for item in result["claims"]:
        if not isinstance(item, dict):
            continue

        claim = str(item.get("claim", "")).strip()
        evidence = str(item.get("evidence", "")).strip()
        reason = str(item.get("reason", "")).strip()

        if not claim:
            continue

        normalized_claim = _normalize_text(claim)
        normalized_evidence = _normalize_text(evidence)

        # Exact quotation matching is a signal, not the entire decision.
        evidence_exists = (
            bool(normalized_evidence)
            and normalized_evidence in normalized_cv
        )

        ai_status = str(item.get("status", "")).strip().casefold()

        if ai_status not in {
            "supported",
            "unsupported",
            "needs_review",
        }:
            ai_status = "needs_review"

        # Do not infer proficiency from a skills list.
        proficiency_overclaim = bool(
            re.search(
                r"\b(expert|expertise|proficient|proficiency|"
                r"advanced|strong proficiency|highly skilled)\b",
                normalized_claim,
            )
        )

        # Detect explicit claims that Python or React was used in the
        # Personal AI Gmail Assistant project.
        gmail_project_mentioned = bool(
            re.search(
                r"\b(gmail assistant|personal ai gmail assistant)\b",
                normalized_claim,
            )
        )

        project_usage_wording = bool(
            re.search(
                r"\b(using|with|built|developed|implemented|"
                r"programmed|written|wrote|used|utilized)\b",
                normalized_claim,
            )
        )

        gmail_python_claim = (
            gmail_project_mentioned
            and project_usage_wording
            and "python" in normalized_claim
        )

        gmail_react_claim = (
            gmail_project_mentioned
            and project_usage_wording
            and "react" in normalized_claim
        )


        # JavaScript project claims must be supported by project evidence,
        # not merely by the general skills list.
        gmail_javascript_claim = (
            gmail_project_mentioned
            and project_usage_wording
            and bool(
                re.search(r"\bjavascript\b|\bjs\b", normalized_claim)
            )
        )

        gmail_javascript_evidence_missing = (
            gmail_javascript_claim
            and not re.search(
                r"\bjavascript\b|\bjs\b",
                gmail_project_text,
            )
        )

        # A daily digest does not, by itself, prove that it is scheduled.
        scheduling_claim = bool(
            re.search(
                r"\b(schedule|scheduled|scheduling|time-triggered|"
                r"automatically at \d|daily trigger)\b",
                normalized_claim,
            )
        )

        scheduling_evidence_missing = (
            scheduling_claim
            and not re.search(
                r"\b(schedule|scheduled|scheduling|time-triggered|"
                r"daily trigger|time-driven trigger)\b",
                normalized_cv,
            )
        )


        gmail_python_evidence_missing = (
            gmail_python_claim
            and "python" not in gmail_project_text
        )

        gmail_react_evidence_missing = (
            gmail_react_claim
            and "react" not in gmail_project_text
        )

        if gmail_javascript_evidence_missing:
            rule_notes.append(
                "The CV does not establish JavaScript usage in the "
                "Gmail Assistant project. A general JavaScript skill "
                "listing is not sufficient."
            )

        if scheduling_evidence_missing:
            rule_notes.append(
                "The CV mentions a daily digest but does not establish "
                "that its generation is scheduled."
            )


        # Explicit REST API claims require REST API evidence in the CV.
        rest_api_claim = bool(
            re.search(r"\brest\s*apis?\b", normalized_claim)
        )

        rest_api_evidence_missing = (
            rest_api_claim
            and not re.search(r"\brest\s*apis?\b", normalized_cv)
        )

        # Recognize modest API-integration claims when the CV describes
        # an API being used for email classification or processing.
        api_integration_claim = bool(
            re.search(
                r"\b(api integration|integrate APIs?|"
                r"integrated APIs?|ability to integrate APIs?)\b",
                normalized_claim,
            )
        )

        api_integration_supported = (
            api_integration_claim and documented_email_api_use
        )

        rule_notes = []

        if proficiency_overclaim:
            rule_notes.append(
                "The CV does not establish the claimed proficiency level."
            )

        if gmail_python_evidence_missing:
            rule_notes.append(
                "The CV does not establish Python usage in the Gmail "
                "Assistant project. A general Python skill listing "
                "is not sufficient."
            )

        if gmail_react_evidence_missing:
            rule_notes.append(
                "The CV does not establish React usage in the Gmail "
                "Assistant project. A general React skill listing "
                "is not sufficient."
            )

        if rest_api_evidence_missing:
            rule_notes.append(
                "The CV does not explicitly establish REST API experience."
            )

        hard_rule_failure = (
            proficiency_overclaim
            or gmail_python_evidence_missing
            or gmail_react_evidence_missing
            or rest_api_evidence_missing
            or gmail_javascript_evidence_missing
            or rest_api_evidence_missing
            or scheduling_evidence_missing
        )

        if hard_rule_failure:
            status = "FLAGGED — DO NOT USE WITHOUT REVIEW"

        elif api_integration_supported:
            status = "SEMANTIC SUPPORT — MANUAL REVIEW REQUIRED"

            if not evidence_exists:
                reason = (
                    "The CV documents API use for email classification or "
                    "processing. This supports a modest API-integration "
                    "claim, but does not establish REST API experience "
                    "or use of a particular programming language."
                )

        elif ai_status == "unsupported":
            status = "FLAGGED — DO NOT USE WITHOUT REVIEW"

        elif ai_status == "needs_review":
            status = "NEEDS REVIEW — EVIDENCE UNCLEAR"

        elif evidence_exists:
            status = "EVIDENCE MATCHED — MANUAL REVIEW REQUIRED"

        else:
            status = "NEEDS REVIEW — EVIDENCE UNCLEAR"

        if rule_notes:
            reason = " ".join(rule_notes)

        elif not reason:
            reason = (
                "Review the CV evidence and confirm that it supports "
                "the complete claim."
            )

        elif not evidence_exists and status.startswith("SEMANTIC SUPPORT"):
            reason = (
                f"{reason} The supplied quotation was not found verbatim "
                "in the CV, so verify the semantic match manually."
            )

        audited_claims.append({
            "claim": claim,
            "evidence": evidence,
            "status": status,
            "evidence_exists_verbatim": evidence_exists,
            "reason": reason,
        })

    return audited_claims
