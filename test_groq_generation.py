
import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise SystemExit(
        "ERROR: GROQ_API_KEY was not found in .env"
    )

url = "https://api.groq.com/openai/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}

payload = {
    "model": "openai/gpt-oss-20b",
    "messages": [
        {
            "role": "system",
            "content": (
                "You are a professional job application assistant. "
                "Write truthful, concise application materials. "
                "Never invent qualifications or experience."
            ),
        },
        {
            "role": "user",
            "content": (
                "Write a short, professional cover-letter paragraph "
                "for an entry-level AI Automation Intern applicant "
                "with a Software Engineering degree, Python skills, "
                "and experience building a personal AI Gmail assistant. "
                "Use only the facts provided."
            ),
        },
    ],
    "temperature": 0.3,
    "max_completion_tokens": 400,
    "reasoning_effort": "low",
}

print("Testing Groq text generation...")

try:
    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=60,
    )

    if response.status_code == 200:
        result = response.json()
        message = result["choices"][0]["message"]["content"]

        print("\nSUCCESS: AI generation is working.")
        print("\nGenerated cover-letter paragraph:\n")
        print(message)

    else:
        print("API request failed:", response.status_code)
        print(response.text[:1000])

except requests.RequestException as error:
    print("Connection error:", error)
