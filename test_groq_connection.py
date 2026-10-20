
import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("ERROR: GROQ_API_KEY was not found in .env")
    raise SystemExit(1)

print("Checking Groq API connection...")

try:
    response = requests.get(
        "https://api.groq.com/openai/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=20,
    )

    if response.status_code == 200:
        models = response.json().get("data", [])
        print("SUCCESS: Groq API key is working.")
        print(f"Models available to your key: {len(models)}")

        for model in models:
            print("-", model.get("id", "Unknown model"))

    elif response.status_code == 401:
        print("ERROR: API key was rejected. Check .env.")

    elif response.status_code == 429:
        print("RATE LIMIT: The API returned a rate-limit response.")

    else:
        print("API returned status:", response.status_code)
        print("Response:", response.text[:500])

except requests.RequestException as error:
    print("CONNECTION ERROR:", error)
