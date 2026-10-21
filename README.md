# AI Job Hunter 🎯

**An AI-Powered Job Discovery and Application Assistant**

AI Job Hunter is a Python-based application that helps candidates discover remote job opportunities, evaluate their CVs against job descriptions, and prepare tailored job applications with AI assistance.

Built with Streamlit, Python, and the Groq API, the project brings job discovery, application tracking, CV matching, and AI-powered career tools into one interactive application.

## 🚀 Features

- **Live Remote Job Discovery:** Retrieves publicly available job listings from Jobicy and Remotive.
- **Interactive Dashboard:** Provides a centralized interface for exploring opportunities and managing your job search.
- **Job Search and Filtering:** Browse and filter matching job listings using available search criteria.
- **Application Tracking:** Track job application statuses and organize opportunities throughout the application process.
- **Follow-up Reminders:** Support for managing application follow-ups.
- **CV-to-Job Matching:** Evaluate how well a candidate's CV aligns with a target job description.
- **AI-Powered CV Tailoring:** Generate role-specific CV improvement suggestions grounded in the candidate's existing experience.
- **AI Cover Letter Generation:** Create tailored cover letters using CV information and the target job description.
- **Generated Content Verification:** Review AI-generated application content against available CV evidence to help identify unsupported claims.
- **LinkedIn Search Support:** Includes functionality for LinkedIn-related job searching.
- **Automated Tests:** Includes scripts for testing application-assistance functionality and Groq API connectivity and generation.

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| User Interface | Streamlit |
| Data Processing | Pandas |
| Machine Learning | Scikit-learn |
| Generative AI Provider | Groq |
| Large Language Model | `openai/gpt-oss-20b` |
| AI Integration | Groq Chat Completions API |
| Job Data Sources | Jobicy, Remotive |
| Configuration | Environment variables and Streamlit TOML configuration |
| Version Control | Git and GitHub |

## 🤖 AI Model and Configuration

AI-powered application assistance uses the `openai/gpt-oss-20b` model through the Groq API.

The AI assistant supports CV tailoring, cover-letter generation, and evidence-based verification of generated application content.

**Model configuration**

- **Provider:** Groq
- **Model:** `openai/gpt-oss-20b`
- **API endpoint:** `https://api.groq.com/openai/v1/chat/completions`
- **API key environment variable:** `GROQ_API_KEY`

Generation parameters may vary between the application and individual test scripts.

## 📁 Project Structure

```text
ai-job-hunter/
├── .streamlit/
│   └── config.toml
├── src/
│   ├── ai_application_assistant.py
│   ├── cv_matcher.py
│   ├── database.py
│   ├── job_matcher.py
│   ├── job_sources.py
│   ├── linkedin_search.py
│   └── main.py
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── .gitattributes
├── README.md
├── test_application_assistant.py
├── test_groq_connection.py
└── test_groq_generation.py
```

### Main files

- **`app.py`** — Main Streamlit application and user interface.
- **`src/ai_application_assistant.py`** — AI-assisted CV tailoring, cover-letter generation, and generated-content verification.
- **`src/cv_matcher.py`** — CV-to-job-description matching functionality.
- **`src/database.py`** — Database operations for application data.
- **`src/job_matcher.py`** — Job-matching functionality.
- **`src/job_sources.py`** — Integration with public job listing sources.
- **`src/linkedin_search.py`** — LinkedIn-related search functionality.
- **`src/main.py`** — Main entry point for the job discovery workflow.
- **`.streamlit/config.toml`** — Streamlit theme configuration.
- **`requirements.txt`** — Python dependencies.
- **`.env.example`** — Example environment variable configuration.

## ⚙️ Installation and Setup

### Prerequisites

- Python installed on your system.
- Git installed on your system.
- Internet access for retrieving live job listings and accessing the AI API.
- A Groq API key for AI-powered features.

### 1. Clone the repository

```bash
git clone https://github.com/umerzahid109-sketch/ai-job-hunter.git
cd ai-job-hunter
```

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks environment activation, you can use the virtual environment's Python executable directly.

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Configure the Groq API

1. Visit the [Groq Console](https://console.groq.com/) and obtain an API key.
2. Create a `.env` file in the project root, following the format in `.env.example`.
3. Add your API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

4. Ensure your environment variable is loaded by the application before making API requests.

**Security:** Never commit your real API key or `.env` file to GitHub. Keep credentials private and ensure `.env` remains excluded by `.gitignore`.

### 5. Run the application

```powershell
streamlit run app.py
```

Once Streamlit starts, open the local URL displayed in your terminal.

## 🧪 Testing

Run the test scripts from the project's root directory.

### Test application assistance

```powershell
python test_application_assistant.py
```

### Test Groq API connectivity

```powershell
python test_groq_connection.py
```

### Test AI text generation

```powershell
python test_groq_generation.py
```

The Groq tests require valid API credentials and an internet connection. Results may vary depending on API availability, model access, and rate limits.

## 🔐 Security and Responsible AI

- Keep API keys and credentials out of source control.
- Review AI-generated CVs and cover letters before submitting applications.
- Verify that generated statements accurately reflect your actual skills, qualifications, and experience.
- Treat generated content as assistance rather than a guarantee of job suitability or application success.

## 👨‍💻 Author

**Umer Zahid**

Software Engineering Graduate | AI and Automation Projects

GitHub: [@umerzahid109-sketch](https://github.com/umerzahid109-sketch)

## 📄 License

No license has been specified for this repository. All rights remain with the respective copyright holder unless a license is added.
