# AI Job Hunter 🎯

**An AI-Powered Job Discovery and Application Assistant**

AI Job Hunter is a Python-based application designed to simplify the job search process. It retrieves publicly available remote job listings, helps candidates evaluate their CVs against job descriptions, and uses AI to assist with CV tailoring, cover-letter generation, and verification of generated application content.

The project combines job discovery, machine learning-based CV matching, and generative AI in an interactive Streamlit application.

## 🚀 Features

- **Live Job Discovery:** Retrieves job listings from public sources, including Jobicy and Remotive.
- **CV-to-Job Matching:** Evaluates the compatibility between a candidate's CV and a job description.
- **AI-Powered CV Tailoring:** Generates role-specific CV improvement suggestions grounded in the candidate's existing experience.
- **Cover Letter Generation:** Creates tailored cover letters based on the candidate's CV and target job description.
- **AI Content Verification:** Checks generated application content against CV evidence to help identify unsupported claims.
- **LinkedIn Search Support:** Includes a module for LinkedIn-related job searching.
- **Interactive Dashboard:** Uses Streamlit to provide a graphical interface for job discovery and application assistance.
- **Automated Testing:** Includes test scripts for CV assistance and Groq API connectivity and generation.

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
| Configuration | Environment variables |
| Version Control | Git and GitHub |

## 🤖 AI Model and Configuration

AI-powered application assistance uses the **`openai/gpt-oss-20b`** model through the Groq API.

The model supports features such as CV tailoring, cover-letter generation, and evidence-based verification.

### Model settings

- **Provider:** Groq
- **Model:** `openai/gpt-oss-20b`
- **API endpoint:** `https://api.groq.com/openai/v1/chat/completions`
- **Temperature:** `0.3`
- **Reasoning effort:** `low`
- **Maximum completion tokens:** `400` in the generation test configuration
- **API key environment variable:** `GROQ_API_KEY`

These settings reflect the project's Groq integration and generation-test configuration.

## 📁 Project Structure

```text
ai-job-hunter/
│
├── app.py
├── requirements.txt
├── .gitattributes
│
├── src/
│   ├── ai_application_assistant.py
│   ├── cv_matcher.py
│   └── linkedin_search.py
│
├── test_application_assistant.py
├── test_groq_connection.py
└── test_groq_generation.py
```

### Main files

- **`app.py`** — Streamlit application and user interface.
- **`src/ai_application_assistant.py`** — AI-assisted CV tailoring, cover-letter generation, and generated-content verification.
- **`src/cv_matcher.py`** — CV and job-description matching functionality.
- **`src/linkedin_search.py`** — LinkedIn-related search functionality.
- **`test_application_assistant.py`** — Tests application-assistance features.
- **`test_groq_connection.py`** — Tests connectivity to the Groq API.
- **`test_groq_generation.py`** — Tests AI text generation.
- **`requirements.txt`** — Python dependencies.

## ⚙️ Installation and Setup

### Prerequisites

- Python installed on your system.
- Git installed on your system.
- Internet access for retrieving live job listings and calling the AI API.
- A Groq API key for AI-powered functionality.

### 1. Clone the repository

```bash
git clone https://github.com/umerzahid109-sketch/ai-job-hunter.git
cd ai-job-hunter
```

### 2. Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks environment activation, you can run the environment's Python directly using `.venv\Scripts\python.exe`.

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Configure the Groq API

1. Visit the [Groq Console](https://console.groq.com/) and obtain an API key.
2. Create a `.env` file in the project's root directory.
3. Add your API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

4. Ensure the application loads the environment variable before making API requests.

**Security:** Never commit your real API key or `.env` file to GitHub. The `.env` file should remain excluded through `.gitignore`.

### 5. Run the application

```powershell
streamlit run app.py
```

Once Streamlit starts, open the local URL printed in your terminal to access the application.

## 🧪 Testing

Run the tests from the project root.

### Test application assistance

```powershell
python test_application_assistant.py
```

### Test the Groq API connection

```powershell
python test_groq_connection.py
```

### Test AI text generation

```powershell
python test_groq_generation.py
```

The Groq tests require valid API credentials and an internet connection. API availability, model access, and rate limits may affect their results.

## 🔐 Security and Responsible AI

- Keep API keys and other credentials out of source control.
- Use environment variables or a secure secrets manager to store credentials.
- Review AI-generated CV suggestions and cover letters before submitting applications.
- Ensure candidate claims remain consistent with actual skills, experience, and qualifications.
- Verify job details and application requirements using the original job listing or employer website.

## 🎯 Project Objectives

AI Job Hunter aims to:

1. Reduce the time required to discover relevant job opportunities.
2. Help candidates understand how their skills align with job requirements.
3. Simplify the preparation of role-specific CVs and cover letters.
4. Encourage evidence-based and factually accurate AI-generated application content.
5. Demonstrate practical integration of Python, machine learning, generative AI, APIs, and web application development.

## 🔮 Future Improvements

Potential enhancements include:

- More job-source integrations and advanced filtering.
- Personalized job recommendations based on skills and career preferences.
- Job bookmarking and application tracking.
- Improved CV parsing and skill extraction.
- More comprehensive automated tests.
- Deployment to a cloud hosting platform.

## 👨‍💻 Author

**Umer Zahid**

Software Engineering Graduate | AI and Automation Projects

GitHub: [@umerzahid109-sketch](https://github.com/umerzahid109-sketch)

## 📄 License

No license has been specified for this repository. All rights remain with the respective copyright holder unless a license is added.