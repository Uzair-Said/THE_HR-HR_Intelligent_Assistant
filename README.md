# THE HR - HR Intelligent Assistant 👥

An AI-powered HR policy management system that helps employees quickly understand HR policies and enables HR teams to maintain, check, and improve policy documents through an intuitive interface.

## 🎯 Features

### Employee Portal

- **Natural Language Q&A**: Ask questions about HR policies in plain English
- **Smart Document Retrieval**: AI finds and cites relevant policy documents
- **Quick Action Buttons**: Common queries like leave policy, benefits, and remote work
- **Chat History**: Track all conversations with the HR assistant
- **Multi-Document Reasoning**: Combines information from multiple sources

### Admin Dashboard (HR Team)

- **Secure Access**: Password-protected admin interface
- **Document Management**: Upload, manage, and delete HR documents
- **Policy Health Analysis**: Automated detection of:
  - Policy contradictions between documents
  - Missing essential clauses
  - Ambiguous language
  - Compliance gaps
- **Cross-Document Validation**: Identifies conflicts across multiple policies
- **Health Reports**: Export detailed analysis reports in JSON format
- **Dashboard Statistics**: Track documents, queries, and policy health metrics

## 🚀 Quick Start

### Installation

1. **Clone or download the project files**

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Run the application:**

```bash
streamlit run hr_assistant.py
```

4. **Access the application:**
   Open your browser and navigate to `http://localhost:8501`

## 📱 How to Use

### For Employees

1. Click on the **Employee Portal** tab
2. View available HR documents in the expandable section
3. Type your question in the chat interface or use quick action buttons
4. Receive AI-powered answers with document citations

### For HR Admins

1. Click on the **Admin Dashboard** tab
2. Login with credentials:
   - Username: `hradmin`
   - Password: `hrpass123`
3. **Load Sample Documents**: Click "Load Sample HR Documents" for demo data
4. **Upload Documents**: Add your own HR PDFs
5. **Analyze Policies**: Select any document and click "Analyze" to check for issues
6. **Export Reports**: Download policy health analysis as JSON

## 🎮 Demo Features

### Sample Documents Included

- **Employee Handbook 2024**: Comprehensive policies including leave, benefits, code of conduct
- **IT Security Policy**: Data protection and security guidelines
- **Compensation & Benefits Guide**: Detailed salary and benefits information

### Pre-configured Scenarios

The system includes intentional policy conflicts for demonstration:

- Conflicting annual leave days between documents
- Missing clauses detection
- Ambiguous language identification

## 🔧 Technical Architecture

### Core Components

1. **Document Processing**

   - PDF text extraction using PyPDF2
   - Simulated embeddings for document indexing
   - In-memory vector storage for quick retrieval

2. **AI Question Answering**

   - Document retrieval based on query relevance

   # THE HR - HR Intelligent Assistant 👥

   An AI-driven HR policy assistant that helps employees find policy answers and lets HR teams manage, analyze, and improve HR documents through a simple Streamlit UI.

   **Repository:** `THE_HR-HR_Intelligent_Assistant`

   **Quick summary:** Upload HR documents (PDF/TXT), ask natural‑language questions, analyze policies for contradictions or missing clauses, and export policy health reports.

   **Status:** Demo / Hackathon-ready (local use)

   **Admin demo credentials:** Username: `hradmin` Password: `hrpass123`

   **Main app file:** `hr_assistant_v2.py`

   **Environment variables:**

   - `GROQ_API_KEY` (optional) — configure in a `.env` file for Groq/LLM integration.

   **Supported uploads:** PDF (requires `PyPDF2`) and plain text files.

   **Key files:**

   - `hr_assistant_v2.py` — Streamlit app (employee portal + admin dashboard)
   - `install_and_run.py` — helper script to install dependencies and run the app
   - `requirements.txt` — pinned Python dependencies
   - `hr_documents.json`, `chat_history.json` — local persistence files (created on first run)

   **CI:** A GitHub Actions workflow `/.github/workflows/ci.yml` is included to run `flake8` and `pytest` on pushes and PRs.

   **License:** MIT

   ***

   ## Getting started (recommended)

   1. Clone the repo:

   ```powershell
   git clone https://github.com/Uzair-Said/THE_HR-HR_Intelligent_Assistant.git
   cd THE_HR-HR_Intelligent_Assistant
   ```

   2. Create and activate a virtual environment (recommended):

   ```powershell
   # Windows (PowerShell)
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   # macOS / Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   3. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

   4. (Optional) Install dependencies and start via helper script:

   ```powershell
   python install_and_run.py
   ```

   5. Run the app:

   ```powershell
   streamlit run hr_assistant_v2.py
   ```

   Open `http://localhost:8501` in your browser.

   ***

   ## How to use

   - **Employees:** Use the **Employee Portal** tab to ask questions in plain English. Sample questions are provided.
   - **HR Admins:** Use the **Admin Dashboard** tab and login with the demo credentials. Upload documents, load sample documents, analyze policies, and export JSON reports.

   ## Sample data

   - The app ships with sample documents (Employee Handbook, IT Security Policy, Compensation & Benefits Guide) that demonstrate: conflicting clauses, missing sections, and ambiguous wording.

   ## Notes & development details

   - The app uses simulated embeddings and simple pattern-based analysis for demonstrations. It is not a production LLM integration out of the box.
   - To enable full PDF extraction install `PyPDF2` (already included in `requirements.txt`). If missing, the app still runs with a fallback message.
   - To integrate a real LLM, set `GROQ_API_KEY` in a `.env` file and add LLM call logic where appropriate.

   ## Persistence

   - Documents and chat history are stored locally in `hr_documents.json` and `chat_history.json`. For production, replace with a proper database.

   ## CI

   - A basic GitHub Actions workflow is included at `/.github/workflows/ci.yml` to run linting and tests.

   ***

   If you'd like, I can also:

   - add a short development section with contribution guidelines,
   - add a `CODE_OF_CONDUCT.md` and `CONTRIBUTING.md`, or
   - create a GitHub release/tag for this initial version.

   If you'd like any of those, tell me which item to do next.

   **Built with ❤️ — contact:** `uzair.said@awkum.edu.pk`
