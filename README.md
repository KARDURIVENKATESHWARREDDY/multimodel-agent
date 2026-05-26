# Multimodal Document Analyzer

A lightweight Streamlit app that can:

- Upload and extract text from **images, PDFs, DOCX, and videos**
- Run OCR-based extraction (with PDF fallback OCR for scanned pages)
- Summarize extracted content
- Answer questions grounded in uploaded documents with source citations
- Analyze resume-to-job-description fit (strengths, gaps, suggestions)
- Export summaries and Q&A outputs to **TXT, PDF, DOCX**

---

## Features

### 1) Document Analyzer Mode

- Multi-file upload: image, PDF, DOCX, video
- Text extraction pipeline:
  - Image OCR via Tesseract
  - PDF text extraction via `pdfplumber` + OCR fallback for scanned pages
  - DOCX paragraph extraction
  - Video frame OCR at configurable intervals
- Summarization over extracted content
- Grounded Q&A with source + chunk citations

### 2) Resume Analyzer Mode

- Upload resume (PDF/DOCX/image)
- Paste job description
- Receive:
  - Match score with rationale
  - Strengths
  - Gaps/missing requirements
  - Suggestions to improve alignment

### 3) Export

- Download summaries and answers as:
  - `.txt`
  - `.pdf`
  - `.docx`

### 4) LLM Provider Support

Configured via environment variables:

- OpenAI
- Google (Gemini)
- Groq
- Anthropic

Optional tracing is enabled automatically if LangSmith key is provided.

---

## Project Structure

- `app.py` – Streamlit app and all core logic
- `requirements.txt` – Python dependencies
- `.env.example` – Environment configuration template
- `Dockerfile` – Container build for local/Docker deployments
- `docker-compose.yml` – One-command Docker run

---

## Setup (Local)

## 1) Clone and create virtual environment

```bash
python -m venv .venv
```

Activate:

- **Windows (cmd):**
  ```bash
  .venv\Scripts\activate
  ```
- **PowerShell:**
  ```bash
  .venv\Scripts\Activate.ps1
  ```

## 2) Install Python packages

```bash
pip install -r requirements.txt
```

## 3) Install system dependencies

### Tesseract OCR (required)

Install Tesseract and ensure it is available on PATH.

- **Windows:**
  - Install from UB Mannheim build (popular Windows distribution) or official package manager routes.
  - Typical binary path:
    `C:\Program Files\Tesseract-OCR\tesseract.exe`
  - If not on PATH, set `TESSERACT_CMD` in `.env`.

- **Ubuntu/Debian:**
  ```bash
  sudo apt-get update
  sudo apt-get install -y tesseract-ocr tesseract-ocr-eng
  ```

- **macOS (Homebrew):**
  ```bash
  brew install tesseract
  ```

### FFmpeg (recommended for robust video handling)

- **Windows:** install via winget/choco or official binaries and add to PATH
- **Ubuntu/Debian:**
  ```bash
  sudo apt-get install -y ffmpeg
  ```
- **macOS:**
  ```bash
  brew install ffmpeg
  ```

## 4) Configure environment

Copy the template and edit:

```bash
copy .env.example .env
```

Set at least:

- `LLM_PROVIDER` = `openai` / `google` / `groq` / `anthropic`
- Corresponding API key + model

Optional:

- `LANGSMITH_API_KEY` + `LANGSMITH_TRACING=true`
- `TESSERACT_CMD` (if needed)
- `OCR_LANGUAGE` (default `eng`)
- `VIDEO_FRAME_INTERVAL_SECONDS` (default `2`)

## 5) Run app

```bash
streamlit run app.py
```

Open: `http://localhost:8501`

---

## Setup (Docker)

## Option A: Docker Compose

1. Create `.env` from `.env.example`
2. Run:

```bash
docker compose up --build
```

## Option B: Docker CLI

```bash
docker build -t multimodal-document-analyzer .
docker run --rm -p 8501:8501 --env-file .env multimodal-document-analyzer
```

Open: `http://localhost:8501`

---

## Notes on Grounding and Trust

- Q&A uses retrieval over document chunks and includes source references.
- If answer context is missing, the assistant is instructed to say: **"Not found in uploaded documents"**.
- OCR quality depends on input quality (resolution, blur, language, layout).

---

## Troubleshooting

- **Tesseract not found**
  - Set `TESSERACT_CMD` in `.env`, for example:
    `TESSERACT_CMD=C:\\Program Files\\Tesseract-OCR\\tesseract.exe`

- **No text from scanned PDF/image**
  - Increase source resolution before upload.
  - Use high-contrast images.

- **Model auth/config errors**
  - Confirm provider-specific API key and model env vars.
  - Ensure `LLM_PROVIDER` matches configured key.

# multimodel-agent 
