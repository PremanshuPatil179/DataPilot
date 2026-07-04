# 🧭 DataPilot AI

**Intelligent CSV data cleaning powered by Google Gemini + Streamlit**

DataPilot AI is a multi-agent data pipeline that automatically inspects, cleans, and reports on your CSV datasets using three specialised AI agents.

It now also includes a FastAPI backend that reuses the same cleaning pipeline as the Streamlit app.

---

## 🚀 Features

| Agent | Responsibility |
|-------|---------------|
| 🔎 **Inspector** | Detects missing values, duplicates, type mismatches, and outliers |
| 🧹 **Cleaner** | Asks Gemini to reason about the best fixes, then executes them |
| 📋 **Reporter** | Generates a structured markdown report with an AI narrative |

- 💾 Saves cleaned data to a local **SQLite database**
- ⬇️ Download the **cleaned CSV** and **markdown report**
- 📊 Interactive **Plotly charts**: distributions, correlations, before/after
- 🎨 Premium dark UI with glassmorphism design

---

## 📁 Project Structure

```
DataPilot/
├── app.py              # Streamlit entry point
├── api.py              # FastAPI entry point
├── database.py         # SQLite persistence layer
├── pipeline.py         # Shared inspection/cleaning/report orchestration
├── utils.py            # Shared utility functions
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
└── agents/
    ├── __init__.py
    ├── inspector.py    # InspectorAgent
    ├── cleaner.py      # CleanerAgent (LLM-powered)
    └── reporter.py     # ReporterAgent
```

---

## ⚙️ Setup

### 1. Clone / open the project

```powershell
cd DataPilot
```

### 2. Create a virtual environment (recommended)

```powershell
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Configure your Gemini API key

Copy `.env.example` to `.env` and add your key:

```
GEMINI_API_KEY=your_actual_api_key_here
```

Or paste the key directly in the sidebar of the app.

> Get a free key at https://aistudio.google.com/

### 5. Run the app

```powershell
streamlit run app.py
```

### 6. Run the FastAPI backend

```powershell
uvicorn api:app --host 0.0.0.0 --port 8000
```

For Render deployment, use the same host and let the platform supply `PORT`, for example:

```powershell
python api.py
```

or

```bash
uvicorn api:app --host 0.0.0.0 --port $PORT
```

The app falls back to port `8000` locally when `PORT` is not set.

Example direct server startup:

```powershell
python api.py
```

## FastAPI API

### `GET /`

Returns a simple health response:

```json
{
    "status": "running",
    "service": "DataPilot API"
}
```

### `POST /clean`

Accepts `multipart/form-data` with one file field named `file`.

This works directly with n8n's HTTP Request node. Configure the request as:

- Method: `POST`
- Body type: `multipart/form-data`
- Field name: `file`
- File: CSV, XLS, or XLSX

The endpoint returns the cleaned dataset as a downloadable CSV file.

---

## 🧹 Cleaning Capabilities

The Cleaner Agent (guided by Gemini) can apply:

- `drop_duplicates` — Remove duplicate rows
- `drop_column` — Remove columns with excessive missing values (>70%)
- `fill_missing` — Fill NaNs with mean / median / mode / constant
- `drop_missing_rows` — Drop rows missing a critical column
- `convert_dtype` — Parse strings as numeric, datetime, or boolean
- `strip_whitespace` — Clean string columns
- `rename_to_snake_case` — Normalise column names

If the API is unavailable, a sensible rule-based fallback is applied automatically.

The Streamlit app and FastAPI backend both use the shared pipeline helpers in `pipeline.py`, so the cleaning logic stays in one place.

---

## 📄 License

MIT
# DataPilot
