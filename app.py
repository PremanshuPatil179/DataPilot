"""
app.py - DataPilot AI — Main Streamlit Application

Entry point for the DataPilot AI data cleaning pipeline.
Orchestrates three AI agents:
  1. InspectorAgent  — data quality analysis
  2. CleanerAgent    — LLM-driven cleaning via Gemini
  3. ReporterAgent   — transformation summary report

Run with:
    streamlit run app.py
"""

import os
import time
import logging
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

from agents.gemini_utils import resolve_gemini_api_key
from database import save_dataframe, list_tables, load_table, get_table_info
from pipeline import generate_cleaning_report, run_cleaning, run_inspection
from utils import dataframe_to_csv_bytes, format_number, load_uploaded_tabular_file

logger = logging.getLogger(__name__)

# ── Load environment variables ─────────────────────────────────────────────────
load_dotenv()

# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DataPilot AI",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Background ── */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
        color: #e2e8f0;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.04);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* ── Hero header ── */
    .hero-header {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem;
        background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(168,85,247,0.1) 100%);
        border-radius: 20px;
        border: 1px solid rgba(99,102,241,0.3);
        margin-bottom: 2rem;
    }
    .hero-header h1 {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #818cf8 0%, #c084fc 50%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .hero-header p {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }

    /* ── Metric cards ── */
    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        transition: border-color 0.2s;
    }
    .metric-card:hover { border-color: rgba(99,102,241,0.5); }
    .metric-card .label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #64748b;
        margin-bottom: 0.4rem;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #e2e8f0;
    }
    .metric-card .value.green  { color: #34d399; }
    .metric-card .value.red    { color: #f87171; }
    .metric-card .value.yellow { color: #fbbf24; }
    .metric-card .value.blue   { color: #60a5fa; }

    /* ── Agent section headers ── */
    .agent-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 1rem 1.5rem;
        border-radius: 14px;
        margin: 1.5rem 0 1rem;
        font-weight: 600;
        font-size: 1.15rem;
        border: 1px solid;
    }
    .agent-inspector {
        background: rgba(59,130,246,0.12);
        border-color: rgba(59,130,246,0.35);
        color: #93c5fd;
    }
    .agent-cleaner {
        background: rgba(168,85,247,0.12);
        border-color: rgba(168,85,247,0.35);
        color: #d8b4fe;
    }
    .agent-reporter {
        background: rgba(52,211,153,0.12);
        border-color: rgba(52,211,153,0.35);
        color: #6ee7b7;
    }

    /* ── Issue tags ── */
    .issue-tag {
        display: inline-block;
        background: rgba(251,191,36,0.12);
        border: 1px solid rgba(251,191,36,0.3);
        color: #fde68a;
        border-radius: 8px;
        padding: 0.35rem 0.8rem;
        font-size: 0.85rem;
        margin: 0.25rem;
    }

    /* ── Tables ── */
    .stDataFrame { border-radius: 12px; overflow: hidden; }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.6rem 2rem;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.88; }

    /* ── Download button ── */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
    }

    /* ── Expanders ── */
    [data-testid="stExpander"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
    }

    /* ── Step badges ── */
    .step-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 28px;
        height: 28px;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        border-radius: 50%;
        font-size: 0.75rem;
        font-weight: 700;
        color: white;
        margin-right: 0.5rem;
    }

    /* ── Status pill ── */
    .status-pill {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .status-pill.ok  { background: rgba(52,211,153,0.15); color: #34d399; border: 1px solid rgba(52,211,153,0.4); }
    .status-pill.err { background: rgba(248,113,113,0.15); color: #f87171; border: 1px solid rgba(248,113,113,0.4); }

    /* ── Scrollable table area ── */
    .scroll-table { max-height: 400px; overflow-y: auto; border-radius: 10px; }

    /* Tab styling */
    [data-testid="stTab"] { font-weight: 500; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def metric_card(label: str, value: str, color: str = "") -> str:
    return (
        f'<div class="metric-card">'
        f'<div class="label">{label}</div>'
        f'<div class="value {color}">{value}</div>'
        f"</div>"
    )


def agent_header(icon: str, title: str, cls: str) -> None:
    st.markdown(
        f'<div class="agent-header {cls}">{icon} {title}</div>',
        unsafe_allow_html=True,
    )


def show_issue_tags(issues: list[str]) -> None:
    tags_html = "".join(f'<span class="issue-tag">{i}</span>' for i in issues)
    st.markdown(tags_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state bootstrap
# ─────────────────────────────────────────────────────────────────────────────

for key in ["inspection_report", "cleaning_result", "cleaning_report",
            "original_df", "cleaned_df", "pipeline_done"]:
    if key not in st.session_state:
        st.session_state[key] = None

if "pipeline_done" not in st.session_state:
    st.session_state.pipeline_done = False

# Pre-populate API key from env var on first load only
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = resolve_gemini_api_key()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧭 DataPilot AI")
    st.markdown("---")

    st.markdown("### ⚙️ Configuration")
    st.text_input(
        "Gemini API Key",
        type="password",
        help="Get your key from https://aistudio.google.com/",
        placeholder="Paste your AIza... key here",
        key="gemini_api_key",
    )

    # Save to .env file button
    if st.button("💾 Save Key to .env", use_container_width=True):
        _k = st.session_state.gemini_api_key or ""
        if _k:
            env_path = Path(__file__).parent / ".env"
            env_path.write_text(f"GEMINI_API_KEY={_k}\n", encoding="utf-8")
            load_dotenv(override=True)          # reload immediately
            st.success("✅ Key saved! It will auto-load on next start.")
        else:
            st.error("Paste your key first.")

    # Show status
    _current_key = resolve_gemini_api_key(st.session_state.get("gemini_api_key", ""))
    if _current_key:
        st.markdown(
            '<p style="color:#34d399;font-size:0.82rem">✅ API key is set — Gemini will be used</p>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<p style="color:#fbbf24;font-size:0.82rem">⚠️ No key — rule-based fallback</p>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### 🗄️ Saved Datasets")
    tables = list_tables()
    if tables:
        selected_table = st.selectbox("View stored table:", tables)
        if st.button("📂 Load Table Preview"):
            preview_df = load_table(selected_table)
            if preview_df is not None:
                st.dataframe(preview_df.head(20), use_container_width=True)
                info = get_table_info(selected_table)
                st.caption(f"{info['row_count']:,} rows × {info['column_count']} cols")
    else:
        st.caption("No datasets saved yet.")

    st.markdown("---")
    st.markdown(
        "<div style='color:#475569;font-size:0.78rem;'>"
        "DataPilot AI · Built with Streamlit & Gemini"
        "</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main content
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="hero-header">
        <h1>🧭 DataPilot AI</h1>
        <p>Intelligent Data Cleaning · Powered by Google Gemini</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── File uploader ──────────────────────────────────────────────────────────────
st.markdown("### 📂 Upload Your Dataset")
uploaded_file = st.file_uploader(
    "Drop a CSV or Excel file here",
    type=["csv", "xls", "xlsx"],
    help="Auto-detects CSV encoding and Excel workbooks, even if the file extension is wrong.",
    label_visibility="collapsed",
)

if uploaded_file is not None:
    # Load DataFrame
    try:
        df_raw, load_info = load_uploaded_tabular_file(uploaded_file)
        st.session_state.original_df = df_raw
    except Exception as exc:
        st.error(f"❌ Could not read uploaded file: {exc}")
        st.stop()

    if load_info.get("source_format") == "csv":
        encoding_used = load_info.get("encoding") or "unknown"
        logger.info("CSV load completed using encoding: %s", encoding_used)
        st.caption(f"Detected encoding: {encoding_used}")
    else:
        logger.info("Uploaded file detected as %s and loaded with read_excel", load_info.get("source_format"))
        st.caption(f"Detected Excel workbook ({load_info.get('source_format')})")

    # ── Overview metrics ───────────────────────────────────────────────────
    total_cells = df_raw.shape[0] * df_raw.shape[1]
    missing_cells = int(df_raw.isna().sum().sum())
    dup_rows = int(df_raw.duplicated().sum())
    num_cols = df_raw.select_dtypes(include="number").shape[1]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            metric_card("Total Rows", format_number(df_raw.shape[0]), "blue"),
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            metric_card("Total Columns", str(df_raw.shape[1]), "blue"),
            unsafe_allow_html=True,
        )
    with col3:
        color = "red" if missing_cells > 0 else "green"
        st.markdown(
            metric_card("Missing Cells", format_number(missing_cells), color),
            unsafe_allow_html=True,
        )
    with col4:
        color = "yellow" if dup_rows > 0 else "green"
        st.markdown(
            metric_card("Duplicate Rows", format_number(dup_rows), color),
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Raw data preview ───────────────────────────────────────────────────
    with st.expander("🔍 Raw Data Preview (first 100 rows)", expanded=False):
        st.dataframe(df_raw.head(100), use_container_width=True)

    # ── Run pipeline button ────────────────────────────────────────────────
    st.markdown("---")

    # Re-read api_key at render time (always fresh from session_state + env)
    api_key = resolve_gemini_api_key(st.session_state.get("gemini_api_key", ""))

    run_btn = st.button("🚀 Run DataPilot Pipeline", use_container_width=True)

    if run_btn:
        # Re-read one more time at click time
        api_key = resolve_gemini_api_key(st.session_state.get("gemini_api_key", ""))

        if not api_key:
            st.info(
                "💡 No Gemini API key provided — running with smart rule-based cleaning. "
                "Enter your key in the sidebar for LLM-powered analysis."
            )

        # ══ STEP 1: Inspector Agent ════════════════════════════════════════
        agent_header("🔎", "Inspector Agent — Analysing Dataset Quality", "agent-inspector")
        with st.spinner("Inspector Agent is scanning your dataset…"):
            try:
                report = run_inspection(df_raw)
                st.session_state.inspection_report = report
                time.sleep(0.3)
            except Exception as exc:
                st.error(f"❌ Inspector Agent failed: {exc}")
                import traceback
                st.code(traceback.format_exc(), language="python")
                st.stop()

        # Issues
        st.markdown("**Issues Detected:**")
        show_issue_tags(report.issues)

        # Column summary table
        with st.expander("📋 Column-by-Column Summary", expanded=True):
            col_df = pd.DataFrame(report.column_summary)
            st.dataframe(col_df, use_container_width=True, hide_index=True)

        # Dtype issues
        if report.dtype_issues:
            with st.expander("🔧 Data Type Mismatches"):
                st.dataframe(
                    pd.DataFrame(report.dtype_issues),
                    use_container_width=True,
                    hide_index=True,
                )

        # Outliers
        if report.outlier_info:
            with st.expander("📊 Outlier Summary (IQR Method)"):
                st.dataframe(
                    pd.DataFrame(report.outlier_info),
                    use_container_width=True,
                    hide_index=True,
                )

        # Numeric stats
        if report.numeric_stats is not None:
            with st.expander("📈 Numeric Statistics"):
                st.dataframe(report.numeric_stats, use_container_width=True)

        # Missing value heatmap
        if missing_cells > 0:
            with st.expander("🗺️ Missing Value Heatmap"):
                missing_per_col = df_raw.isna().sum().reset_index()
                missing_per_col.columns = ["Column", "Missing Count"]
                missing_per_col = missing_per_col[missing_per_col["Missing Count"] > 0]
                fig = px.bar(
                    missing_per_col,
                    x="Column",
                    y="Missing Count",
                    color="Missing Count",
                    color_continuous_scale="Purples",
                    template="plotly_dark",
                    title="Missing Values per Column",
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e2e8f0",
                )
                st.plotly_chart(fig, use_container_width=True)

        # ══ STEP 2: Cleaner Agent ══════════════════════════════════════════
        agent_header("🧹", "Cleaner Agent — LLM-Driven Data Cleaning", "agent-cleaner")
        with st.spinner("Cleaner Agent is reasoning about the best fixes…"):
            try:
                cleaning_result = run_cleaning(df_raw, report.issues, api_key=api_key or None)
                st.session_state.cleaning_result = cleaning_result
                st.session_state.cleaned_df = cleaning_result.cleaned_df
                time.sleep(0.2)
            except Exception as exc:
                st.error(f"❌ Cleaner Agent failed: {exc}")
                import traceback
                st.code(traceback.format_exc(), language="python")
                st.stop()

        if cleaning_result.success:
            st.markdown(
                '<span class="status-pill ok">✅ Cleaning Complete</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="status-pill err">⚠️ Cleaning had errors</span>',
                unsafe_allow_html=True,
            )

        # Transformation log
        if cleaning_result.transformations:
            with st.expander("🔧 Transformation Log", expanded=True):
                log_df = pd.DataFrame(cleaning_result.transformations)
                st.dataframe(log_df, use_container_width=True, hide_index=True)
        else:
            st.info("No transformations applied — dataset was already clean.")

        # Errors
        if cleaning_result.errors:
            with st.expander("⚠️ Agent Warnings"):
                for e in cleaning_result.errors:
                    st.warning(e)

        # LLM plan
        if cleaning_result.llm_plan:
            with st.expander("🤖 LLM Cleaning Plan (raw)"):
                st.code(cleaning_result.llm_plan, language="json")

        # Cleaned data preview
        if cleaning_result.cleaned_df is not None:
            with st.expander("✨ Cleaned Dataset Preview", expanded=True):
                st.dataframe(
                    cleaning_result.cleaned_df.head(100),
                    use_container_width=True,
                )

        # ══ STEP 3: Reporter Agent ═════════════════════════════════════════
        agent_header("📋", "Reporter Agent — Generating Summary Report", "agent-reporter")
        with st.spinner("Reporter Agent is compiling the final report…"):
            try:
                cleaning_report = generate_cleaning_report(
                    original_df=df_raw,
                    cleaning_result=cleaning_result,
                    api_key=api_key or None,
                )
                st.session_state.cleaning_report = cleaning_report
                time.sleep(0.2)
            except Exception as exc:
                st.error(f"❌ Reporter Agent failed: {exc}")
                import traceback
                st.code(traceback.format_exc(), language="python")
                st.stop()

        # Before / after comparison metrics
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                metric_card("Original Rows", format_number(cleaning_report.original_shape[0])),
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                metric_card("Cleaned Rows", format_number(cleaning_report.cleaned_shape[0]), "green"),
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                metric_card("Rows Removed", format_number(cleaning_report.rows_removed), "red"),
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                metric_card("Cols Removed", str(cleaning_report.columns_removed), "yellow"),
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Narrative
        if cleaning_report.narrative:
            st.info(f"🤖 **AI Summary:** {cleaning_report.narrative}")

        if getattr(cleaning_report, "warnings", None):
            with st.expander("⚠️ Report Warnings"):
                for warning in cleaning_report.warnings:
                    st.warning(warning)

        # Full markdown report
        with st.expander("📄 Full Markdown Report"):
            st.markdown(cleaning_report.markdown_report)

        # ══ Save to SQLite ═════════════════════════════════════════════════
        st.markdown("---")
        st.markdown("### 🗄️ Save & Download")

        table_name = Path(uploaded_file.name).stem
        db_msg = save_dataframe(cleaning_result.cleaned_df, table_name)
        st.success(db_msg)

        # Download cleaned CSV
        csv_bytes = dataframe_to_csv_bytes(cleaning_result.cleaned_df)
        st.download_button(
            label="⬇️ Download Cleaned CSV",
            data=csv_bytes,
            file_name=f"{table_name}_cleaned.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # Download full report
        st.download_button(
            label="📄 Download Cleaning Report (Markdown)",
            data=cleaning_report.markdown_report.encode("utf-8"),
            file_name=f"{table_name}_report.md",
            mime="text/markdown",
            use_container_width=True,
        )

        st.session_state.pipeline_done = True

    # ── Visualisations tab (shown after pipeline) ──────────────────────────
    if st.session_state.pipeline_done and st.session_state.cleaned_df is not None:
        st.markdown("---")
        st.markdown("### 📊 Data Visualisations")
        tab1, tab2, tab3 = st.tabs(["Distributions", "Correlations", "Before vs After"])

        cleaned = st.session_state.cleaned_df
        original = st.session_state.original_df

        with tab1:
            num_cols_list = cleaned.select_dtypes(include="number").columns.tolist()
            if num_cols_list:
                chosen = st.selectbox("Select a numeric column:", num_cols_list)
                fig = px.histogram(
                    cleaned,
                    x=chosen,
                    nbins=40,
                    template="plotly_dark",
                    color_discrete_sequence=["#818cf8"],
                    title=f"Distribution of {chosen} (cleaned)",
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e2e8f0",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No numeric columns found for distribution chart.")

        with tab2:
            num_df = cleaned.select_dtypes(include="number")
            if num_df.shape[1] >= 2:
                corr = num_df.corr()
                fig = px.imshow(
                    corr,
                    color_continuous_scale="RdBu",
                    zmin=-1, zmax=1,
                    template="plotly_dark",
                    title="Correlation Heatmap (cleaned data)",
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#e2e8f0",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Need at least 2 numeric columns for a correlation heatmap.")

        with tab3:
            comparison = pd.DataFrame({
                "Metric": ["Rows", "Columns", "Missing Cells", "Duplicate Rows"],
                "Before": [
                    original.shape[0],
                    original.shape[1],
                    int(original.isna().sum().sum()),
                    int(original.duplicated().sum()),
                ],
                "After": [
                    cleaned.shape[0],
                    cleaned.shape[1],
                    int(cleaned.isna().sum().sum()),
                    int(cleaned.duplicated().sum()),
                ],
            })
            fig = go.Figure(data=[
                go.Bar(name="Before", x=comparison["Metric"], y=comparison["Before"],
                       marker_color="#f87171"),
                go.Bar(name="After",  x=comparison["Metric"], y=comparison["After"],
                       marker_color="#34d399"),
            ])
            fig.update_layout(
                barmode="group",
                template="plotly_dark",
                title="Before vs After Cleaning",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
            )
            st.plotly_chart(fig, use_container_width=True)

else:
    # ── Landing state ─────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="
            text-align:center;
            padding:4rem 2rem;
            color:#475569;
        ">
            <div style="font-size:5rem;">📂</div>
            <h3 style="color:#64748b; font-weight:500;">Upload a CSV file to get started</h3>
            <p>DataPilot AI will inspect, clean, and report on your dataset using<br>
            three specialised AI agents powered by Google Gemini.</p>
            <br>
            <div style="display:flex;justify-content:center;gap:2rem;flex-wrap:wrap;">
                <div style="
                    background:rgba(99,102,241,0.1);
                    border:1px solid rgba(99,102,241,0.3);
                    border-radius:12px;padding:1rem 1.5rem;max-width:180px;">
                    <div style="font-size:1.8rem;">🔎</div>
                    <strong style="color:#818cf8;">Inspector</strong>
                    <p style="font-size:0.82rem;margin:0;color:#64748b;">Detects issues in your data</p>
                </div>
                <div style="
                    background:rgba(168,85,247,0.1);
                    border:1px solid rgba(168,85,247,0.3);
                    border-radius:12px;padding:1rem 1.5rem;max-width:180px;">
                    <div style="font-size:1.8rem;">🧹</div>
                    <strong style="color:#c084fc;">Cleaner</strong>
                    <p style="font-size:0.82rem;margin:0;color:#64748b;">LLM-driven cleaning via Gemini</p>
                </div>
                <div style="
                    background:rgba(52,211,153,0.1);
                    border:1px solid rgba(52,211,153,0.3);
                    border-radius:12px;padding:1rem 1.5rem;max-width:180px;">
                    <div style="font-size:1.8rem;">📋</div>
                    <strong style="color:#6ee7b7;">Reporter</strong>
                    <p style="font-size:0.82rem;margin:0;color:#64748b;">Generates transformation report</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
