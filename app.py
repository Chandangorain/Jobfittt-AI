"""
app.py
Streamlit frontend for JobFit-AI. Talks to the FastAPI backend over HTTP:
  POST /analyze  -> upload resume + job description, get the analysis
  GET  /history  -> list past analyses

Run (backend must already be running):
    streamlit run app.py
"""

import os
from datetime import datetime
from html import escape

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 120  # the graph makes several LLM calls; allow time

st.set_page_config(page_title="JobFit-AI", page_icon="🎯", layout="centered")


# ── helpers ──────────────────────────────────────────────────────────

def score_color(score: int) -> str:
    if score >= 70:
        return "#2e9e5b"  # green
    if score >= 40:
        return "#e0a100"  # amber
    return "#d64545"      # red


def chips(items, color: str) -> str:
    """Render a list of keywords as colored pills."""
    if not items:
        return "<span style='color:gray'>None</span>"
    return " ".join(
        f"<span style='display:inline-block;margin:2px 4px 2px 0;padding:3px 10px;"
        f"border-radius:12px;background:{color}22;color:{color};font-size:0.85rem;"
        f"border:1px solid {color}55'>{escape(str(i))}</span>"
        for i in items
    )


def fmt_time(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).strftime("%d %b %Y, %H:%M")
    except ValueError:
        return iso


def render_result(r: dict):
    color = score_color(r["match_score"])
    c1, c2 = st.columns(2)
    c1.markdown(
        f"<div style='text-align:center'><div style='font-size:3rem;font-weight:700;"
        f"color:{color}'>{r['match_score']}<span style='font-size:1.2rem'>/100</span></div>"
        f"<div style='color:gray'>Match score</div></div>",
        unsafe_allow_html=True,
    )
    c2.metric("Keywords matched", f"{r['matched_percentage']}%")
    st.progress(min(max(r["match_score"], 0), 100) / 100)

    st.markdown("**Summary**")
    st.write(r["summary"])

    st.markdown("**✅ Matched keywords**")
    st.markdown(chips(r["matched_keywords"], "#2e9e5b"), unsafe_allow_html=True)
    st.markdown("**❌ Missing keywords**")
    st.markdown(chips(r["missing_keywords"], "#d64545"), unsafe_allow_html=True)

    st.markdown("**🛠 Areas to improve**")
    for item in r["improvement_areas"]:
        with st.expander(item["area"], expanded=True):
            st.write(item["suggestion"])


def call_analyze(resume, jd):
    files = {
        "resume": (resume.name, resume.getvalue(), resume.type or "application/octet-stream"),
        "job_description": (jd.name, jd.getvalue(), jd.type or "application/octet-stream"),
    }
    return requests.post(f"{API_URL}/analyze", files=files, timeout=REQUEST_TIMEOUT)


# ── page ─────────────────────────────────────────────────────────────

st.title("🎯 JobFit-AI")
st.caption("Upload a resume and a job description to see how well they match.")

tab_analyze, tab_history = st.tabs(["Analyze", "History"])

with tab_analyze:
    col_a, col_b = st.columns(2)
    resume_file = col_a.file_uploader("Resume", type=["pdf", "docx", "txt"], key="resume")
    jd_file = col_b.file_uploader("Job description", type=["pdf", "docx", "txt"], key="jd")

    if st.button("Analyze", type="primary", disabled=not (resume_file and jd_file)):
        try:
            with st.spinner("Analyzing... this can take up to a minute."):
                resp = call_analyze(resume_file, jd_file)
            if resp.status_code == 200:
                st.session_state["result"] = resp.json()
            else:
                st.session_state.pop("result", None)
                try:
                    detail = resp.json().get("detail", resp.text)
                except ValueError:
                    detail = resp.text
                st.error(detail)
        except requests.exceptions.ConnectionError:
            st.error(f"Cannot reach the backend at {API_URL}. Is it running?")
        except requests.exceptions.Timeout:
            st.error("The request timed out. Please try again.")

    if "result" in st.session_state:
        st.divider()
        render_result(st.session_state["result"])

with tab_history:
    try:
        resp = requests.get(f"{API_URL}/history", params={"limit": 20}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException:
        st.warning("Could not load history. Is the backend running?")
    else:
        if data["count"] == 0:
            st.info("No analyses yet. Run one in the Analyze tab.")
        for rec in data["items"]:
            label = (
                f"{rec['match_score']}/100 · {rec.get('resume_filename') or 'resume'} "
                f"vs {rec.get('jd_filename') or 'job description'} · {fmt_time(rec['created_at'])}"
            )
            with st.expander(label):
                render_result(rec)