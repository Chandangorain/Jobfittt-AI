"""
main.py
FastAPI app: routes only. All real work lives elsewhere:
  extraction.py -> file to text
  graph/        -> LLM analysis
  database.py   -> MongoDB persistence

Run from inside the backend/ folder:
    uvicorn main:app --reload
"""

# Load .env BEFORE importing graph: nodes.py builds the OpenAI client at
# import time and needs OPENAI_API_KEY to already be in the environment.
from dotenv import load_dotenv

load_dotenv()

import logging

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from starlette.concurrency import run_in_threadpool

from database import get_analysis, get_history, ping, save_analysis
from extraction import extract_text_from_upload
from graph import analyze_resume
from schemas import AnalysisRecord, HistoryResponse

#fast api et up
logger = logging.getLogger("jobfit")

app = FastAPI(title="JobFit-AI", version="1.0.0")


@app.get("/health")
def health():
    """Confirms the API is up and MongoDB is reachable."""
    try:
        ping()
    except Exception:
        raise HTTPException(status_code=503, detail="Database unavailable.")
    return {"status": "ok"}

#POST /analyze
@app.post("/analyze", response_model=AnalysisRecord)
async def analyze(
    resume: UploadFile = File(..., description="Resume: PDF, DOCX or TXT"),
    job_description: UploadFile = File(..., description="Job description: PDF, DOCX or TXT"),
):
    # 1. Convert file -> text (bad input becomes a 400 with a readable message)
    try:
        resume_text = await extract_text_from_upload(resume)
        jd_text = await extract_text_from_upload(job_description)   #extract_text_from_upload comes from extraction.py
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))   #HTTP 400 Bad Request

    # 2. run the LangGraph workflow (blocking LLM calls -> worker thread)
    try:
        result = await run_in_threadpool(analyze_resume, jd_text, resume_text)
    except Exception:
        logger.exception("Graph analysis failed")
        raise HTTPException(
            status_code=502,
            detail="The analysis service failed. Please try again in a moment.",
        )

    # 3. persist and return the saved record
    try:
        record = await run_in_threadpool(
            save_analysis, result, resume.filename, job_description.filename
        )
    except Exception:
        logger.exception("Saving analysis failed")
        raise HTTPException(status_code=503, detail="Could not save the analysis.")

    return record


@app.get("/history", response_model=HistoryResponse)
def history(limit: int = Query(20, ge=1, le=100)):
    """Most recent analyses first."""
    items = get_history(limit)
    return {"count": len(items), "items": items}


@app.get("/history/{analysis_id}", response_model=AnalysisRecord)
def history_item(analysis_id: str):
    record = get_analysis(analysis_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    return record