#nodes.py only defines what each node does; 


"""
nodes.py
One function per graph node. Each takes GraphState in, returns a dict
of updates. No knowledge of FastAPI, MongoDB, or HTTP — pure functions,
easy to unit test in isolation.
"""

from typing import List

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from .state import GraphState

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ── Structured output schemas (one per LLM call) ────────────────────

class JDRequirements(BaseModel):
    requirements: List[str] = Field(description="Key required skills/qualifications from the JD")
    seniority_level: str = Field(description="e.g. Junior, Mid, Senior, Lead")


class ResumeSkills(BaseModel):
    skills: List[str] = Field(description="Skills/technologies mentioned in the resume")
    years_experience: float = Field(description="Estimated total years of relevant experience")


class ScoreResult(BaseModel):
    match_score: int = Field(ge=0, le=100)
    matched_keywords: List[str]
    missing_keywords: List[str]


class ImprovementArea(BaseModel):
    area: str = Field(description="Short name of the area needing improvement")
    suggestion: str = Field(description="Specific, actionable suggestion for this area")


class FeedbackResult(BaseModel):
    improvement_areas: List[ImprovementArea]
    summary: str = Field(description="2-3 sentence overall assessment, kept concise")


# ── Node 1: extract JD requirements (runs parallel with Node 2) ─────

def extract_jd_requirements(state: GraphState) -> dict:
    structured_llm = llm.with_structured_output(JDRequirements)
    result = structured_llm.invoke(
        f"Extract the key requirements and seniority level from this job "
        f"description:\n\n{state['jd_text']}"
    )
    return {"jd_requirements": result.requirements, "seniority_level": result.seniority_level}


# ── Node 2: extract resume skills (runs parallel with Node 1) ───────

def extract_resume_skills(state: GraphState) -> dict:
    structured_llm = llm.with_structured_output(ResumeSkills)
    result = structured_llm.invoke(
        f"Extract skills and estimated years of relevant experience from "
        f"this resume:\n\n{state['resume_text']}"
    )
    return {"resume_skills": result.skills, "years_experience": result.years_experience}


# ── Node 3: compare + score (fan-in — waits for both above) ─────────

def compare_and_score(state: GraphState) -> dict:
    structured_llm = llm.with_structured_output(ScoreResult)
    result = structured_llm.invoke(
        f"Job requires ({state['seniority_level']} level): "
        f"{', '.join(state['jd_requirements'])}\n"
        f"Candidate has ({state['years_experience']} yrs experience): "
        f"{', '.join(state['resume_skills'])}\n\n"
        "Score the match 0-100, list matched keywords and missing keywords. "
        "Be precise — only count a keyword as matched if it genuinely "
        "appears in both lists."
    )
    return {
        "match_score": result.match_score,
        "matched_keywords": result.matched_keywords,
        "missing_keywords": result.missing_keywords,
    }


# ── Node 4 (conditional): deeper pass for borderline scores ─────────

def deep_dive_analysis(state: GraphState) -> dict:
    structured_llm = llm.with_structured_output(ScoreResult)
    result = structured_llm.invoke(
        f"This candidate scored {state['match_score']}/100 — a borderline "
        "case that needs closer review.\n\n"
        f"JD requirements: {', '.join(state['jd_requirements'])}\n"
        f"Candidate skills: {', '.join(state['resume_skills'])}\n"
        f"Currently matched: {', '.join(state['matched_keywords'])}\n"
        f"Currently missing: {', '.join(state['missing_keywords'])}\n\n"
        "Look for adjacent/transferable skills (e.g. Django counts partially "
        "toward FastAPI experience) and re-score more carefully. Adjust the "
        "score, matched, and missing keywords if warranted."
    )
    return {
        "match_score": result.match_score,
        "matched_keywords": result.matched_keywords,
        "missing_keywords": result.missing_keywords,
    }


# ── Node 5: generate feedback (always runs last) ────────────────────

def generate_feedback(state: GraphState) -> dict:
    structured_llm = llm.with_structured_output(FeedbackResult)
    result = structured_llm.invoke(
        f"Match score: {state['match_score']}/100\n"
        f"Matched: {', '.join(state['matched_keywords'])}\n"
        f"Missing: {', '.join(state['missing_keywords'])}\n\n"
        "Write 3-5 improvement areas, each as a short area name paired with a "
        "specific, actionable suggestion (e.g. area: 'Quantify impact', "
        "suggestion: 'Add metrics to the X project bullet, like % improvement "
        "or users affected'). Then write a summary in 2-3 sentences max — keep "
        "it short and to the point."
    )
    return {
        "improvement_areas": [ia.model_dump() for ia in result.improvement_areas],
        "summary": result.summary,
    }