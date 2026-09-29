"""
state.py
Shape of the data flowing through the graph. Every node reads some of
these fields and returns a partial update to others.
"""

from typing import TypedDict, List


class GraphState(TypedDict, total=False):
    # inputs
    jd_text: str
    resume_text: str

    # from extract_jd_requirements
    jd_requirements: List[str]
    seniority_level: str

    # from extract_resume_skills
    resume_skills: List[str]
    years_experience: float

    # from compare_and_score / deep_dive_analysis
    match_score: int
    matched_keywords: List[str]
    missing_keywords: List[str]
    matched_percentage: float  # computed: matched / (matched + missing) * 100

    # from generate_feedback
    improvement_areas: List[dict]  # each: {"area": str, "suggestion": str}
    summary: str