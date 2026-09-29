"""
workflow.py
Assembles the graph: fan-out into parallel extraction, fan-in to scoring,
a conditional branch for borderline scores, then feedback generation.

Compiled once at import time and reused across every request — the
graph's structure never changes between requests, only the input state.
"""

from langgraph.graph import StateGraph, START, END

from .state import GraphState
from .nodes import (
    extract_jd_requirements,
    extract_resume_skills,
    compare_and_score,
    deep_dive_analysis,
    generate_feedback,
)

#decision makeing router , whether need improvement or not
def route_after_scoring(state: GraphState) -> str:
    """Borderline scores get a second, more careful pass; clear cases skip it."""
    score = state["match_score"] # match_score comes from comapre_and_score node .
    if 40 <= score <= 70:
        return "deep_dive_analysis"
    return "generate_feedback"


def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("extract_jd_requirements", extract_jd_requirements)
    workflow.add_node("extract_resume_skills", extract_resume_skills)
    workflow.add_node("compare_and_score", compare_and_score)
    workflow.add_node("deep_dive_analysis", deep_dive_analysis)
    workflow.add_node("generate_feedback", generate_feedback)

    # fan-out: both extraction nodes run independently off START
    workflow.add_edge(START, "extract_jd_requirements")
    workflow.add_edge(START, "extract_resume_skills")

    # fan-in: compare_and_score waits for both to finish
    workflow.add_edge("extract_jd_requirements", "compare_and_score")
    workflow.add_edge("extract_resume_skills", "compare_and_score")

    # conditional branch
    workflow.add_conditional_edges(
        "compare_and_score",
        route_after_scoring,
        {
            "deep_dive_analysis": "deep_dive_analysis",
            "generate_feedback": "generate_feedback",
        },
    )
    workflow.add_edge("deep_dive_analysis", "generate_feedback")
    workflow.add_edge("generate_feedback", END)

    return workflow.compile()


# compiled once, module-level — main.py imports resume_graph or analyze_resume
resume_graph = build_graph()


def analyze_resume(jd_text: str, resume_text: str) -> dict:
    """Public entry point called from main.py after file extraction."""
    final_state = resume_graph.invoke({"jd_text": jd_text, "resume_text": resume_text})

    matched = final_state["matched_keywords"]
    missing = final_state["missing_keywords"]
    total = len(matched) + len(missing)
    matched_percentage = round((len(matched) / total) * 100, 1) if total > 0 else 0.0

    return {
        "match_score": final_state["match_score"],
        "matched_keywords": matched,
        "matched_percentage": matched_percentage,
        "missing_keywords": missing,
        "improvement_areas": final_state["improvement_areas"],
        "summary": final_state["summary"],
    }