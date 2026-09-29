#Pydantic response models
"""
schemas.py
Pydantic response models for the FastAPI routes. They mirror exactly what
graph.analyze_resume() returns, plus the metadata MongoDB adds when a
result is saved (id, timestamp, filenames).
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class ImprovementArea(BaseModel):
    area: str
    suggestion: str


class AnalysisResult(BaseModel):
    """Shape returned by graph.analyze_resume()."""
    match_score: int = Field(ge=0, le=100)
    matched_keywords: List[str]
    matched_percentage: float
    missing_keywords: List[str]
    improvement_areas: List[ImprovementArea]
    summary: str


class AnalysisRecord(AnalysisResult):
    """A saved analysis: the result plus DB metadata. Returned by /analyze and /history."""
    id: str
    created_at: datetime
    resume_filename: Optional[str] = None
    jd_filename: Optional[str] = None


class HistoryResponse(BaseModel):
    count: int
    items: List[AnalysisRecord]