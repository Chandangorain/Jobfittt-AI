"""
database.py
MongoDB connection + CRUD for saved analyses. One collection, no users.
Every function returns plain dicts shaped like schemas.AnalysisRecord
(string `id` instead of Mongo's ObjectId), so routes can return them directly.
"""

import os
from datetime import datetime, timezone
from typing import List, Optional

from bson import ObjectId
from bson.errors import InvalidId
from dotenv import load_dotenv
from pymongo import DESCENDING, MongoClient

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "jobfit_ai")

# tz_aware=True so created_at comes back as an aware UTC datetime
client = MongoClient(MONGODB_URI, tz_aware=True, serverSelectionTimeoutMS=5000)
db = client[DB_NAME]
analyses = db["analyses"]

analyses.create_index([("created_at", DESCENDING)])


def ping() -> bool:
    """Used by a startup / health check to confirm Mongo is reachable."""
    client.admin.command("ping")
    return True


def _to_record(doc: dict) -> dict:
    """Mongo document -> API-friendly dict (ObjectId -> str id)."""
    doc = dict(doc)
    doc["id"] = str(doc.pop("_id"))
    return doc


def save_analysis(
    result: dict,
    resume_filename: Optional[str] = None,
    jd_filename: Optional[str] = None,
) -> dict:
    """Store the output of graph.analyze_resume() and return the saved record."""
    doc = {
        **result,
        "resume_filename": resume_filename,
        "jd_filename": jd_filename,
        "created_at": datetime.now(timezone.utc),
    }
    inserted = analyses.insert_one(doc)
    doc["_id"] = inserted.inserted_id
    return _to_record(doc)


def get_history(limit: int = 20) -> List[dict]:
    """Most recent analyses first."""
    cursor = analyses.find().sort("created_at", DESCENDING).limit(limit)
    return [_to_record(d) for d in cursor]


def get_analysis(analysis_id: str) -> Optional[dict]:
    """Fetch one analysis by id. Returns None if the id is invalid or not found."""
    try:
        oid = ObjectId(analysis_id)
    except (InvalidId, TypeError):
        return None
    doc = analyses.find_one({"_id": oid})
    return _to_record(doc) if doc else None