from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    session_id: Optional[str] = None


class SessionSummary(BaseModel):
    history: List[str]
    preferences: Dict[str, List[str]]


class SearchResponse(BaseModel):
    session_id: Optional[str]
    results: List[dict]
    explain: List[List[str]]
    session: SessionSummary


class FeedbackRequest(BaseModel):
    session_id: Optional[str]
    doc_id: Optional[int] = None
    path: Optional[str] = None
    ext: Optional[str] = None
    owner: Optional[str] = None
    action: str


class FeedbackResponse(BaseModel):
    session_id: Optional[str]
    status: str


class ReindexRequest(BaseModel):
    force: bool = False


class ReindexResponse(BaseModel):
    status: str


class SessionResetResponse(BaseModel):
    session_id: Optional[str]
    history: List[str]
