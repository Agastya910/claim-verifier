from datetime import date
from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field

class Company(BaseModel):
    ticker: str
    name: str
    cik: str

class TranscriptSegment(BaseModel):
    speaker: str
    role: str
    text: str

class Transcript(BaseModel):
    ticker: str
    year: int
    quarter: int
    date: date
    segments: List[TranscriptSegment]

class Claim(BaseModel):
    id: str
    ticker: str
    quarter: int
    year: int
    speaker: str
    metric: str
    value: float
    unit: str
    period: str
    is_gaap: bool
    is_forward_looking: bool
    hedging_language: str
    raw_text: str
    extraction_method: str
    confidence: float
    context: str

class Verdict(BaseModel):
    claim_id: str
    verdict: Literal["VERIFIED", "APPROXIMATELY_TRUE", "FALSE", "MISLEADING", "UNVERIFIABLE"]
    actual_value: Optional[float] = None
    claimed_value: float
    difference: Optional[float] = None
    explanation: str
    misleading_flags: List[str] = Field(default_factory=list)
    confidence: float
    data_sources: List[str] = Field(default_factory=list)
    evidence: List[str] = Field(default_factory=list)

class VerificationResult(BaseModel):
    company: str
    quarter: str
    claims: List[Claim]
    verdicts: List[Verdict]
    summary_stats: dict
