import enum
from datetime import datetime
from typing import List, Optional

try:
    from pgvector.sqlalchemy import Vector, SPARSEVEC
    _HAS_PGVECTOR = True
except ImportError:
    # pgvector not installed (e.g. on Streamlit Cloud) — use placeholder types.
    # DocumentChunk table won't be usable, but the rest of the schema works fine.
    _HAS_PGVECTOR = False
    Vector = lambda dim: Text  # noqa: E731
    SPARSEVEC = lambda dim: Text  # noqa: E731
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass

class FinancialData(Base):
    __tablename__ = "financial_data"
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String, index=True)
    year = Column(Integer, index=True)
    quarter = Column(Integer, index=True)
    metric = Column(String, index=True)
    value = Column(Float)
    unit = Column(String)
    source = Column(String)  # "10-Q", "10-K", "finnhub"
    is_gaap = Column(Boolean, default=True)
    filing_date = Column(Date)
    created_at = Column(DateTime, server_default=func.now())

class TranscriptRecord(Base):
    __tablename__ = "transcripts"
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String, index=True)
    year = Column(Integer, index=True)
    quarter = Column(Integer, index=True)
    date = Column(Date)
    source = Column(String)  # "finnhub", "huggingface"
    full_text = Column(Text)
    segments = Column(JSON)  # list of {speaker, role, text}

class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String, index=True)
    year = Column(Integer, index=True)
    quarter = Column(Integer, index=True)
    chunk_type = Column(String)  # "financial", "transcript"
    metric_type = Column(String, nullable=True)
    source_type = Column(String)
    is_gaap = Column(Boolean, nullable=True)
    text = Column(Text)
    
    # New columns
    sequence_index = Column(Integer, index=True)
    is_analyst_question = Column(Boolean, default=False)
    
    # pgvector columns (placeholder types when pgvector is not installed)
    dense_embedding = Column(Vector(1024))      # 1024 dimensions for dense
    sparse_embedding = Column(SPARSEVEC(30522))  # 30522 for SPLADE
    
    created_at = Column(DateTime, server_default=func.now())

class ClaimRecord(Base):
    __tablename__ = "claims"
    
    id = Column(String, primary_key=True)  # Using matching ID from extracted claim
    ticker = Column(String, index=True)
    quarter = Column(Integer, index=True)
    year = Column(Integer, index=True)
    speaker = Column(String)
    metric = Column(String)
    value = Column(Float)
    unit = Column(String)
    period = Column(String)
    is_gaap = Column(Boolean)
    is_forward_looking = Column(Boolean)
    hedging_language = Column(Text)
    raw_text = Column(Text)
    extraction_method = Column(String)
    confidence = Column(Float)
    context = Column(Text)
    
    created_at = Column(DateTime, server_default=func.now())

class VerdictRecord(Base):
    __tablename__ = "verdicts"
    
    id = Column(Integer, primary_key=True)
    claim_id = Column(String, ForeignKey("claims.id"), index=True)
    verdict = Column(String)  # VERIFIED, FALSE, etc.
    actual_value = Column(Float, nullable=True)
    claimed_value = Column(Float)
    difference = Column(Float, nullable=True)
    explanation = Column(Text)
    misleading_flags = Column(JSON)  # list of strings
    confidence = Column(Float)
    data_sources = Column(JSON)  # list of strings
    evidence = Column(JSON)  # list of strings containing exact quotes
    
    created_at = Column(DateTime, server_default=func.now())
class JobRecord(Base):
    __tablename__ = "jobs"
    
    id = Column(String, primary_key=True)  # UUID
    status = Column(String)  # PENDING, RUNNING, COMPLETED, FAILED
    progress = Column(Float, default=0.0)
    message = Column(Text, nullable=True)
    result_metadata = Column(JSON, nullable=True)
    
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
