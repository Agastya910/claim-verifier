import logging
import tempfile
import os
import re
from typing import List, Dict, Any, Optional
from fastembed import SparseTextEmbedding, TextEmbedding
from sqlalchemy.orm import Session
from pgvector.sqlalchemy import Vector, SPARSEVEC, SparseVector
from sqlalchemy import insert, func

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer
from docling.document_converter import DocumentConverter

from src.db.schema import DocumentChunk, TranscriptRecord, FinancialData as FinancialDataModel
from src.models import Transcript

logger = logging.getLogger(__name__)


def split_text_preserve_sentences(text: str, hf_tokenizer, max_tokens: int = 450):
    """
    Split long text into sentence-preserving subtexts that fit under max_tokens (measured by hf_tokenizer).
    Returns list[str].
    """
    # quick sentence splitter (works decently for transcripts)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if not sentences:
        return [text]

    chunks = []
    cur = ""
    for s in sentences:
        candidate = (cur + " " + s).strip() if cur else s
        tok_len = len(hf_tokenizer.encode(candidate, add_special_tokens=False))
        if tok_len <= max_tokens:
            cur = candidate
        else:
            if cur:
                chunks.append(cur)
            # if single sentence > max_tokens, force-split by tokens (rare)
            if len(hf_tokenizer.encode(s, add_special_tokens=False)) > max_tokens:
                tokens = hf_tokenizer.encode(s, add_special_tokens=False)
                for i in range(0, len(tokens), max_tokens):
                    sub = hf_tokenizer.decode(tokens[i:i+max_tokens], skip_special_tokens=True)
                    chunks.append(sub)
                cur = ""
            else:
                cur = s
    if cur:
        chunks.append(cur)
    return chunks

# Initialize embedding models (using bge-small for memory efficiency)
sparse_model = SparseTextEmbedding("prithivida/Splade_PP_en_v1")
dense_model = TextEmbedding("BAAI/bge-large-en-v1.5")

# 1. Initialize the Hugging Face tokenizer
hf_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
hf_tokenizer.model_max_length = 100_00000

# 2. Wrap it for Docling with your specific constraints
# max_tokens: upper limit for chunks
docling_tokenizer = HuggingFaceTokenizer(
    tokenizer=hf_tokenizer,
    max_tokens=450 
)

# 3. Initialize the HybridChunker
# merge_peers: ensures structural context (headings/lists) isn't lost
hybrid_chunker = HybridChunker(
    tokenizer=docling_tokenizer,
    merge_peers=True 
)

# Usage example:
# chunks = list(hybrid_chunker.chunk(dl_doc=doc))

def chunk_financial_data(ticker: str, financials: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert financial data into text chunks: one per metric per quarter.
    financials format: { "2025Q2": { "metrics": { "revenue": 94836000000, ... }, "source": "10-Q" }, ... }
    """
    chunks = []
    
    for period, data in financials.items():
        try:
            year = int(period[:4])
            quarter = int(period[-1])
        except (ValueError, IndexError):
            logger.warning(f"Invalid period format: {period}. Skipping.")
            continue
            
        source = data.get("source", "SEC")
        metrics = data.get("metrics", {})
        
        for metric_name, value in metrics.items():
            if value is None:
                continue
            
            # Format value nicely
            if isinstance(value, (int, float)):
                if abs(value) >= 1_000_000:
                    formatted_value = f"${value:,.0f}"
                elif abs(value) < 100:  # Likely EPS or ratio
                    formatted_value = f"{value:.2f}"
                else:
                    formatted_value = f"${value:,.0f}"
            else:
                formatted_value = str(value)
            
            text = f"Company: {ticker} | Period: Q{quarter} {year} | Form: {source}\n{metric_name}: {formatted_value}"
            
            chunks.append({
                "ticker": ticker,
                "year": year,
                "quarter": quarter,
                "chunk_type": "financial",
                "metric_type": metric_name,
                "source_type": source,
                "is_gaap": True,
                "text": text,
                "sequence_index": len(chunks),
                "is_analyst_question": False
            })
                
    return chunks


def chunk_transcript_data(ticker: str, transcript: Transcript) -> List[Dict[str, Any]]:
    """
    Chunk transcript segments with speaker attribution using docling's hybrid chunker.
    Handles entire transcript as a single document to ensure consistent chunking.
    """
    chunks = []
    sequence_index = 0
    
    # 1. Flatten entire transcript into a single structured text string
    structured_text = f"Company: {ticker} | Period: Q{transcript.quarter} {transcript.year}\n\n"
    for segment in transcript.segments:
        structured_text += f"Speaker: {segment.speaker}\n{segment.text}\n\n"
    
    # 2. Create a single DoclingDocument from this full transcript text
    from docling.datamodel.document import DoclingDocument
    doc = DoclingDocument(name=f"transcript_{ticker}_{transcript.year}Q{transcript.quarter}")
    doc.add_text(label="text", text=structured_text)
    
    # 3. Run hybrid_chunker.chunk once on the full document
    docling_chunks = list(hybrid_chunker.chunk(dl_doc=doc))
    
    # 4. For each resulting chunk
    for chunk in docling_chunks:
        # Determine if this is an analyst question
        is_analyst_question = False
        if "Speaker: Analyst" in chunk.text or "analyst" in chunk.text.lower():
            is_analyst_question = True
            
        chunks.append({
            "ticker": ticker,
            "year": transcript.year,
            "quarter": transcript.quarter,
            "chunk_type": "transcript",
            "source_type": "transcript",
            "text": chunk.text,
            "sequence_index": sequence_index,
            "is_analyst_question": is_analyst_question
        })
        
        sequence_index += 1
            
    return chunks


def index_documents(chunks: List[Dict[str, Any]], db: Session, batch_size: int = 64):
    """
    For each chunk, generate SPLADE sparse embedding and BGE dense embedding.
    Insert into document_chunks table in PostgreSQL.
    Handles deduplication and bulk insertion.
    """
    if not chunks:
        logger.info("No chunks to index.")
        return

    # Deduplicate chunks based on text content
    seen_texts = set()
    unique_chunks = []
    for chunk in chunks:
        if chunk["text"] not in seen_texts:
            seen_texts.add(chunk["text"])
            unique_chunks.append(chunk)
    
    logger.info(f"Removed {len(chunks) - len(unique_chunks)} duplicate chunks.")
    chunks = unique_chunks

    # Process in batches
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]

        # Quick debug: detect any overly long chunk before embedding
        for t in texts:
            token_len = len(hf_tokenizer.encode(t))
            if token_len > 600:
                print("🚨 LONG TEXT DETECTED:", token_len)
                print(t[:500])
                break

        # Generate embeddings
        for c in batch:
            token_len = len(hf_tokenizer.encode(c["text"]))
            if token_len > 512:
                print("OVERFLOW DETECTED")
                print("Ticker:", c["ticker"])
                print("Year:", c["year"])
                print("Quarter:", c["quarter"])
                print("Chunk type:", c["chunk_type"])
                print("Token length:", token_len)
                print("Text preview:", c["text"][:500])
                break

        logger.info(f"Generating embeddings for batch of {len(batch)} chunks...")
        dense_embeddings = list(dense_model.embed(texts))
        sparse_embeddings = list(sparse_model.embed(texts))
        
        records = []
        for j, chunk_data in enumerate(batch):
            # SPLADE returns a generator of SparseEmbedding objects
            sparse_emb = sparse_embeddings[j]
            # Convert to SparseVector object with explicit dimension 30522
            sparse_dict = dict(zip(sparse_emb.indices.tolist(), sparse_emb.values.tolist()))
            sparse_vec = SparseVector(sparse_dict, 30522)
            
            # Create DocumentChunk instance (as dict for bulk_insert_mappings)
            record = {
                "ticker": chunk_data["ticker"],
                "year": chunk_data["year"],
                "quarter": chunk_data["quarter"],
                "chunk_type": chunk_data["chunk_type"],
                "metric_type": chunk_data.get("metric_type"),
                "source_type": chunk_data.get("source_type"),
                "is_gaap": chunk_data.get("is_gaap"),
                "text": chunk_data["text"],
                "sequence_index": chunk_data.get("sequence_index"),
                "is_analyst_question": chunk_data.get("is_analyst_question", False),
                "dense_embedding": dense_embeddings[j].tolist(),
                "sparse_embedding": sparse_vec
            }
            records.append(record)
            
        try:
            # Use bulk_insert_mappings for ORM compatibility
            db.bulk_insert_mappings(DocumentChunk, records)
            db.commit()
            logger.info(f"Indexed batch of {len(records)} chunks.")
        except Exception as e:
            db.rollback()
            logger.error(f"Error indexing batch: {e}")
            # Log the first record to see what might be wrong
            if records:
                logger.error(f"Sample record keys: {records[0].keys()}")
                logger.error(f"Sample dense shape: {len(records[0]['dense_embedding'])}")
            raise


def index_company(ticker: str, transcripts: List[Transcript], financials: Dict[str, Any], db: Session = None):
    """
    Chunk all data, index everything into PostgreSQL.
    """
    logger.info(f"Starting indexing for {ticker}")
    all_chunks = []
    
    # Chunk financial data
    fin_chunks = chunk_financial_data(ticker, financials)
    all_chunks.extend(fin_chunks)
    logger.info(f"Generated {len(fin_chunks)} financial chunks for {ticker}")
    
    # Chunk transcripts
    for transcript in transcripts:
        tr_chunks = chunk_transcript_data(ticker, transcript)
        all_chunks.extend(tr_chunks)
        logger.info(f"Generated {len(tr_chunks)} transcript chunks for {ticker} {transcript.year}Q{transcript.quarter}")
    
    # === PRE-FLIGHT: detect and split any overly-long chunks ===
    # ensure tokenizer is the same one used for embeddings
    # hf_tokenizer is already defined globally in your file
    MAX_MODEL_TOKENS = 512
    SAFE_CHUNK_TOKENS = 450

    new_all_chunks = []
    for idx, ch in enumerate(all_chunks):
        text = ch["text"]
        tok_len = len(hf_tokenizer.encode(text, add_special_tokens=False))
        if tok_len > MAX_MODEL_TOKENS:
            logger.warning(f"LONG CHUNK detected (tokens={tok_len}) for {ch.get('ticker')} {ch.get('chunk_type')} — splitting by sentences.")
            # preview log
            preview = text[:400].replace("\n", " ")
            logger.warning(f"Preview: {preview}...")
            # split into safe subchunks (preserve sentences)
            sub_texts = split_text_preserve_sentences(text, hf_tokenizer, max_tokens=SAFE_CHUNK_TOKENS)
            for sub in sub_texts:
                new_ch = ch.copy()
                new_ch["text"] = sub
                # keep sequence_index as -1 for now; will be set later if you want
                new_all_chunks.append(new_ch)
        else:
            new_all_chunks.append(ch)

    # Optionally reassign sequence_index deterministically
    for i, ch in enumerate(new_all_chunks):
        ch["sequence_index"] = i

    all_chunks = new_all_chunks

    # Now safe to embed
    index_documents(all_chunks, db)
    logger.info(f"Finished indexing for {ticker}")


def index_all_companies_from_db(companies: List[str], quarters: List[str], db_session: Session):
    """
    Index all companies and quarters from the database.
    """
    logger.info("Starting indexing from database")
    
    for ticker in companies:
        logger.info(f"Processing {ticker}...")
        
        # Get financial data from DB
        financial_query = db_session.query(FinancialDataModel).filter(
            FinancialDataModel.ticker == ticker
        )
        
        # Organize financial data by period
        financials = {}
        for record in financial_query.all():
            period = f"{record.year}Q{record.quarter}"
            if period not in financials:
                financials[period] = {
                    "source": record.source,
                    "metrics": {}
                }
            financials[period]["metrics"][record.metric] = record.value
            
        logger.info(f"Found financial data for {len(financials)} quarters for {ticker}")
        
        # Get transcripts from DB
        transcript_query = db_session.query(TranscriptRecord).filter(
            TranscriptRecord.ticker == ticker
        )
        
        # Convert DB records to Transcript objects
        transcripts = []
        for record in transcript_query.all():
            transcript = Transcript(
                ticker=record.ticker,
                year=record.year,
                quarter=record.quarter,
                date=record.date,
                segments=record.segments
            )
            transcripts.append(transcript)
            
        logger.info(f"Found {len(transcripts)} transcripts for {ticker}")
        
        # Index the company data
        try:
            index_company(ticker, transcripts, financials, db=db_session)
        except Exception as e:
            logger.error(f"Error indexing {ticker}: {e}")
            continue
            
    logger.info("Indexing from database completed")


def clear_existing_chunks(ticker: str, db: Session):
    """
    Clear existing chunks for a specific company (useful for re-indexing)
    
    CRITICAL: This is a DESTRUCTIVE operation. It will DELETE all indexed chunks for a company.
    This function is DISABLED in production to protect the verified data.
    """
    from src.config import ALLOW_DESTRUCTIVE_OPERATIONS
    
    if not ALLOW_DESTRUCTIVE_OPERATIONS:
        raise RuntimeError(
            f"DESTRUCTIVE OPERATION BLOCKED: Cannot clear chunks for {ticker}. "
            f"The data has been fully ingested and verified. "
            f"Set ALLOW_DESTRUCTIVE_OPERATIONS=true in .env ONLY if you are in development "
            f"and explicitly need to re-index data."
        )
    
    try:
        db.query(DocumentChunk).filter(DocumentChunk.ticker == ticker).delete()
        db.commit()
        logger.info(f"Cleared existing chunks for {ticker}")
    except Exception as e:
        db.rollback()
        logger.error(f"Error clearing chunks for {ticker}: {e}")


def get_document_chunks_stats(db: Session) -> Dict[str, Any]:
    """
    Get statistics about document chunks in the database
    """
    stats = {}
    
    # Total chunks
    total_chunks = db.query(func.count(DocumentChunk.id)).scalar()
    stats["total_chunks"] = total_chunks
    
    # Chunks by type
    type_stats = db.query(
        DocumentChunk.chunk_type,
        func.count(DocumentChunk.id)
    ).group_by(DocumentChunk.chunk_type).all()
    
    stats["type_distribution"] = {chunk_type: count for chunk_type, count in type_stats}
    
    # Chunks by company
    company_stats = db.query(
        DocumentChunk.ticker,
        func.count(DocumentChunk.id)
    ).group_by(DocumentChunk.ticker).all()
    
    stats["company_distribution"] = {ticker: count for ticker, count in company_stats}
    
    return stats
