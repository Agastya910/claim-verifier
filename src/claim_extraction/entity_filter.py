import logging
import re
from gliner import GLiNER
from src.models import Transcript
from src.config import FINANCIAL_ENTITY_TYPES

logger = logging.getLogger(__name__)

# Global model instance for reuse
_GLINER_MODEL = None

def _get_model():
    global _GLINER_MODEL
    if _GLINER_MODEL is None:
        logger.info("Initializing GLiNER model: urchade/gliner_medium-v2.1")
        _GLINER_MODEL = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
    return _GLINER_MODEL

def filter_financial_sentences(transcript: Transcript) -> list[dict]:
    """
    Filters earnings call transcript sentences using GLiNER to identify those with financial entities.
    Skips analyst Q&A segments.
    """
    model = _get_model()
    
    kept_sentences = []
    total_sentences = 0
    
    for i, segment in enumerate(transcript.segments):
        # Skip analyst Q&A segments (only verify management claims)
        if "Analyst" in segment.role:
            continue
            
        # Simple sentence splitting using regex
        # This handles common sentence endings while avoiding splitting on common abbreviations like 'vs.' or 'U.S.'
        # A more robust splitter like NLTK or SpaCy could be used if dependencies permit.
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', segment.text)
        total_sentences += len(sentences)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            entities = model.predict_entities(sentence, FINANCIAL_ENTITY_TYPES)
            
            # Filter for entities with confidence > 0.5
            significant_entities = [e for e in entities if e.get("score", 0) > 0.5]
            
            if significant_entities:
                kept_sentences.append({
                    "sentence": sentence,
                    "speaker": segment.speaker,
                    "role": segment.role,
                    "entities": significant_entities,
                    "segment_index": i
                })
                
    logger.info(f"Filtered {total_sentences} → {len(kept_sentences)} sentences containing financial entities")
    return kept_sentences
