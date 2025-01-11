import os
import logging
from typing import List, Dict, Any
from pathlib import Path

# Folder Configuration
GENERATED_DATA_FOLDER = "Generated Data"
REPORT_FOLDER = "Report"

# Analysis Thresholds
SIMILARITY_THRESHOLD = 0.92  # Combined similarity threshold for deduplication
EMBEDDING_SIMILARITY_THRESHOLD = 0.98  # Threshold for semantic similarity
NGRAM_SIMILARITY_THRESHOLD = 0.85  # Threshold for structural similarity
SENTIMENT_CONFIDENCE_THRESHOLD = 0.92  # Threshold for sentiment validation confidence
LINGUISTIC_QUALITY_THRESHOLD = 0.70  # Minimum acceptable linguistic quality score

# Similarity Analysis Features
ENABLE_TIERED_SIMILARITY = True  # Feature flag for tiered similarity analysis
SIMILARITY_TIERS = {
    'tier1': 0.98,  # Identical semantic content (embedding similarity)
    'tier2': 0.92,  # High combined similarity (for deduplication)
    'tier3': 0.85   # Structural similarity (n-gram based)
}

# Topic Analysis Configuration
MIN_TOPICS = 2  # Minimum number of topics to consider
MAX_TOPICS = 10  # Maximum number of topics to consider
MIN_TOPIC_COHERENCE = 0.3  # Minimum coherence score for topic modeling
MAX_ITERATIONS = 100  # Maximum iterations for topic modeling
NUM_PASSES = 5  # Number of passes for LDA

# Report Configuration
MAX_TEXT_LENGTH = 1000  # Maximum length for text snippets in reports
PDF_FONT_SIZE = 12  # Default font size for PDF reports
PDF_TITLE_SIZE = 16  # Font size for PDF titles
PDF_MARGIN = 15  # PDF margin in points

# Language Processing
MIN_WORDS_PER_REVIEW = 5  # Minimum number of words for a valid review
MAX_WORDS_PER_REVIEW = 300  # Maximum number of words to process per review
LANGUAGE_MODEL = 'en_core_web_sm'  # Default spaCy model to use

# Resource Management
BATCH_SIZE = 1000  # Number of items to process in each batch
CACHE_SIZE = 100  # Maximum number of items to keep in cache
TIMEOUT_SECONDS = 300  # Maximum time for processing operations

def setup_folders() -> None:
    """Create necessary folders if they don't exist"""
    folders = [GENERATED_DATA_FOLDER, REPORT_FOLDER]
    for folder in folders:
        try:
            os.makedirs(folder, exist_ok=True)
            logging.info(f"Ensured folder exists: {folder}")
        except Exception as e:
            logging.error(f"Failed to create folder {folder}: {str(e)}")
            raise

def validate_thresholds() -> None:
    """Validate threshold values"""
    threshold_checks = [
        (SIMILARITY_THRESHOLD, "SIMILARITY_THRESHOLD"),
        (EMBEDDING_SIMILARITY_THRESHOLD, "EMBEDDING_SIMILARITY_THRESHOLD"),
        (NGRAM_SIMILARITY_THRESHOLD, "NGRAM_SIMILARITY_THRESHOLD"),
        (SENTIMENT_CONFIDENCE_THRESHOLD, "SENTIMENT_CONFIDENCE_THRESHOLD"),
        (LINGUISTIC_QUALITY_THRESHOLD, "LINGUISTIC_QUALITY_THRESHOLD"),
        (MIN_TOPIC_COHERENCE, "MIN_TOPIC_COHERENCE")
    ]
    
    for threshold, name in threshold_checks:
        if not 0 <= threshold <= 1:
            raise ValueError(f"{name} must be between 0 and 1")

def validate_topic_settings() -> None:
    """Validate topic analysis settings"""
    if MIN_TOPICS >= MAX_TOPICS:
        raise ValueError("MIN_TOPICS must be less than MAX_TOPICS")
    
    if MAX_ITERATIONS <= 0:
        raise ValueError("MAX_ITERATIONS must be positive")
    
    if NUM_PASSES <= 0:
        raise ValueError("NUM_PASSES must be positive")

def validate_text_limits() -> None:
    """Validate text processing limits"""
    if MIN_WORDS_PER_REVIEW >= MAX_WORDS_PER_REVIEW:
        raise ValueError("MIN_WORDS_PER_REVIEW must be less than MAX_WORDS_PER_REVIEW")
    
    if MAX_TEXT_LENGTH <= 0:
        raise ValueError("MAX_TEXT_LENGTH must be positive")

def validate_resource_settings() -> None:
    """Validate resource management settings"""
    if BATCH_SIZE <= 0:
        raise ValueError("BATCH_SIZE must be positive")
    
    if CACHE_SIZE <= 0:
        raise ValueError("CACHE_SIZE must be positive")
    
    if TIMEOUT_SECONDS <= 0:
        raise ValueError("TIMEOUT_SECONDS must be positive")

def validate_config() -> None:
    """
    Validate all configuration settings.
    Raises ValueError if any validation fails.
    """
    try:
        setup_folders()
        validate_thresholds()
        validate_topic_settings()
        validate_text_limits()
        validate_resource_settings()
        logging.info("Configuration validated successfully")
    except Exception as e:
        logging.error(f"Configuration validation failed: {str(e)}")
        raise

def get_config() -> Dict[str, Any]:
    """
    Get configuration as a dictionary.
    Validates configuration before returning.
    """
    validate_config()
    return {
        # Folders
        'generated_data_folder': GENERATED_DATA_FOLDER,
        'report_folder': REPORT_FOLDER,
        
        # Analysis Thresholds
        'similarity_threshold': SIMILARITY_THRESHOLD,
        'embedding_similarity_threshold': EMBEDDING_SIMILARITY_THRESHOLD,
        'ngram_similarity_threshold': NGRAM_SIMILARITY_THRESHOLD,
        'sentiment_confidence_threshold': SENTIMENT_CONFIDENCE_THRESHOLD,
        'linguistic_quality_threshold': LINGUISTIC_QUALITY_THRESHOLD,
        
        # Similarity Analysis Features
        'enable_tiered_similarity': ENABLE_TIERED_SIMILARITY,
        'similarity_tiers': SIMILARITY_TIERS,
        
        # Topic Analysis
        'min_topics': MIN_TOPICS,
        'max_topics': MAX_TOPICS,
        'min_topic_coherence': MIN_TOPIC_COHERENCE,
        'max_iterations': MAX_ITERATIONS,
        'num_passes': NUM_PASSES,
        
        # Report Configuration
        'max_text_length': MAX_TEXT_LENGTH,
        'pdf_font_size': PDF_FONT_SIZE,
        'pdf_title_size': PDF_TITLE_SIZE,
        'pdf_margin': PDF_MARGIN,
        
        # Language Processing
        'min_words_per_review': MIN_WORDS_PER_REVIEW,
        'max_words_per_review': MAX_WORDS_PER_REVIEW,
        'language_model': LANGUAGE_MODEL,
        
        # Resource Management
        'batch_size': BATCH_SIZE,
        'cache_size': CACHE_SIZE,
        'timeout_seconds': TIMEOUT_SECONDS
    }

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    validate_config()