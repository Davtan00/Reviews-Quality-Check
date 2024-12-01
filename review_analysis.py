import os
import json
import pandas as pd
import sys
import nltk
import numpy as np
from typing import Dict, List, Any
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from analyzers.sentiment import SentimentValidator
from analyzers.similarity import SophisticatedSimilarityAnalyzer
from analyzers.topic import SophisticatedTopicAnalyzer
from analyzers.linguistics import SophisticatedLinguisticAnalyzer
from utils.text_processing import sanitize_text, clean_text, calculate_flesch_reading_ease
from utils.report_generator import generate_pdf_report
from config import (
    GENERATED_DATA_FOLDER, 
    REPORT_FOLDER,
    MIN_TOPICS,
    MAX_TOPICS,
    SIMILARITY_THRESHOLD,
    SENTIMENT_CONFIDENCE_THRESHOLD
)
from contextlib import contextmanager
import logging
from pathlib import Path
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models.ldamulticore import LdaMulticore
from multiprocessing import cpu_count

def initialize_nltk():
    """Initialize NLTK resources once at startup"""
    required_packages = ['stopwords', 'punkt', 'averaged_perceptron_tagger']
    for package in required_packages:
        try:
            nltk.data.find(f'tokenizers/{package}')
        except LookupError:
            nltk.download(package, quiet=True)

def find_duplicates(data):
    texts = [entry['text'] for entry in data]
    duplicates = pd.Series(texts).duplicated(keep=False)
    return [data[i] for i in range(len(data)) if duplicates[i]]

def calculate_similarity(data):
    analyzer = SophisticatedSimilarityAnalyzer()
    texts = [entry['text'] for entry in data]
    return analyzer.analyze_similarity(texts)

def analyze_quality(data):
    results = []
    for entry in data:
        text = entry['text']
        score = calculate_flesch_reading_ease(text)
        results.append({
            "id": entry['id'],
            "text": text,
            "flesch_score": score
        })
    return results

def validate_sentiments_batch(data: List[Dict[str, Any]], domain: str) -> List[Dict[str, Any]]:
    """
    Batch sentiment validation for multiple reviews.
    Returns only high-confidence mismatches.
    """
    validator = SentimentValidator()
    mismatches = []
    
    for entry in data:
        result = validator.validate_sentiment(
            text=entry['text'],
            labeled_sentiment=entry['sentiment'],
            domain=domain
        )
        
        if result['is_mismatch']:
            mismatches.append({
                'id': entry['id'],
                'text': entry['text'],
                'expected': entry['sentiment'],
                'actual': result['predicted'],
                'confidence': result['confidence']
            })
    
    return mismatches

class ResourceManager:
    def __init__(self):
        self.active_resources = []
        self.model_cache = {}
        
    def register(self, resource: Any) -> None:
        """Register a resource for cleanup"""
        self.active_resources.append(resource)
        
    def cleanup(self) -> None:
        """Clean up all registered resources"""
        for resource in reversed(self.active_resources):
            try:
                if hasattr(resource, 'cleanup'):
                    resource.cleanup()
                elif hasattr(resource, 'close'):
                    resource.close()
            except Exception as e:
                logging.error(f"Error cleaning up resource {resource}: {str(e)}")
        self.active_resources.clear()
        self.model_cache.clear()

@contextmanager
def managed_analyzers():
    """Context manager for analyzer resources"""
    resource_manager = ResourceManager()
    try:
        # Initialize analyzers with resource management
        sentiment_validator = SentimentValidator()
        similarity_analyzer = SophisticatedSimilarityAnalyzer()
        topic_analyzer = SophisticatedTopicAnalyzer()
        
        # Register resources for cleanup
        resource_manager.register(sentiment_validator)
        resource_manager.register(similarity_analyzer)
        resource_manager.register(topic_analyzer)
        
        yield {
            'sentiment': sentiment_validator,
            'similarity': similarity_analyzer,
            'topic': topic_analyzer
        }
    finally:
        resource_manager.cleanup()

def process_file(file_path: Path) -> Dict[str, Any]:
    """Process a single file with proper resource management"""
    with managed_analyzers() as analyzers:
        try:
            print(f"\nProcessing file: {file_path}")
            
            with open(file_path, 'r') as f:
                json_data = json.load(f)
                data = json_data["generated_data"]
                domain = json_data.get("domain", "general")
            
            total_reviews = len(data)
            
            print("Finding duplicates...")
            duplicates = find_duplicates(data)
            
            print("Calculating similarities...")
            similarity_result = analyzers['similarity'].analyze_similarity(
                [entry['text'] for entry in data]
            )
            
            print("Analyzing quality...")
            quality_scores = analyze_quality(data)
            
            print("Validating sentiments...")
            sentiment_mismatches = validate_sentiments_batch(data, domain)
            
            print("Analyzing topics...")
            topic_analysis = analyze_topic_coherence(data, n_topics=5)
            
            # Calculate n-gram diversity
            ngram_diversity = analyze_ngram_diversity(data)
            
            # Prepare the report dictionary with all required fields
            return {
                'total_reviews': total_reviews,
                'duplicates_found': len(duplicates),
                'average_similarity': similarity_result.get('average_similarity', 0.0),
                'high_similarity_pairs': len(similarity_result.get('high_similarity_pairs', [])),
                'average_linguistic_quality': sum(s['flesch_score'] for s in quality_scores) / total_reviews if quality_scores else 0,
                'unigram_diversity': ngram_diversity.get('1-gram_diversity', 0.0),
                'bigram_diversity': ngram_diversity.get('2-gram_diversity', 0.0),
                'trigram_diversity': ngram_diversity.get('3-gram_diversity', 0.0),
                'topic_diversity': topic_analysis.get('model_perplexity', 0.0),
                'dominant_topic_coherence': max((t.get('coherence', 0.0) for t in topic_analysis.get('topics', [])), default=0.0),
                'sentiment_mismatches': len(sentiment_mismatches),
                'sentiment_confidence': sum(m['confidence'] for m in sentiment_mismatches) / len(sentiment_mismatches) if sentiment_mismatches else 0.0,
                'duplicates': duplicates,
                'similarity': similarity_result.get('high_similarity_pairs', []),
                'sentiment_mismatches': sentiment_mismatches,
                'topic_analysis_details': {
                    'topics': {i: [t['term'] for t in topic['terms']] 
                             for i, topic in enumerate(topic_analysis.get('topics', []))}
                },
                'linguistic_analysis': {
                    'individual_scores': [{'overall_score': s['flesch_score'], 
                                         'coherence_score': s['flesch_score'],  # TODO: Add more specific scores
                                         'sophistication_score': s['flesch_score']} 
                                        for s in quality_scores]
                }
            }
            
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {str(e)}")
            raise
# I might make a general version of this that can be used efficiently no matter the hardware and not just Ryzen 7 7800X3D
def analyze_topic_coherence(data: List[Dict[str, Any]], n_topics: int = 5):
    """
    Topic modeling optimized for high-performance hardware (Ryzen 7 7800X3D, 32GB DDR5)
    using LdaMulticore for parallel processing.
    
    Hardware optimization:
    - Uses up to 12 cores (leaving 4 threads for system)
    - Optimized chunk size for DDR5 memory
    - Batch processing enabled for better parallelization
    - Float32 dtype for memory efficiency
    
    Args:
        data: List of dictionaries containing review data
        n_topics: Number of topics to extract
        
    Returns:
        Dict containing topic analysis results
    """
    # Preprocess and tokenize texts
    logging.info("Starting text preprocessing...")
    processed_texts = []
    for entry in data:
        text = clean_text(sanitize_text(entry['text']))
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        processed_texts.append(tokens)
    
    # Hardware-optimized settings for my Ryzen 7 7800X3D
    n_cores = min(12, cpu_count() - 1)  # Use up to 12 threads, leaving 4 for the system
    chunk_size = max(2000, len(processed_texts) // (n_cores * 4))  # Optimize chunk size based on data and cores
    
    logging.info(f"Using {n_cores} cores with chunk size of {chunk_size}")
    
    # Create dictionary and corpus
    dictionary = Dictionary(processed_texts)
    dictionary.filter_extremes(no_below=5, no_above=0.7)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]
    
    # Updated parameters with only supported arguments
    lda_params = {
        'num_topics': n_topics,
        'passes': 15,
        'chunksize': chunk_size,
        'workers': n_cores,
        'random_state': 42,
        'minimum_probability': 0.01,
        'dtype': np.float32,
        'per_word_topics': True
    }
    
    try:
        logging.info("Starting LdaMulticore training with parallel processing...")
        lda_model = LdaMulticore(
            corpus=corpus,
            id2word=dictionary,
            **lda_params
        )
        logging.info("LdaMulticore training completed successfully")
        
    except Exception as e:
        logging.error(f"LdaMulticore error: {str(e)}")
        logging.warning("Falling back to single-core LdaModel (this will be slower)...")
        # Remove multicore-specific parameters for fallback
        multicore_params = ['workers', 'batch']
        for param in multicore_params:
            lda_params.pop(param, None)
            
        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            **lda_params
        )
    
    logging.info("Extracting topics and calculating distributions...")
    
    # Extract topics and their terms
    topics = []
    for topic_id in range(n_topics):
        topic_terms = lda_model.show_topic(topic_id, topn=10)
        topics.append({
            'id': topic_id,
            'terms': [{'term': term, 'weight': weight} for term, weight in topic_terms],
            'coherence': calculate_topic_coherence(topic_terms)
        })
    
    # Calculate document-topic distributions using parallel processing
    doc_topics = []
    for doc in corpus:
        topic_dist = lda_model.get_document_topics(doc, minimum_probability=0.01)
        doc_topics.append([{'topic_id': topic_id, 'weight': float(weight)} 
                          for topic_id, weight in topic_dist])
    
    results = {
        'topics': topics,
        'doc_topic_distribution': doc_topics,
        'model_perplexity': float(lda_model.log_perplexity(corpus)),
        'num_terms': len(dictionary),
        'num_documents': len(corpus),
        'used_multicore': isinstance(lda_model, LdaMulticore)
    }
    
    logging.info(f"Topic analysis completed using {'multicore' if results['used_multicore'] else 'single-core'} processing")
    
    return results

def calculate_topic_coherence(topic_terms):
    """Helper function to calculate coherence for a single topic"""
    terms = [term for term, _ in topic_terms]
    weights = [weight for _, weight in topic_terms]
    
    # Simple coherence measure based on term weights
    return sum(w1 * w2 for w1, w2 in zip(weights[:-1], weights[1:])) / (len(weights) - 1)

def analyze_ngram_diversity(data, n_range=(1, 3)):
    """
    N-gram diversity analysis for detecting text uniqueness:
    - Processes unigrams through trigrams
    - Calculates unique n-gram ratio
    - Handles edge cases (empty/invalid text)
    - Returns normalized diversity scores
    """
    results = {}
    stop_words = set(stopwords.words('english'))
    
    for n in range(n_range[0], n_range[1] + 1):
        all_ngrams = []
        total_ngrams = 0
        
        for entry in data:
            text = entry['text']
            if not text or not isinstance(text, str):
                continue
                
            # Tokenize and filter stop words for unigrams
            tokens = word_tokenize(text.lower())
            if n == 1:
                tokens = [t for t in tokens if t not in stop_words]
            
            if len(tokens) >= n:
                text_ngrams = list(ngrams(tokens, n))
                all_ngrams.extend(text_ngrams)
                total_ngrams += len(text_ngrams)
        
        if total_ngrams > 0:
            unique_ngrams = len(set(all_ngrams))
            diversity_score = unique_ngrams / total_ngrams
            results[f"{n}-gram_diversity"] = diversity_score
        else:
            results[f"{n}-gram_diversity"] = 0.0
    
    return results

def analyze_reviews_comprehensively(data):
    """
    Master analysis function combining topic modeling and linguistic quality assessment
    """
  
    topic_analyzer = SophisticatedTopicAnalyzer(min_topics=2, max_topics=10)
    linguistic_analyzer = SophisticatedLinguisticAnalyzer()
    
    texts = [entry['text'] for entry in data]
    
    
    topic_analysis = topic_analyzer.analyze_topics(texts)
    
    linguistic_analyses = []
    for entry in data:
        analysis = linguistic_analyzer.analyze_quality(entry['text'])
        linguistic_analyses.append({
            'id': entry['id'],
            'analysis': analysis
        })
    
    return {
        'topic_analysis': topic_analysis,
        'linguistic_analyses': linguistic_analyses
    }

def main():
    logging.info("Starting review analysis...")
    
    data_folder = Path(GENERATED_DATA_FOLDER)
    report_folder = Path(REPORT_FOLDER)
    
    # Add logging to see what files are found
    json_files = list(data_folder.glob('*.json'))
    logging.info(f"Found {len(json_files)} JSON files to process: {[f.name for f in json_files]}")
    
    for file_path in json_files:
        try:
            logging.info(f"Processing file: {file_path}")
            results = process_file(file_path)
            report_path = report_folder / f"{file_path.stem}_report.pdf"
            
            # Add logging before and after report generation
            logging.info(f"Generating report for {file_path.stem}")
            generate_pdf_report(
                file_name=str(report_path),
                report=results,
                duplicates=results['duplicates'],
                sentiment_mismatches=results['sentiment_mismatches'],
                similarity_pairs=results['similarity']
            )
            logging.info(f"Successfully generated report: {report_path}")
            
        except Exception as e:
            logging.error(f"Failed to process {file_path}: {str(e)}")
            # Add stack trace for better error diagnosis
            logging.exception("Detailed error information:")
            continue

if __name__ == "__main__":
    main()
