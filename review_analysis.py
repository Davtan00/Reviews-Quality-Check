import argparse
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
from configs.models import ModelConfig, DomainIndicators

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

def validate_sentiments_batch(data: List[Dict[str, Any]], domain: str, model_key: str = 'distilbert-sst2') -> List[Dict[str, Any]]:
    """
    Batch sentiment validation for multiple reviews.
    Returns only high-confidence mismatches.
    """
    validator = SentimentValidator(model_key=model_key)
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

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Advanced Review Analysis System')
    parser.add_argument('-m', '--model', 
                       choices=list(ModelConfig.SUPPORTED_MODELS.keys()),
                       default='distilbert-sst2',
                       help='Sentiment analysis model to use')
    parser.add_argument('-l', '--list-models', 
                       action='store_true',
                       help='List available sentiment analysis models')
    parser.add_argument('-d', '--domain', 
                       type=str,
                       help='Override json specifieddomain for ALL files (use with caution)')
    parser.add_argument('-f', '--filter-domain',
                       type=str,
                       help='Process only files with specified domain')
    parser.add_argument('--filter-domains',
                       type=str,
                       help='Process only files with specified domains (comma-separated)')
    return parser.parse_args()

def get_file_domain(file_path: Path) -> str:
    """Get domain from JSON file"""
    try:
        with open(file_path, 'r') as f:
            json_data = json.load(f)
            return json_data.get("domain", "general")
    except Exception as e:
        logging.error(f"Error reading domain from {file_path}: {str(e)}")
        return "general"

def should_process_file(file_path: Path, args) -> bool:
    """Determine if file should be processed based on domain filters"""
    if not (args.filter_domain or args.filter_domains):
        return True
        
    file_domain = get_file_domain(file_path)
    
    if args.filter_domain:
        return file_domain == args.filter_domain
        
    if args.filter_domains:
        allowed_domains = {d.strip() for d in args.filter_domains.split(',')}
        return file_domain in allowed_domains
        
    return True

@contextmanager
def managed_analyzers(model_key: str = 'distilbert-sst2'):
    """Context manager for analyzer resources"""
    resource_manager = ResourceManager()
    try:
        # Initialize analyzers with resource management
        sentiment_validator = SentimentValidator(model_key=model_key)
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

def process_file(file_path: Path, model_key: str, domain_override: str = None) -> Dict[str, Any]:
    """Process a single file with comprehensive topic and quality analysis"""
    with managed_analyzers(model_key=model_key) as analyzers:
        try:
            print(f"\nProcessing file: {file_path}")
            
            with open(file_path, 'r') as f:
                json_data = json.load(f)
                data = json_data["generated_data"]
                
                # More explicit domain handling with logging
                json_domain = json_data.get("domain")
                if domain_override:
                    domain = domain_override
                    logging.info(f"Using override domain: {domain} (original domain in file: {json_domain})")
                elif json_domain:
                    domain = json_domain
                    logging.info(f"Using domain from JSON file: {domain}")
                else:
                    domain = "general"
                    logging.warning(f"No domain specified in file or command line, using default: {domain}")
            
            total_reviews = len(data)
            logging.info(f"Processing {total_reviews} reviews with domain: {domain}")
            
            print("Finding duplicates...")
            duplicates = find_duplicates(data)
            
            print("Calculating similarities...")
            similarity_result = analyzers['similarity'].analyze_similarity(
                [entry['text'] for entry in data]
            )
            
            print("Analyzing quality...")
            quality_scores = analyze_quality(data)
            
            print("Validating sentiments...")
            sentiment_mismatches = validate_sentiments_batch(data, domain, model_key)
            
            print("Analyzing topics...")
            texts = [entry['text'] for entry in data]  # Ensure texts are defined
            topic_analysis = analyzers['topic'].analyze_topics(texts)
            
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
                'topic_diversity': topic_analysis.get('topic_diversity', 0.0),
                'dominant_topic_coherence': topic_analysis.get('topic_coherence', 0.0),
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
    Topic modeling with adaptive convergence parameters based on corpus size.
    
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
    
    # Optimize parameters based on corpus size
    corpus_size = len(processed_texts)
    n_cores = min(12, cpu_count() - 1)
    chunk_size = max(100, min(2000, corpus_size // (n_cores * 4)))
    
    # Adaptive parameters based on corpus size
    if corpus_size < 1000:
        passes = 100  # More passes for small corpora
        iterations = 1000
        eval_every = 10
    elif corpus_size < 5000:
        passes = 50
        iterations = 500
        eval_every = 25
    else:
        passes = 25  # Fewer passes needed for large corpora
        iterations = 250
        eval_every = 50
    
    logging.info(f"Corpus size: {corpus_size}, Using parameters: passes={passes}, iterations={iterations}")
    
    dictionary = Dictionary(processed_texts)
    dictionary.filter_extremes(no_below=5, no_above=0.7)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]
    
    print("Training LDA model...")
    lda_params = {
        'num_topics': n_topics,
        'passes': 50,  # Increased from 15 to 50
        'iterations': 500,  # Added explicit iterations
        'chunksize': chunk_size,
        'workers': n_cores,
        'random_state': 42,
        'minimum_probability': 0.01,
        'dtype': np.float32,
        'per_word_topics': True,
        'update_every': 1,  # Update model after every chunk
        'eval_every': 10,   # Compute perplexity every 10 updates
        'alpha': 'auto',    # Learn alpha parameter from data
        'eta': 'auto'       # Learn eta parameter from data
    }
    
    # Add minimum document length check
    min_doc_length = 10
    filtered_corpus = [doc for doc in corpus if len(doc) >= min_doc_length]
    
    if len(filtered_corpus) < len(corpus):
        logging.warning(f"Filtered out {len(corpus) - len(filtered_corpus)} documents with fewer than {min_doc_length} tokens")
    
    if not filtered_corpus:
        logging.error("No documents remaining after filtering")
        return {
            'topics': [],
            'doc_topic_distribution': [],
            'model_perplexity': 0.0,
            'num_terms': len(dictionary),
            'num_documents': 0,
            'used_multicore': False,
            'error': 'Insufficient data for topic modeling'
        }

    lda_params = {
        'num_topics': n_topics,
        'passes': passes,
        'iterations': iterations,
        'chunksize': chunk_size,
        'workers': n_cores,
        'random_state': 42,
        'minimum_probability': 0.01,
        'dtype': np.float32,
        'per_word_topics': True,
        'update_every': 1,
        'eval_every': eval_every,
        'alpha': 'auto',
        'eta': 'auto'
    }

    try:
        # Enable convergence monitoring through logging
        logging.basicConfig(level=logging.INFO)
        
        lda_model = LdaMulticore(
            corpus=filtered_corpus,
            id2word=dictionary,
            **lda_params
        )
        
    except Exception as e:
        logging.error(f"LdaMulticore error: {str(e)}")
        # Fallback to single-core processing
        multicore_params = ['workers']
        for param in multicore_params:
            lda_params.pop(param, None)
            
        lda_model = LdaModel(
            corpus=filtered_corpus,
            id2word=dictionary,
            **lda_params
        )
    
    # Extract topics and calculate coherence
    topics = []
    for topic_id in range(n_topics):
        topic_terms = lda_model.show_topic(topic_id, topn=10)
        topics.append({
            'id': topic_id,
            'terms': [{'term': term, 'weight': weight} for term, weight in topic_terms],
            'coherence': calculate_topic_coherence(topic_terms)
        })
    
    # Calculate document-topic distributions
    doc_topics = []
    for doc in filtered_corpus:
        topic_dist = lda_model.get_document_topics(doc, minimum_probability=0.01)
        doc_topics.append([{'topic_id': topic_id, 'weight': float(weight)} 
                          for topic_id, weight in topic_dist])
    
    return {
        'topics': topics,
        'doc_topic_distribution': doc_topics,
        'model_perplexity': float(lda_model.log_perplexity(filtered_corpus)),
        'num_terms': len(dictionary),
        'num_documents': len(filtered_corpus),
        'used_multicore': isinstance(lda_model, LdaMulticore)
    }

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
    """Main execution function"""
    args = parse_arguments()
    
    # Handle --list-models flag
    if args.list_models:
        models = SentimentValidator.list_available_models()
        print("\nAvailable sentiment analysis models:")
        for key, description in models.items():
            print(f"  {key}: {description}")
        return

    # Initialize NLTK resources
    initialize_nltk()
    
    # Domain override warning
    if args.domain:
        logging.warning(
            f"\nDOMAIN OVERRIDE WARNING:\n"
            f"Domain override '{args.domain}' will be applied to ALL processed files.\n"
            f"This will ignore the original domains in the files."
        )
        user_input = input("Do you want to continue? (y/n): ").lower()
        if user_input != 'y':
            print("Operation aborted.")
            return
    
    try:
        input_folder = Path(GENERATED_DATA_FOLDER)
        output_folder = Path(REPORT_FOLDER)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Show domain summary
        print("\nFile Domain Summary:")
        json_files = list(input_folder.glob('*.json'))
        for file_path in json_files:
            domain = get_file_domain(file_path)
            will_process = should_process_file(file_path, args)
            status = "WILL PROCESS" if will_process else "SKIPPED"
            print(f"  - {file_path.name}: {domain} [{status}]")
        print()
        
        # Process files
        processed_count = 0
        for file_path in json_files:
            if not should_process_file(file_path, args):
                continue
                
            try:
                results = process_file(file_path, args.model, args.domain)
                
                # Generate report name
                report_name = f"analysis_report_{file_path.stem}_{args.model}.pdf"
                report_path = output_folder / report_name
                
                # Generate PDF report
                generate_pdf_report(
                    file_name=str(report_path),
                    report=results,
                    duplicates=results['duplicates'],
                    sentiment_mismatches=results['sentiment_mismatches'],
                    similarity_pairs=results['similarity']
                )
                print(f"Report generated: {report_path}")
                processed_count += 1
                
            except Exception as e:
                logging.error(f"Error processing file {file_path}: {str(e)}")
                continue
        
        print(f"\nProcessing complete. {processed_count} file(s) processed.")
                
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
