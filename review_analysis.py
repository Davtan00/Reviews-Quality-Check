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
from gensim.models import CoherenceModel
from tqdm import tqdm

def initialize_nltk():
    """
    Initialize NLTK resources once at startup. This checks for certain 
    tokenizers and downloads them if missing.
    """
    required_packages = ['stopwords', 'punkt', 'averaged_perceptron_tagger']
    for package in required_packages:
        try:
            nltk.data.find(f'tokenizers/{package}')
        except LookupError:
            nltk.download(package, quiet=True)

def calculate_similarity(data):
    """
    Run the SophisticatedSimilarityAnalyzer on the given data.
    Returns the dictionary from analyze_similarity(), which includes:
        {
            'exact_duplicates': [...],  # list of groups
            'similar_pairs': [...],     # list of pairs
        }
    """
    analyzer = SophisticatedSimilarityAnalyzer()
    texts = [entry['text'] for entry in data]
    return analyzer.analyze_similarity(texts)

def analyze_quality(data):
    """
    Simple legacy function that calculates the Flesch Reading Ease score for each text.
    Returns a list of dictionaries with keys:
        - 'id'
        - 'text'
        - 'flesch_score'
    """
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
    
    logging.info(f"Starting sentiment validation for {len(data)} reviews...")
    
    with tqdm(total=len(data), desc="Validating sentiments") as pbar:
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
            
            pbar.update(1)
            pbar.set_postfix({
                'mismatches': len(mismatches),
                'current_id': entry['id']
            })
    
    logging.info(f"Found {len(mismatches)} sentiment mismatches")
    return mismatches

class ResourceManager:
    """
    Simple manager to handle creation and cleanup of shared resources.
    """
    def __init__(self):
        self.active_resources = []
        self.model_cache = {}
        
    def register(self, resource: Any) -> None:
        """Register a resource for cleanup."""
        self.active_resources.append(resource)
        
    def cleanup(self) -> None:
        """Clean up all registered resources."""
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
    """
    Parse command line arguments for the review analysis system.
    
    Returns:
        An argparse.Namespace with the parsed arguments.
    """
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
                       help='Override JSON-specified domain for ALL files (use with caution)')
    parser.add_argument('-f', '--filter-domain',
                       type=str,
                       help='Process only files with specified domain')
    parser.add_argument('--filter-domains',
                       type=str,
                       help='Process only files with specified domains (comma-separated)')
    return parser.parse_args()

def get_file_domain(file_path: Path) -> str:
    """
    Get a 'domain' from a JSON file (defaults to 'general' on error).
    """
    try:
        with open(file_path, 'r') as f:
            json_data = json.load(f)
            return json_data.get("domain", "general")
    except Exception as e:
        logging.error(f"Error reading domain from {file_path}: {str(e)}")
        return "general"

def should_process_file(file_path: Path, args) -> bool:
    """
    Determine if file should be processed based on domain filters.
    If no filter options are set, all files are processed.
    """
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
    """
    Context manager for instantiating the core analyzers
    (sentiment, similarity, topic), ensuring they are cleaned up properly.
    """
    resource_manager = ResourceManager()
    try:
        # Initialize analyzers with resource management
        sentiment_validator = SentimentValidator(model_key=model_key)
        similarity_analyzer = SophisticatedSimilarityAnalyzer()
        topic_analyzer = SophisticatedTopicAnalyzer(min_topics=2, max_topics=10, verbose=True)
        
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
    """
    Process a single file with comprehensive topic and quality analysis,
    plus sentiment mismatch detection and similarity checks.
    
    1. Load data from JSON.
    2. Run similarity analysis to detect exact duplicates & near-duplicates.
    3. Calculate Flesch reading scores for each review.
    4. Validate sentiments for mismatches.
    5. Perform topic analysis.
    6. N-gram diversity analysis.
    7. Return a dictionary of results including duplicates, mismatches, etc.
    """
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
            
            # 1) Similarity: detect exact duplicates, near duplicates
            print("Analyzing similarities...")
            similarity_result = analyzers['similarity'].analyze_similarity(
                [entry['text'] for entry in data]
            )
            exact_duplicates = similarity_result['exact_duplicates']
            similar_pairs = similarity_result['similar_pairs']
            
            # 2) Flesch reading scores
            print("Analyzing reading quality (Flesch) per review...")
            quality_scores = analyze_quality(data)
            
            # 3) Sentiment validation
            print("Validating sentiments...")
            sentiment_mismatches = validate_sentiments_batch(data, domain, model_key)
            
            # 4) Topic analysis
            print("Analyzing topics...")
            texts = [entry['text'] for entry in data]
            topic_analysis = analyzers['topic'].analyze_topics(texts)
            
            # 5) N-gram diversity
            print("Analyzing n-gram diversity...")
            ngram_diversity = analyze_ngram_diversity(data)
            
            # Convert exact_duplicates to a single integer count of duplicates
            # (sum of group sizes minus 1 per group).
            duplicates_found_count = sum((len(group) - 1) for group in exact_duplicates if len(group) > 1)
            
            # Prepare final report dictionary
            return {
                'total_reviews': total_reviews,
                'duplicates_found': duplicates_found_count,
                'average_similarity': 0.0,  # TODO: implement
                'high_similarity_pairs': len(similar_pairs),
                
                'average_linguistic_quality': (
                    sum(s['flesch_score'] for s in quality_scores) / total_reviews
                    if quality_scores else 0.0
                ),
                'unigram_diversity': ngram_diversity.get('1-gram_diversity', 0.0),
                'bigram_diversity': ngram_diversity.get('2-gram_diversity', 0.0),
                'trigram_diversity': ngram_diversity.get('3-gram_diversity', 0.0),
                
                'topic_coherence_cv': topic_analysis['coherence_scores']['c_v'],
                'topic_coherence_umass': topic_analysis['coherence_scores']['u_mass'],
                'topic_diversity': topic_analysis['topic_diversity'],
                
                'sentiment_mismatches': len(sentiment_mismatches),
                'sentiment_confidence': (
                    sum(m['confidence'] for m in sentiment_mismatches) / len(sentiment_mismatches)
                    if sentiment_mismatches else 0.0
                ),
                # Additional details
                'duplicates': exact_duplicates,  # list of groups
                'similarity': similar_pairs,
                'sentiment_mismatches_details': sentiment_mismatches,
                'topic_analysis_details': {
                    'topics': {
                        i: [t['term'] for t in topic['terms']] 
                        for i, topic in enumerate(topic_analysis.get('topics', []))
                    }
                },
                'linguistic_analysis': {
                    'individual_scores': [
                        {
                            'overall_score': s['flesch_score'],
                            'coherence_score': s['flesch_score'],  # TODO: refine if needed
                            'sophistication_score': s['flesch_score']
                        }
                        for s in quality_scores
                    ]
                }
            }
            
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {str(e)}")
            raise

# Helper function for specialized usage
def analyze_topic_coherence(data: List[Dict[str, Any]], n_topics: int = 5):
    """
    Specialized LDA-based topic modeling approach with adaptive parameters.
    Can be used for deeper analysis or separate from the main pipeline.
    """
    analyzer = SophisticatedTopicAnalyzer(
        min_topics=n_topics,
        max_topics=n_topics,
        verbose=False
    )
    
    texts = [entry['text'] for entry in data]
    processed_texts = analyzer.preprocess_text(texts)
    
    dictionary = Dictionary(processed_texts)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]
    
    corpus_size = len(processed_texts)
    n_cores = max(1, cpu_count() - 1)
    
    # Fine-tune LDA hyperparams
    if corpus_size < 2000:
        passes = 15
        iterations = 200
        eval_every = 10
        chunk_size = 200
        alpha = 'symmetric'
        minimum_prob = 0.01
    else:
        passes = 10
        iterations = 100
        eval_every = 50
        chunk_size = max(2000, corpus_size // (n_cores * 2))
        alpha = 'auto'
        minimum_prob = 0.01
    
    lda_params = {
        'num_topics': n_topics,
        'passes': passes,
        'iterations': iterations,
        'chunksize': chunk_size,
        'random_state': 42,
        'minimum_probability': minimum_prob,
        'dtype': np.float32,
        'per_word_topics': True,
        'update_every': 1,
        'eval_every': eval_every,
        'alpha': alpha,
        'eta': 'symmetric'
    }
    
    try:
        lda_model = LdaMulticore(
            corpus=corpus,
            id2word=dictionary,
            workers=n_cores,
            **lda_params
        )
    except Exception as e:
        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            **lda_params
        )
    
    # Compute c_v coherence
    coherence_score = 0.0
    try:
        coherence_model = CoherenceModel(
            model=lda_model,
            texts=processed_texts,
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence_score = coherence_model.get_coherence()
    except Exception:
        pass
    
    # Extract topics
    topics = []
    for topic_id in range(n_topics):
        topic_terms = lda_model.show_topic(topic_id, topn=10)
        topics.append({
            'id': topic_id,
            'terms': [{'term': term, 'weight': weight} for term, weight in topic_terms],
            'coherence': calculate_topic_coherence(topic_terms)
        })
    
    # Compute doc-topic distributions
    doc_topics = []
    topic_distributions = []
    for doc in corpus:
        topic_dist = lda_model.get_document_topics(doc, minimum_probability=0.01)
        dist = [0] * n_topics
        for tid, weight in topic_dist:
            dist[tid] = weight
        topic_distributions.append(dist)
        doc_topics.append([
            {'topic_id': tid, 'weight': float(weight)}
            for tid, weight in topic_dist
        ])
    
    # Entropy-based diversity
    topic_diversity = np.mean([
        -sum(p * np.log2(p) if p > 0 else 0 for p in dist)
        for dist in topic_distributions
    ])
    
    return {
        'topics': topics,
        'doc_topic_distribution': doc_topics,
        'model_perplexity': float(lda_model.log_perplexity(corpus)),
        'coherence_score': float(coherence_score),
        'topic_diversity': float(topic_diversity),
        'num_terms': len(dictionary),
        'num_documents': len(corpus),
        'used_multicore': isinstance(lda_model, LdaMulticore),
        'model_info': {
            'model_type': type(lda_model).__name__,
            'cores_used': getattr(lda_model, 'workers', 1),
            'corpus_size': len(corpus),
            'dictionary_size': len(dictionary),
            'parameters': lda_params
        }
    }

def calculate_topic_coherence(topic_terms):
    """
    Simple coherence measure for a single topic, based on pairwise 
    products of term weights.
    """
    terms = [term for term, _ in topic_terms]
    weights = [weight for _, weight in topic_terms]
    if len(weights) < 2:
        return 0.0
    
    return sum(w1 * w2 for w1, w2 in zip(weights[:-1], weights[1:])) / (len(weights) - 1)

def analyze_ngram_diversity(data, n_range=(1, 3)):
    """
    N-gram diversity analysis:
      - For n from 1 to 3, compute the ratio unique_ngrams / total_ngrams.
      - Removes NLTK stopwords if n == 1.
    """
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    results = {}
    stop_words = set(stopwords.words('english'))
    
    for n in range(n_range[0], n_range[1] + 1):
        all_ngrams = []
        total_ngrams = 0
        
        for entry in data:
            text = entry['text']
            if not text or not isinstance(text, str):
                continue
                
            tokens = word_tokenize(text.lower())
            if n == 1:
                tokens = [t for t in tokens if t not in stop_words]
            
            if len(tokens) >= n:
                from nltk.util import ngrams
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
    Master analysis function combining:
      - Similarity detection (with duplicates removal)
      - Topic analysis
      - Linguistic analysis
      - N-gram diversity
    Then saves cleaned data to JSON for further usage.
    """
    reviews = data['generated_data']
    texts = [review['text'] for review in reviews]
    
    # Initialize analyzers
    similarity_analyzer = SophisticatedSimilarityAnalyzer()
    topic_analyzer = SophisticatedTopicAnalyzer()
    linguistic_analyzer = SophisticatedLinguisticAnalyzer()
    
    # Analyze similarities
    similarity_results = similarity_analyzer.analyze_similarity(texts)
    
    # Remove duplicates
    indices_to_remove = set()
    for group in similarity_results['similar_pairs']:
        if len(group) > 1:
            # Keep the first item in each group, remove others
            indices_to_remove.update(d['index'] for d in group[1:])
    
    cleaned_reviews = [review for i, review in enumerate(reviews) if i not in indices_to_remove]
    
    # Summarize sentiment distribution
    sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    for review in cleaned_reviews:
        sentiment_counts[review['sentiment']] += 1
    
    cleaned_data = {
        'domain': data['domain'],
        'generated_data': cleaned_reviews,
        'summary': {
            'total_generated': len(cleaned_reviews),
            'sentiment_distribution': sentiment_counts,
            'duplicates_removed': len(indices_to_remove)
        }
    }
    
    # Save cleaned data to JSON
    cleaned_json_path = os.path.join(REPORT_FOLDER, 'cleaned_reviews.json')
    with open(cleaned_json_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
    
    # Topic analysis
    topic_results = topic_analyzer.analyze_topics(texts)
    
    # Run our new linguistic analysis on each text
    linguistic_results = [linguistic_analyzer.analyze_quality(t) for t in texts]
    
    # N-gram diversity
    ngram_diversity = analyze_ngram_diversity(texts)
    
    return {
        'similarity_analysis': similarity_results,
        'topic_analysis': topic_results,
        'linguistic_analysis': linguistic_results,
        'ngram_diversity': ngram_diversity,
        'cleaned_data_path': cleaned_json_path
    }

def main():
    """
    Main entry point for the command-line usage.
    Performs:
      1. Argument parsing
      2. Optional listing of sentiment models
      3. Domain override warning
      4. NLTK initialization
      5. File discovery + domain listing
      6. Processing each file with process_file()
      7. Generating PDF reports if needed
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(REPORT_FOLDER, 'analysis.log'))
        ]
    )
    
    args = parse_arguments()
    
    # Handle --list-models flag
    if args.list_models:
        models = SentimentValidator.list_available_models()
        logging.info("\nAvailable sentiment analysis models:")
        for key, description in models.items():
            logging.info(f"  {key}: {description}")
        return
    
    # Ensure output folder exists
    os.makedirs(REPORT_FOLDER, exist_ok=True)
    
    # Optional domain override warning
    if args.domain:
        logging.warning(
            f"\nDOMAIN OVERRIDE WARNING:\n"
            f"Domain override '{args.domain}' will be applied to ALL processed files.\n"
            f"This will ignore the original domains in the files."
        )
        user_input = input("Do you want to continue? (y/n): ").lower()
        if user_input != 'y':
            logging.info("Operation aborted by user.")
            return
    
    # NLTK setup
    initialize_nltk()
    
    try:
        # Summarize domains for discovered JSON files
        logging.info("\nFile Domain Summary:")
        json_files = list(Path(GENERATED_DATA_FOLDER).glob('*.json'))
        for file_path in json_files:
            domain = get_file_domain(file_path)
            will_process = should_process_file(file_path, args)
            status = "WILL PROCESS" if will_process else "SKIPPED"
            logging.info(f"  - {file_path.name}: {domain} [{status}]")
        
        # Process files that pass domain filters
        processed_count = 0
        for file_path in json_files:
            if not should_process_file(file_path, args):
                continue
            
            try:
                logging.info(f"\nProcessing file: {file_path.name}")
                results = process_file(file_path, args.model, args.domain)
                
                # Build up metrics for PDF
                quality_metrics = {
                    'total_reviews': results['total_reviews'],
                    'average_linguistic_quality': results['average_linguistic_quality'],
                    'topic_diversity': results['topic_diversity'],
                    'topic_coherence_cv': results['topic_coherence_cv'],
                    'topic_coherence_umass': results['topic_coherence_umass'],
                    'sentiment_confidence': results['sentiment_confidence']
                }

                # Generate PDF report
                report_file_path = os.path.join(
                    REPORT_FOLDER,
                    f"analysis_report_{file_path.stem}_{args.model}.pdf"
                )
                
                generate_pdf_report(
                    report_file_path,
                    quality_metrics,
                    results['duplicates'],
                    results['sentiment_mismatches_details'],
                    results['similarity']
                )
                
                logging.info(f"Report generated: {report_file_path}")
                processed_count += 1
            
            except Exception as e:
                logging.error(f"Error processing {file_path}: {str(e)}", exc_info=True)
                continue
        
        logging.info(f"\nProcessing complete. {processed_count} file(s) processed.")
                
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
