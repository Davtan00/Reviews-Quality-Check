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
            texts = [entry['text'] for entry in data]
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
                'topic_coherence_cv': topic_analysis['coherence_scores']['c_v'],
                'topic_coherence_umass': topic_analysis['coherence_scores']['u_mass'],
                'topic_diversity': topic_analysis['topic_diversity'],
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
     References:
    - Optimal LDA parameters: https://dl.acm.org/doi/10.1145/2133806.2133826
    - Small corpus optimization: https://arxiv.org/abs/1706.03797
    """
    analyzer = SophisticatedTopicAnalyzer(
        min_topics=n_topics,
        max_topics=n_topics,
        verbose=False
    )
    
    # Preprocess texts using the optimized parallel processor
    texts = [entry['text'] for entry in data]
    processed_texts = analyzer.preprocess_text(texts)
    
    # Create dictionary with more aggressive filtering
    dictionary = Dictionary(processed_texts)
    dictionary.filter_extremes(no_below=5, no_above=0.5)  # More aggressive filtering
    corpus = [dictionary.doc2bow(text) for text in processed_texts]
    
    # Optimize parameters based on corpus size
    corpus_size = len(processed_texts)
    n_cores = max(1, cpu_count() - 1)
    
    # Use optimized parameters
    if corpus_size < 2000:
        passes = 15          # Reduced from 150
        iterations = 200     # Reduced from 2000
        eval_every = 10      # Increased from 1
        chunk_size = 200     # Increased from 100
        alpha = 'symmetric'
        minimum_prob = 0.01  # Increased from 0.001
    else:
        passes = 10          # Reduced from 25
        iterations = 100     # Reduced from 250
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
    
    # Try multicore first
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
    
    # Calculate coherence
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
    
    # Extract topics and calculate coherence
    topics = []
    for topic_id in range(n_topics):
        topic_terms = lda_model.show_topic(topic_id, topn=10)
        topics.append({
            'id': topic_id,
            'terms': [{'term': term, 'weight': weight} for term, weight in topic_terms],
            'coherence': calculate_topic_coherence(topic_terms)
        })
    
    # Calculate document-topic distributions with entropy-based diversity
    doc_topics = []
    topic_distributions = []
    for doc in corpus:
        topic_dist = lda_model.get_document_topics(doc, minimum_probability=0.01)
        dist = [0] * n_topics
        for topic_id, weight in topic_dist:
            dist[topic_id] = weight
        topic_distributions.append(dist)
        doc_topics.append([{'topic_id': topic_id, 'weight': float(weight)} 
                          for topic_id, weight in topic_dist])
    
    # Calculate topic diversity using entropy
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
    reviews = data['generated_data']
    texts = [review['text'] for review in reviews]
    
    # Initialize analyzers
    similarity_analyzer = SophisticatedSimilarityAnalyzer()
    topic_analyzer = SophisticatedTopicAnalyzer()
    linguistic_analyzer = SophisticatedLinguisticAnalyzer()
    
    # Analyze similarities and find duplicates
    similarity_results = similarity_analyzer.analyze_similarity(texts)
    
    # Create a set of indices to remove (duplicates)
    indices_to_remove = set()
    for group in similarity_results['similar_pairs']:
        # Always keep the first review from each group, mark others as duplicates
        if len(group) > 1:
            # First review is kept, others are marked as duplicates
            indices_to_remove.update(d['index'] for d in group[1:])
    
    # Create cleaned data without duplicates
    cleaned_reviews = [review for i, review in enumerate(reviews) if i not in indices_to_remove]
    
    # Update summary statistics
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
    
    # Continue with other analyses
    topic_results = topic_analyzer.analyze_topics(texts)
    linguistic_results = linguistic_analyzer.analyze_quality(texts)
    ngram_diversity = analyze_ngram_diversity(texts)
    
    return {
        'similarity_analysis': similarity_results,
        'topic_analysis': topic_results,
        'linguistic_analysis': linguistic_results,
        'ngram_diversity': ngram_diversity,
        'cleaned_data_path': cleaned_json_path
    }

def main():
    """Main execution function"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(REPORT_FOLDER, 'analysis.log'))
        ]
    )
    
    # Parse arguments and process files
    args = parse_arguments()
    
    # Handle --list-models flag
    if args.list_models:
        models = SentimentValidator.list_available_models()
        logging.info("\nAvailable sentiment analysis models:")
        for key, description in models.items():
            logging.info(f"  {key}: {description}")
        return
    
    # Create necessary folders
    os.makedirs(REPORT_FOLDER, exist_ok=True)
    
    # Domain override warning
    if args.domain:
        logging.warning(
            f"\nDOMAIN OVERRIDE WARNING:\n"
            f"Domain override '{args.domain}' will be applied to ALL processed files.\n"
            f"This will ignore the original domains in the files."
        )
        user_input = input("Do you want to continue? (y/n): ").lower()
        if user_input != 'y':
            logging.info("Operation aborted by user")
            return
    
    # Initialize NLTK resources
    initialize_nltk()
    
    try:
        # Show domain summary
        logging.info("\nFile Domain Summary:")
        json_files = list(Path(GENERATED_DATA_FOLDER).glob('*.json'))
        for file_path in json_files:
            domain = get_file_domain(file_path)
            will_process = should_process_file(file_path, args)
            status = "WILL PROCESS" if will_process else "SKIPPED"
            logging.info(f"  - {file_path.name}: {domain} [{status}]")
        
        # Process files
        processed_count = 0
        for file_path in json_files:
            if not should_process_file(file_path, args):
                continue
            
            try:
                logging.info(f"\nProcessing file: {file_path.name}")
                results = process_file(file_path, args.model, args.domain)
                
                # Calculate quality metrics for the report
                quality_metrics = {
                    'total_reviews': results['stats']['total_reviews'],  
                    'average_linguistic_quality': results['stats']['average_linguistic_quality'],
                    'topic_diversity': results['stats']['topic_diversity'],
                    'topic_coherence_cv': results['stats']['topic_coherence_cv'],
                    'topic_coherence_umass': results['stats']['topic_coherence_umass'],
                    'sentiment_confidence': results['stats']['sentiment_confidence']
                }

                # Generate the analysis report
                report_file_path = os.path.join(
                    REPORT_FOLDER,
                    f"analysis_report_{file_path.stem}_{args.model}.pdf"
                )
                
                generate_pdf_report(
                    report_file_path,
                    quality_metrics,
                    results['duplicates'],
                    results['sentiment_mismatches'],
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

def process_file(file_path: Path, model: str, domain_override: str = None) -> Dict:
    """Process a single file and generate analysis results."""
    logging.info(f"Loading data from {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    reviews = data.get('generated_data', [])
    if not reviews:
        raise ValueError("No reviews found in the input file")
    
    logging.info(f"Loaded {len(reviews)} reviews")
    
    # Extract just the text from each review for similarity analysis
    review_texts = [review['text'] for review in reviews]
    
    # Initialize analyzers
    similarity_analyzer = SophisticatedSimilarityAnalyzer()
    sentiment_validator = SentimentValidator(model)
    
    # Find duplicates and similar reviews
    similarity_results = similarity_analyzer.analyze_similarity(review_texts)
    exact_duplicates = similarity_results['exact_duplicates']
    similar_pairs = similarity_results['similar_pairs']
    
    # Get sentiment mismatches
    sentiment_mismatches = sentiment_validator.validate_sentiments_batch(reviews, domain_override)
    
    # Create cleaned dataset without duplicates or very similar reviews
    cleaned_reviews = []
    duplicate_indices = set()
    
    # Add indices from exact duplicate groups (keeping first from each group)
    for group in exact_duplicates:
        # Keep the first review from each group, mark others as duplicates
        if len(group) > 1:
            # First review is kept, others are marked as duplicates
            duplicate_indices.update(d['index'] for d in group[1:])
    
    # Add indices from similar pairs (keeping first from each pair)
    for pair in similar_pairs:
        # Always keep the first review in the pair, remove the second
        duplicate_indices.add(pair['index2'])
    
    logging.info(f"Removing {len(duplicate_indices)} reviews marked as duplicates or too similar")
    
    # Create cleaned dataset
    for i, review in enumerate(reviews):
        if i not in duplicate_indices:
            # Create a new review dict with re-enumerated ID
            new_review = review.copy()
            new_review['id'] = len(cleaned_reviews) + 1  # Start IDs from 1
            cleaned_reviews.append(new_review)
    
    # Calculate sentiment distribution for cleaned reviews
    sentiment_dist = {
        'positive': sum(1 for r in cleaned_reviews if r.get('sentiment') == 'positive'),
        'negative': sum(1 for r in cleaned_reviews if r.get('sentiment') == 'negative'),
        'neutral': sum(1 for r in cleaned_reviews if r.get('sentiment') == 'neutral')
    }
    
    # Save cleaned dataset with proper structure
    cleaned_data = {
        'domain': domain_override or data.get('domain', ''),
        'generated_data': cleaned_reviews,
        'summary': {
            'total_generated': len(cleaned_reviews),
            'sentiment_distribution': sentiment_dist,
            'duplicates_removed': len(reviews) - len(cleaned_reviews),
            'similar_pairs_found': len(similar_pairs)
        }
    }
    
    cleaned_file_path = os.path.join(
        REPORT_FOLDER, 
        f"cleaned_{file_path.stem}_{model}.json"
    )
    with open(cleaned_file_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=2)
    
    logging.info(f"Saved cleaned dataset with {len(cleaned_reviews)} reviews to: {cleaned_file_path}")
    
    # Save sentiment mismatches to a separate file for reference
    sm_file_path = os.path.join(
        REPORT_FOLDER,
        f"SM_analysis_report_{file_path.stem}_{model}.json"
    )
    with open(sm_file_path, 'w', encoding='utf-8') as f:
        json.dump(sentiment_mismatches, f, indent=2)
    logging.info(f"Saved sentiment mismatches to: {sm_file_path}")
    
    return {
        'duplicates': exact_duplicates,
        'sentiment_mismatches': sentiment_mismatches,
        'similarity': similar_pairs,
        'stats': {
            'total_reviews': len(reviews),
            'average_linguistic_quality': 0.0,  # TODO: Implement linguistic quality analysis
            'topic_diversity': 0.0,  # TODO: Implement topic diversity analysis
            'topic_coherence_cv': 0.0,  # TODO: Implement topic coherence analysis
            'topic_coherence_umass': 0.0,
            'sentiment_confidence': sum(m['confidence'] for m in sentiment_mismatches) / len(sentiment_mismatches) if sentiment_mismatches else 0.0
        }
    }

if __name__ == "__main__":
    main()
