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
import glob
import traceback

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
            
            # Calculate sentiment distribution
            sentiment_dist = {
                'positive': sum(1 for d in data if d.get('sentiment') == 'positive'),
                'negative': sum(1 for d in data if d.get('sentiment') == 'negative'),
                'neutral': sum(1 for d in data if d.get('sentiment') == 'neutral')
            }
            
            # Calculate linguistic quality metrics
            print("Analyzing linguistic quality...")
            linguistic_analyzer = analyzers.get('linguistic', SophisticatedLinguisticAnalyzer())
            linguistic_scores = []
            for text in texts:
                try:
                    if isinstance(text, str) and text.strip():
                        score = linguistic_analyzer.analyze_quality(text)
                        linguistic_scores.append(score)
                except Exception as e:
                    logging.warning(f"Error analyzing text: {str(e)}")
                    linguistic_scores.append({
                        'quality_score': 0.0,
                        'readability_score': 0.0,
                        'structure_score': 0.0,
                        'vocabulary_score': 0.0,
                        'coherence_score': 0.0,
                        'grammar_score': 0.0
                    })
            
            # Calculate average linguistic scores
            avg_linguistic_quality = (
                sum(score['quality_score'] for score in linguistic_scores) / 
                len(linguistic_scores)
            ) if linguistic_scores else 0.0
            
            # Prepare the report dictionary with all required fields
            return {
                'total_reviews': total_reviews,
                'duplicates_found': len(duplicates),
                'average_similarity': similarity_result.get('average_similarity', 0.0),
                'high_similarity_pairs': len(similarity_result.get('high_similarity_pairs', [])),
                'linguistic_quality': avg_linguistic_quality,
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
                'sentiment_distribution': sentiment_dist,
                'topic_analysis_details': {
                    'topics': {i: [t['term'] for t in topic['terms']] 
                             for i, topic in enumerate(topic_analysis.get('topics', []))}
                },
                'linguistic_analysis': {
                    'individual_scores': linguistic_scores,
                    'average_score': avg_linguistic_quality
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
    statistics_analyzer = SophisticatedStatisticsAnalyzer()
    
    # Analyze similarities with tiered analysis
    similarity_results = similarity_analyzer.analyze_similarity(texts)
    
    # Create sets of indices to remove based on similarity tiers
    indices_to_remove = set()
    semantic_duplicates = []  # Tier 1: Embedding similarity > 0.98
    combined_duplicates = []  # Tier 2: Combined similarity > 0.92
    structural_similars = []  # Tier 3: N-gram similarity > 0.85
    
    # Process tiered results
    if 'tiered_results' in similarity_results:
        for pair in similarity_results['tiered_results']['tier1']:
            semantic_duplicates.append(pair)
            indices_to_remove.add(pair['index2'])
        
        for pair in similarity_results['tiered_results']['tier2']:
            if pair['index1'] not in indices_to_remove and pair['index2'] not in indices_to_remove:
                combined_duplicates.append(pair)
                indices_to_remove.add(pair['index2'])
        
        for pair in similarity_results['tiered_results']['tier3']:
            if pair['index1'] not in indices_to_remove and pair['index2'] not in indices_to_remove:
                structural_similars.append(pair)
    
    # Create cleaned data without duplicates
    cleaned_reviews = [review for i, review in enumerate(reviews) if i not in indices_to_remove]
    cleaned_texts = [review['text'] for review in cleaned_reviews]
    
    # Analyze sentiment mismatches with improved analysis
    sentiment_mismatches = []
    for review in cleaned_reviews:
        predicted_sentiment = linguistic_analyzer.analyze_sentiment(review['text'])
        if predicted_sentiment['sentiment'] != review['sentiment']:
            mismatch_analysis = statistics_analyzer.analyze_sentiment_mismatch(
                review['text'],
                review['sentiment'],
                predicted_sentiment['sentiment']
            )
            mismatch_analysis['confidence'] = predicted_sentiment['confidence']
            sentiment_mismatches.append(mismatch_analysis)
    
    # Analyze linguistic quality
    linguistic_scores = []
    for text in cleaned_texts:
        quality_result = linguistic_analyzer.analyze_quality(text)
        linguistic_scores.append(quality_result)
    
    # Calculate average scores
    avg_scores = {
        'quality_score': np.mean([score['quality_score'] for score in linguistic_scores]),
        'readability_score': np.mean([score['readability_score'] for score in linguistic_scores]),
        'structure_score': np.mean([score['structure_score'] for score in linguistic_scores]),
        'vocabulary_score': np.mean([score['vocabulary_score'] for score in linguistic_scores]),
        'coherence_score': np.mean([score['coherence_score'] for score in linguistic_scores]),
        'grammar_score': np.mean([score['grammar_score'] for score in linguistic_scores])
    }
    
    # Analyze topics
    topic_results = topic_analyzer.find_optimal_topics(cleaned_texts)
    
    # Update sentiment statistics
    sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    for review in cleaned_reviews:
        sentiment_counts[review['sentiment']] += 1
    
    # Prepare final report data
    cleaned_data = {
        'domain': data['domain'],
        'generated_data': cleaned_reviews,
        'summary': {
            'total_generated': len(cleaned_reviews),
            'sentiment_distribution': sentiment_counts,
            'similarity_summary': {
                'semantic_duplicates': len(semantic_duplicates),
                'combined_duplicates': len(combined_duplicates),
                'structural_similars': len(structural_similars),
                'total_removed': len(indices_to_remove)
            }
        }
    }
    
    # Save cleaned data
    cleaned_json_path = os.path.join(REPORT_FOLDER, 'cleaned_reviews.json')
    with open(cleaned_json_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
    
    return {
        'similarity_analysis': {
            'semantic_duplicates': semantic_duplicates,
            'combined_duplicates': combined_duplicates,
            'structural_similars': structural_similars,
            'total_removed': len(indices_to_remove)
        },
        'sentiment_mismatches': sentiment_mismatches,
        'topic_analysis': topic_results,
        'linguistic_analysis': {
            'average_scores': avg_scores,
            'detailed_scores': linguistic_scores
        },
        'cleaned_data_path': cleaned_json_path
    }

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(REPORT_FOLDER, 'analysis.log'))
        ]
    )

def main():
    """Main function to process files and generate reports"""
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Create reports directory
        os.makedirs(REPORT_FOLDER, exist_ok=True)
        
        # Get list of files to process
        json_files = glob.glob(os.path.join(GENERATED_DATA_FOLDER, "*.json"))
        
        # Print domain summary
        logger.info("\nFile Domain Summary:")
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    domain = data.get('domain', 'unknown')
                    logger.info(f"  - {os.path.basename(file_path)}: {domain} [WILL PROCESS]")
            except Exception as e:
                logger.warning(f"  - {os.path.basename(file_path)}: Error reading domain - {str(e)}")
        
        processed_count = 0
        
        for file_path in json_files:
            try:
                logger.info(f"Processing {file_path}...")
                
                # Load and validate data
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Process file and get analysis results
                analysis_results = process_file(Path(file_path), 'distilbert-sst2')
                
                # Save sentiment mismatch analysis
                sm_file_path = os.path.join(
                    REPORT_FOLDER,
                    f"SM_analysis_{os.path.basename(file_path)}"
                )
                with open(sm_file_path, 'w', encoding='utf-8') as f:
                    json.dump(analysis_results['sentiment_mismatches'], f, indent=2)
                logger.info(f"Saved sentiment mismatch analysis to: {sm_file_path}")
                
                # Generate PDF report
                report_path = os.path.join(
                    REPORT_FOLDER,
                    f"report_{os.path.basename(file_path).replace('.json', '.pdf')}"
                )
                generate_pdf_report(
                    output_path=report_path,
                    analysis_results=analysis_results,
                    original_data=data['generated_data'],
                    duplicates=analysis_results['duplicates'],
                    sentiment_mismatches=analysis_results['sentiment_mismatches'],
                    similarity_analysis={
                        'semantic_duplicates': analysis_results['similarity'],
                        'combined_duplicates': analysis_results['duplicates'],
                        'structural_similars': analysis_results['similarity'],
                        'total_removed': len(analysis_results['duplicates']) + len(analysis_results['similarity'])
                    }
                )
                logger.info(f"Generated analysis report: {report_path}")
                
                processed_count += 1
                logger.info(f"Successfully processed {file_path}")
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                logger.error(traceback.format_exc())
                continue
        
        logger.info(f"\nProcessing complete. {processed_count} file(s) processed.")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
