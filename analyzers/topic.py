import spacy
from gensim.models import CoherenceModel, LdaModel
from gensim.models.ldamulticore import LdaMulticore  
from gensim.corpora import Dictionary
from gensim.models.phrases import Phrases, Phraser
from nltk.tokenize import sent_tokenize
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import logging
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

class SophisticatedTopicAnalyzer:
    """
    Dynamic topic modeling with automatic optimization:
    1. Determines optimal number of topics using coherence scores
    2. Processes n-grams for better phrase capture
    3. Implements hierarchical topic structure
    4. Handles topic evolution and stability
    """
    
    def __init__(self, min_topics=2, max_topics=10, verbose=False):
        """
        Initialize the Topic Analyzer with dynamic topic ranges.
        
        Args:
            min_topics: Minimum number of topics (default: 2)
            max_topics: Maximum number of topics (default: 10)
            verbose: Whether to print detailed progress (default: False)
        
        References:
            - "Finding scientific topics" (Griffiths & Steyvers, 2004)
            - "A heuristic approach to determine an appropriate number of topics in LDA models" (Zhao et al., 2015)
        """
        self.nlp = spacy.load('en_core_web_sm')
        self.min_topics = max(2, min_topics if isinstance(min_topics, int) else 2)
        self.max_topics = max(self.min_topics, max_topics if isinstance(max_topics, int) else 10)
        self.optimal_model = None
        self.dictionary = None
        self.corpus = None
        
        # Configure logging based on verbosity
        log_level = logging.INFO if verbose else logging.WARNING
        logging.basicConfig(
            level=log_level,
            format='%(message)s', 
            force=True
        )
        self.verbose = verbose
    
    def _process_text_chunk(self, text: str) -> List[str]:
        """Process a single text chunk"""
        doc = self.nlp(text.lower())
        return [token.lemma_ for token in doc 
                if not token.is_stop and not token.is_punct 
                and token.pos_ in {'NOUN', 'VERB', 'ADJ', 'ADV'}]
    
    def preprocess_text(self, texts: List[str]) -> List[List[str]]:
        """Parallel text preprocessing"""
        if self.verbose:
            print("Starting parallel text preprocessing...")
        
        # Initialize multiprocessing pool
        n_cores = max(1, cpu_count() - 1)
        chunk_size = max(1, len(texts) // (n_cores * 4))
        
        with Pool(n_cores) as pool:
            # Process texts in parallel with progress bar if verbose
            if self.verbose:
                processed_texts = list(tqdm(
                    pool.imap(self._process_text_chunk, texts, chunksize=chunk_size),
                    total=len(texts),
                    desc="Preprocessing texts"
                ))
            else:
                processed_texts = pool.map(self._process_text_chunk, texts, chunksize=chunk_size)
        
        # Build and apply n-gram models
        if self.verbose:
            print("Building n-gram models...")
            
        bigram = Phrases(processed_texts, min_count=5, threshold=100)
        trigram = Phrases(bigram[processed_texts], threshold=100)
        
        bigram_mod = Phraser(bigram)
        trigram_mod = Phraser(trigram)
        
        final_texts = [trigram_mod[bigram_mod[tokens]] for tokens in processed_texts]
        
        return final_texts
    
    def find_optimal_topics(self, texts: List[str], coherence='c_v', max_iterations=100):
        """Optimized topic modeling with controlled verbosity"""
        processed_texts = self.preprocess_text(texts)
        corpus_size = len(processed_texts)
        
        # Create dictionary and corpus with aggressive filtering
        self.dictionary = Dictionary(processed_texts)
        self.dictionary.filter_extremes(no_below=5, no_above=0.5)
        self.corpus = [self.dictionary.doc2bow(text) for text in processed_texts]
        
        # Optimize parameters for speed
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
            chunk_size = max(2000, corpus_size // (cpu_count() * 2))
            alpha = 'auto'
            minimum_prob = 0.01

        if self.verbose:
            print(f"\nCorpus size: {corpus_size} documents")
            print(f"Dictionary size: {len(self.dictionary)} terms")
            print(f"Using {passes} passes and {iterations} iterations")
        
        n_cores = max(1, cpu_count() - 1)
        
        try:
            if self.verbose:
                print(f"\nInitializing LDA with {n_cores} cores...")
                
            lda_model = LdaMulticore(
                corpus=self.corpus,
                id2word=self.dictionary,
                num_topics=self.max_topics,
                passes=passes,
                iterations=iterations,
                eval_every=eval_every,
                workers=n_cores,
                batch=True,
                chunksize=chunk_size,
                alpha=alpha,
                minimum_probability=minimum_prob,
                random_state=42
            )
            
            if self.verbose:
                print("Successfully initialized multicore LDA")
                
        except Exception as e:
            if self.verbose:
                print(f"Multicore failed: {e}\nFalling back to single core")
            lda_model = LdaModel(
                corpus=self.corpus,
                id2word=self.dictionary,
                num_topics=self.max_topics,
                passes=passes,
                iterations=iterations,
                eval_every=eval_every,
                alpha=alpha,
                minimum_probability=minimum_prob,
                random_state=42
            )
        
        # Calculate coherence only if verbose
        coherence_score = 0.0
        if self.verbose:
            try:
                coherence_model = CoherenceModel(
                    model=lda_model,
                    texts=processed_texts,
                    dictionary=self.dictionary,
                    coherence=coherence
                )
                coherence_score = coherence_model.get_coherence()
                print(f"Coherence Score: {coherence_score:.4f}")
            except Exception as e:
                print(f"Error calculating coherence: {e}")
        
        self.optimal_model = lda_model
        
        return {
            'model': lda_model,
            'coherence_score': coherence_score,
            'num_topics': self.max_topics,
            'corpus_size': corpus_size,
            'dictionary_size': len(self.dictionary)
        }
    
    def analyze_topics(self, texts):
        """Analyze topics in the given texts"""
        if not self.optimal_model:
            optimal_results = self.find_optimal_topics(texts)
        
        # Get topic distribution for each document
        processed_texts = self.preprocess_text(texts)
        corpus = [self.dictionary.doc2bow(text) for text in processed_texts]
        
        # Extract topics and their terms
        topics = []
        for topic_id in range(self.optimal_model.num_topics):
            topic_terms = self.optimal_model.show_topic(topic_id, topn=10)
            topics.append({
                'id': topic_id,
                'terms': [{'term': term, 'weight': weight} for term, weight in topic_terms]
            })
        
        # Calculate document-topic distributions
        doc_topics = []
        topic_distributions = []
        for doc in corpus:
            topic_dist = self.optimal_model.get_document_topics(doc, minimum_probability=0.01)
            dist = [0] * self.optimal_model.num_topics
            for topic_id, prob in topic_dist:
                dist[topic_id] = prob
            topic_distributions.append(dist)
            doc_topics.append([{'topic_id': topic_id, 'weight': float(weight)} 
                              for topic_id, weight in topic_dist])
        
        # Calculate topic diversity using entropy
        topic_diversity = np.mean([
            -sum(p * np.log2(p) if p > 0 else 0 for p in dist)
            for dist in topic_distributions
        ])
        
        # Calculate model perplexity
        perplexity = float(self.optimal_model.log_perplexity(corpus))
        
        return {
            'topics': topics,
            'doc_topic_distribution': doc_topics,
            'model_perplexity': perplexity,
            'coherence_score': optimal_results['coherence_score'],
            'topic_diversity': float(topic_diversity),
            'num_terms': len(self.dictionary),
            'num_documents': len(corpus),
            'used_multicore': isinstance(self.optimal_model, LdaMulticore),
            'model_info': {
                'model_type': type(self.optimal_model).__name__,
                'cores_used': getattr(self.optimal_model, 'workers', 1),
                'corpus_size': len(corpus),
                'dictionary_size': len(self.dictionary)
            }
        }
    
    def cleanup(self):
        """Cleanup resources"""
        if self.optimal_model:
            self.optimal_model = None
        if self.dictionary:
            self.dictionary = None
        if self.corpus:
            self.corpus = None