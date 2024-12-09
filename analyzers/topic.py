import numpy as np
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import logging
from gensim.models import CoherenceModel, LdaModel
from gensim.models.ldamulticore import LdaMulticore
from gensim.corpora import Dictionary
from gensim.models.phrases import Phrases, Phraser
import spacy
from functools import partial

class SophisticatedTopicAnalyzer:
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
    def __init__(self, min_topics=2, max_topics=10, verbose=False):
        self.nlp = spacy.load('en_core_web_sm', 
                             disable=['parser', 'ner'])  # Disable unnecessary components
        self.min_topics = max(2, min_topics if isinstance(min_topics, int) else 2)
        self.max_topics = max(self.min_topics, max_topics if isinstance(max_topics, int) else 10)
        self.optimal_model = None
        self.dictionary = None
        self.corpus = None
        self.verbose = verbose
        self._setup_logging()

    def _setup_logging(self):
        log_level = logging.INFO if self.verbose else logging.WARNING
        logging.basicConfig(level=log_level, format='%(message)s', force=True)

    @staticmethod
    def _process_text_chunk(text: str, nlp) -> List[str]:
        """Optimized text processing"""
        doc = nlp(text.lower())
        return [token.lemma_ for token in doc 
                if not token.is_stop and not token.is_punct 
                and token.pos_ in {'NOUN', 'VERB', 'ADJ'}]  # Removed ADV for better topic focus

    def preprocess_text(self, texts: List[str]) -> List[List[str]]:
        if self.verbose:
            print("Starting text preprocessing...")

        # Adaptive batch sizing based on dataset size
        total_texts = len(texts)
        if total_texts < 5000:
            batch_size = 500
        elif total_texts < 20000:
            batch_size = 1000
        else:
            batch_size = 2000

        n_cores = max(1, cpu_count() - 1)
        process_func = partial(self._process_text_chunk, nlp=self.nlp)

        processed_texts = []
        for i in range(0, total_texts, batch_size):
            batch = texts[i:i + batch_size]
            with Pool(n_cores) as pool:
                if self.verbose:
                    batch_processed = list(tqdm(
                        pool.imap(process_func, batch),
                        total=len(batch),
                        desc=f"Processing batch {i//batch_size + 1}"
                    ))
                else:
                    batch_processed = pool.map(process_func, batch)
                processed_texts.extend(batch_processed)

        # Optimized n-gram processing
        if processed_texts:
            bigram = Phrases(processed_texts, min_count=5, threshold=100)
            bigram_mod = Phraser(bigram)
            
            # Process bigrams in batches
            bigrammed_texts = []
            for i in range(0, len(processed_texts), batch_size):
                batch = processed_texts[i:i + batch_size]
                bigrammed_texts.extend([bigram_mod[doc] for doc in batch])

            return bigrammed_texts
        return processed_texts

    def find_optimal_topics(self, texts: List[str]) -> Dict[str, Any]:
        """Optimized topic modeling with adaptive parameters"""
        processed_texts = self.preprocess_text(texts)
        corpus_size = len(processed_texts)

        # More aggressive preprocessing for better coherence
        # Filter out very short documents
        processed_texts = [doc for doc in processed_texts if len(doc) >= 3]
        
        if not processed_texts:
            return {
                'model': None,
                'coherence_scores': {'c_v': 0.0, 'u_mass': 0.0},
                'num_topics': self.max_topics,
                'corpus_size': 0,
                'dictionary_size': 0
            }

        # Create dictionary with adaptive filtering
        self.dictionary = Dictionary(processed_texts)
        if corpus_size < 5000:
            no_below, no_above = 2, 0.8  # Less aggressive filtering for small corpora
        else:
            no_below, no_above = 3, 0.7
        self.dictionary.filter_extremes(no_below=no_below, no_above=no_above)
        
        if len(self.dictionary) < 50:  # Ensure we have enough terms
            logging.warning("Dictionary too small after filtering, adjusting parameters")
            self.dictionary = Dictionary(processed_texts)
            self.dictionary.filter_extremes(no_below=2, no_above=0.95)

        self.corpus = [self.dictionary.doc2bow(text) for text in processed_texts]

        # Adaptive parameters based on corpus size
        if corpus_size < 5000:
            passes, iterations = 15, 100
            chunk_size = 500
            eval_every = 10
        elif corpus_size < 20000:
            passes, iterations = 10, 50
            chunk_size = 2000
            eval_every = 25
        else:
            passes, iterations = 5, 30
            chunk_size = 3000
            eval_every = 50

        n_cores = max(1, cpu_count() - 1)
        
        try:
            if self.verbose:
                print(f"Training LDA model with {n_cores} cores...")
            
            self.optimal_model = LdaMulticore(
                corpus=self.corpus,
                id2word=self.dictionary,
                num_topics=self.max_topics,
                passes=passes,
                iterations=iterations,
                workers=n_cores,
                eval_every=eval_every,
                chunksize=chunk_size,
                random_state=42,
                dtype=np.float32,
                alpha='symmetric',  # Add explicit alpha
                minimum_probability=0.01  # Lower minimum probability
            )
        except Exception as e:
            logging.warning(f"Multicore LDA failed: {e}. Falling back to single core.")
            self.optimal_model = LdaModel(
                corpus=self.corpus,
                id2word=self.dictionary,
                num_topics=self.max_topics,
                passes=passes,
                iterations=iterations,
                eval_every=eval_every,
                random_state=42,
                dtype=np.float32,
                alpha='symmetric',
                minimum_probability=0.01
            )

        # Calculate coherence with better error handling and logging
        coherence_scores = {}
        for coherence_metric in ['c_v', 'u_mass']:
            try:
                if self.verbose:
                    print(f"Calculating {coherence_metric} coherence...")
                
                coherence_model = CoherenceModel(
                    model=self.optimal_model,
                    texts=processed_texts,
                    dictionary=self.dictionary,
                    coherence=coherence_metric,
                    processes=n_cores
                )
                coherence_scores[coherence_metric] = float(coherence_model.get_coherence())
                
                if self.verbose:
                    print(f"{coherence_metric} coherence: {coherence_scores[coherence_metric]:.4f}")
                    
            except Exception as e:
                logging.error(f"Error calculating {coherence_metric} coherence: {str(e)}")
                coherence_scores[coherence_metric] = 0.0

        return {
            'model': self.optimal_model,
            'coherence_scores': coherence_scores,
            'num_topics': self.max_topics,
            'corpus_size': len(self.corpus),
            'dictionary_size': len(self.dictionary)
        }

    def analyze_topics(self, texts: List[str]) -> Dict[str, Any]:
        """Optimized topic analysis with batched processing"""
        if not self.optimal_model:
            if self.verbose:
                print("Finding optimal topics...")
            optimal_results = self.find_optimal_topics(texts)
            if self.verbose:
                print(f"Coherence scores: {optimal_results['coherence_scores']}")
        else:
            optimal_results = {'coherence_scores': {'c_v': 0.0, 'u_mass': 0.0}}

        # Extract topics with improved memory efficiency
        topics = []
        for topic_id in range(self.optimal_model.num_topics):
            topic_terms = self.optimal_model.show_topic(topic_id, topn=10)
            topics.append({
                'id': topic_id,
                'terms': [{'term': term, 'weight': float(weight)} 
                         for term, weight in topic_terms]
            })

        # Batch process document topics
        batch_size = 1000
        doc_topics = []
        topic_distributions = np.zeros((len(self.corpus), self.optimal_model.num_topics))

        for i in range(0, len(self.corpus), batch_size):
            batch = self.corpus[i:i + batch_size]
            for j, doc in enumerate(batch):
                doc_topic_dist = self.optimal_model.get_document_topics(doc, minimum_probability=0.01)
                for topic_id, prob in doc_topic_dist:
                    topic_distributions[i + j, topic_id] = prob
                doc_topics.append([{'topic_id': topic_id, 'weight': float(weight)} 
                                 for topic_id, weight in doc_topic_dist])

        # Vectorized topic diversity calculation
        topic_diversity = float(np.mean(-np.sum(
            np.where(topic_distributions > 0,
                    topic_distributions * np.log2(topic_distributions + 1e-10),
                    0),
            axis=1
        )))

        return {
            'topics': topics,
            'doc_topic_distribution': doc_topics,
            'topic_diversity': topic_diversity,
            'coherence_scores': optimal_results['coherence_scores'],
            'model_info': {
                'model_type': type(self.optimal_model).__name__,
                'cores_used': getattr(self.optimal_model, 'workers', 1),
                'corpus_size': len(self.corpus),
                'dictionary_size': len(self.dictionary)
            }
        }

    def cleanup(self):
        """Resource cleanup"""
        self.optimal_model = None
        self.dictionary = None
        self.corpus = None