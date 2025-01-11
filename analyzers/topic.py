import logging
from functools import partial
from typing import List, Dict, Any

import numpy as np
import spacy
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from gensim.models import CoherenceModel, LdaModel
from gensim.models.ldamulticore import LdaMulticore
from gensim.corpora import Dictionary
from gensim.models.phrases import Phrases, Phraser

class SophisticatedTopicAnalyzer:
    """
    A class for topic modeling with dynamic topic ranges and coherence-based evaluation.

    Args:
        min_topics: Minimum number of topics to consider (defaults to 2).
        max_topics: Maximum number of topics to consider (defaults to 10).
        verbose: If True, prints progress and debugging info to stdout.

    References:
        - "Finding scientific topics" (Griffiths & Steyvers, 2004)
        - "A heuristic approach to determine an appropriate number of topics in LDA models" 
          (Zhao et al., 2015)
    """
    
    def __init__(self, min_topics=2, max_topics=10, verbose=False):
        """
        Initializes spacy, sets up topic ranges, and configures logging verbosity.
        
        Note: We attempt to use spacy on a GPU or MPS if available, but gensim's
        LDA typically does not support hardware acceleration out of the box.
        """
        # Attempt to use GPU/MPS for spaCy pipeline. If not available, falls back to CPU.
        try:
            spacy.prefer_gpu()
        except Exception:
            pass
        
        # Load spaCy with minimal pipeline to reduce overhead
        self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        
        # Topic range
        self.min_topics = max(2, min_topics if isinstance(min_topics, int) else 2)
        self.max_topics = max(self.min_topics, max_topics if isinstance(max_topics, int) else 10)
        
        # LDA-related placeholders
        self.optimal_model = None
        self.dictionary = None
        self.corpus = None
        
        # Verbosity
        self.verbose = verbose
        self._setup_logging()

    def _setup_logging(self):
        """
        Configure logging level based on self.verbose.
        If verbose=True => INFO. Otherwise => WARNING.
        Uses force=True to override existing logging configs.
        """
        log_level = logging.INFO if self.verbose else logging.WARNING
        logging.basicConfig(level=log_level, format='%(message)s', force=True)

    @staticmethod
    def _process_text_chunk(text: str, nlp) -> List[str]:
        """
        Text processing for a single chunk of text (lowercase, remove stops/punct).
        
        Args:
            text: The text to process.
            nlp: A spaCy language pipeline.
        
        Returns:
            A list of lemmas, including only NOUN, VERB, ADJ tokens 
            for better topic coherence. (ADV is excluded by design.)
        """
        doc = nlp(text.lower())
        return [
            token.lemma_ for token in doc
            if not token.is_stop
            and not token.is_punct
            and token.pos_ in {'NOUN', 'VERB', 'ADJ'}
        ]

    def preprocess_text(self, texts: List[str]) -> List[List[str]]:
        """
        Perform optimized batch preprocessing on the input texts using spaCy,
        removing stopwords, punctuation, and restricting to NOUN/VERB/ADJ.
        
        Then apply a bigram detection step (Phrases) to group frequent pairs.

        Args:
            texts: A list of raw text strings.

        Returns:
            A list of tokenized documents, each doc is a list of strings (lemmas), 
            possibly combined with bigrams from Phrases.
        """
        if self.verbose:
            print("Starting text preprocessing...")

        total_texts = len(texts)
        
        # Adaptive batch sizing
        if total_texts < 5000:
            batch_size = 500
        elif total_texts < 20000:
            batch_size = 1000
        else:
            batch_size = 2000

        # We'll keep at least 1 core busy if no CPU or MPS usage for Gensim
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

        # Bigram detection
        if processed_texts:
            bigram = Phrases(processed_texts, min_count=5, threshold=100)
            bigram_mod = Phraser(bigram)
            
            bigrammed_texts = []
            for i in range(0, len(processed_texts), batch_size):
                batch = processed_texts[i:i + batch_size]
                bigrammed_texts.extend([bigram_mod[doc] for doc in batch])

            return bigrammed_texts
        
        return processed_texts

    def find_optimal_topics(self, texts: List[str]) -> Dict[str, Any]:
        """
        Perform topic modeling on the input texts, storing the 'optimal_model' internally.
        
        Steps:
          1. Preprocess texts (tokenization, lemmatization, bigrams).
          2. Filter out very short docs (<3 tokens).
          3. Build Dictionary; filter extremes.
          4. Build LDA corpus from these tokenized docs.
          5. Decide LDA parameters (passes, iterations, chunk_size, etc.) based on corpus size.
          6. Attempt LdaMulticore with n_cores. If that fails, fallback to single-core LdaModel.
          7. Compute coherence metrics (c_v, u_mass).
        
        Args:
            texts: A list of text documents for topic modeling.
        
        Returns:
            A dictionary containing:
              - 'model': the trained LDA model
              - 'coherence_scores': dict with 'c_v' and 'u_mass'
              - 'num_topics': final number of topics used
              - 'corpus_size': # of documents in corpus
              - 'dictionary_size': # of tokens in dictionary
        """
        processed_texts = self.preprocess_text(texts)
        corpus_size = len(processed_texts)

        # Filter out short documents
        processed_texts = [doc for doc in processed_texts if len(doc) >= 3]
        
        if not processed_texts:
            return {
                'model': None,
                'coherence_scores': {'c_v': 0.0, 'u_mass': 0.0},
                'num_topics': self.max_topics,
                'corpus_size': 0,
                'dictionary_size': 0
            }

        # Create dictionary
        self.dictionary = Dictionary(processed_texts)
        if corpus_size < 5000:
            no_below, no_above = 2, 0.8
        else:
            no_below, no_above = 3, 0.7
        
        self.dictionary.filter_extremes(no_below=no_below, no_above=no_above)
        
        # Ensure dictionary is sufficiently large
        if len(self.dictionary) < 50:
            logging.warning("Dictionary too small after filtering, adjusting parameters.")
            self.dictionary = Dictionary(processed_texts)
            self.dictionary.filter_extremes(no_below=2, no_above=0.95)

        self.corpus = [self.dictionary.doc2bow(doc) for doc in processed_texts]

        # Adaptive LDA parameters
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
        
        # Attempt multi-core LDA
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
                alpha='symmetric',
                minimum_probability=0.01
            )
        except Exception as e:
            logging.warning(f"Multicore LDA failed: {e}. Falling back to single-core LdaModel.")
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

        # Calculate coherence metrics
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
                    processes=n_cores  # The number of processes for coherence calc
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
        """
        Run topic analysis on the given texts. If no optimal_model is loaded, 
        calls find_optimal_topics first.
        
        Steps:
          1. If no model, find_optimal_topics => LDA model.
          2. Extract top words for each topic (topn=10).
          3. Compute doc-topic distributions in batches of 1000.
          4. Calculate a simple 'topic_diversity' measure using the average 
             per-document topic distribution entropy.
        
        Args:
            texts: List of strings.
        
        Returns:
            A dictionary with:
              - 'topics': a list of {id, terms: [{term, weight}, ...]}
              - 'doc_topic_distribution': a list of lists, each describing 
                the topic distribution for each document
              - 'topic_diversity': float
              - 'coherence_scores': dict from the find_optimal_topics step
              - 'model_info': metadata about the model
        """
        if not self.optimal_model:
            if self.verbose:
                print("Finding optimal topics...")
            optimal_results = self.find_optimal_topics(texts)
            if self.verbose:
                print(f"Coherence scores: {optimal_results['coherence_scores']}")
        else:
            # If a model was already loaded, we skip re-training
            optimal_results = {'coherence_scores': {'c_v': 0.0, 'u_mass': 0.0}}

        topics = []
        # Obtain top terms for each topic
        for topic_id in range(self.optimal_model.num_topics):
            topic_terms = self.optimal_model.show_topic(topic_id, topn=10)
            topics.append({
                'id': topic_id,
                'terms': [
                    {'term': term, 'weight': float(weight)}
                    for term, weight in topic_terms
                ]
            })

        # Batch process doc-topic distributions
        batch_size = 1000
        doc_topics = []
        topic_distributions = np.zeros((len(self.corpus), self.optimal_model.num_topics), dtype=float)

        for i in range(0, len(self.corpus), batch_size):
            batch = self.corpus[i:i + batch_size]
            for j, bow_doc in enumerate(batch):
                doc_topic_dist = self.optimal_model.get_document_topics(bow_doc, minimum_probability=0.01)
                for tid, prob in doc_topic_dist:
                    topic_distributions[i + j, tid] = prob
                doc_topics.append([
                    {'topic_id': tid, 'weight': float(prob)}
                    for tid, prob in doc_topic_dist
                ])

        # Calculate topic diversity based on entropy
        # Negative sum(p * log2 p) for each doc => average across docs
        eps = 1e-10
        doc_entropies = []
        for dist in topic_distributions:
            # Only consider > 0, ignoring zero entries to avoid log(0)
            entropy_val = -np.sum(np.where(dist > 0.0, dist * np.log2(dist + eps), 0.0))
            doc_entropies.append(entropy_val)
        topic_diversity = float(np.mean(doc_entropies)) if doc_entropies else 0.0

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
        """
        Resource cleanup to free memory references.
        Sets model, dictionary, and corpus to None.
        """
        self.optimal_model = None
        self.dictionary = None
        self.corpus = None
