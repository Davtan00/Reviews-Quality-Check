from scipy.stats import entropy
import numpy as np
from typing import Dict, List, Any
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import logging

class StatisticalAnalyzer:
    def __init__(self):
        self.vectorizer_bigram = CountVectorizer(ngram_range=(2, 2), stop_words='english')
        self.vectorizer_trigram = CountVectorizer(ngram_range=(3, 3), stop_words='english')
    
    def calculate_kl_divergence(self, real_dist: Dict[str, float], 
                              synthetic_dist: Dict[str, float]) -> float:
        """Calculate KL divergence between real and synthetic distributions"""
        # Ensure both distributions have the same categories
        all_categories = sorted(set(real_dist.keys()) | set(synthetic_dist.keys()))
        real_array = np.array([real_dist.get(cat, 0) for cat in all_categories])
        synth_array = np.array([synthetic_dist.get(cat, 0) for cat in all_categories])
        
        # Normalize distributions
        real_array = real_array / real_array.sum()
        synth_array = synth_array / synth_array.sum()
        
        return float(entropy(real_array, qk=synth_array))
    
    def analyze_ngrams(self, texts: List[str]) -> Dict[str, List[tuple]]:
        """Analyze n-grams in the given texts and return their frequencies."""
        logging.info(f"Starting ngram analysis with {len(texts)} texts")
        
        # Log the input texts
        for idx, text in enumerate(texts):
            logging.info(f"Text {idx + 1} (length {len(text)}): {text[:100]}...")  # First 100 chars
            
        if not texts:
            logging.error("No texts provided for ngram analysis")
            raise ValueError("No texts provided for analysis")
            
        try:
            # Analyze bigrams
            logging.info("Analyzing bigrams...")
            bigram_matrix = self.vectorizer_bigram.fit_transform(texts)
            bigram_freq = bigram_matrix.sum(axis=0).A1
            bigrams = [(word, bigram_freq[idx]) for word, idx in self.vectorizer_bigram.vocabulary_.items()]
            logging.info(f"Found {len(bigrams)} unique bigrams")

            # Analyze trigrams
            logging.info("Analyzing trigrams...")
            trigram_matrix = self.vectorizer_trigram.fit_transform(texts)
            trigram_freq = trigram_matrix.sum(axis=0).A1
            trigrams = [(word, trigram_freq[idx]) for word, idx in self.vectorizer_trigram.vocabulary_.items()]
            logging.info(f"Found {len(trigrams)} unique trigrams")

            return {
                'bigrams': sorted(bigrams, key=lambda x: x[1], reverse=True),
                'trigrams': sorted(trigrams, key=lambda x: x[1], reverse=True)
            }
            
        except Exception as e:
            logging.error(f"Error in ngram analysis: {str(e)}")
            logging.error(f"Text sample causing error: {texts[:2]}")  # Log first two texts
            raise