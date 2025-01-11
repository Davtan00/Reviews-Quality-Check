import logging
from typing import Dict, List
from collections import Counter

import numpy as np
from scipy.stats import entropy
from sklearn.feature_extraction.text import CountVectorizer

class StatisticalAnalyzer:
    """
    A class for performing basic statistical analyses on text data, including:
      1. KL divergence between distributions (via scipy.stats.entropy).
      2. N-gram frequency analysis (bigrams and trigrams).
    """
    
    def __init__(self):
        """
        Initialize the StatisticalAnalyzer with two CountVectorizers:
          - One for bigrams (2,2)
          - One for trigrams (3,3)
        By default, both remove English stop words.
        
        Note:
          - scikit-learn does not natively support GPU or MPS for 
            CountVectorizer. These transformations run on CPU.
        """
        self.vectorizer_bigram = CountVectorizer(ngram_range=(2, 2), stop_words='english')
        self.vectorizer_trigram = CountVectorizer(ngram_range=(3, 3), stop_words='english')
    
    def calculate_kl_divergence(self, real_dist: Dict[str, float],
                                synthetic_dist: Dict[str, float]) -> float:
        """
        Calculate the KL divergence between two discrete distributions.
        
        Steps:
          1. Merge the keys from both distributions to ensure they have 
             the same categories.
          2. Create two arrays (real and synthetic) of probabilities.
          3. Normalize them to sum to 1.
          4. Use scipy.stats.entropy(real, qk=synthetic) to compute KL.
        
        Args:
            real_dist: A dictionary {category: probability} for the real distribution.
            synthetic_dist: A dictionary {category: probability} for the synthetic distribution.
        
        Returns:
            A float representing the KL divergence (D_KL(real || synthetic)).
            Values >= 0, with 0 meaning the two distributions match exactly.
        """
        # Ensure both distributions have the same categories
        all_categories = sorted(set(real_dist.keys()) | set(synthetic_dist.keys()))
        real_array = np.array([real_dist.get(cat, 0.0) for cat in all_categories])
        synth_array = np.array([synthetic_dist.get(cat, 0.0) for cat in all_categories])
        
        # Normalize distributions
        real_sum = real_array.sum()
        synth_sum = synth_array.sum()
        if real_sum == 0 or synth_sum == 0:
            # If either distribution sums to 0, KL is undefined. Return 0.0 or 
            # handle as needed.
            return 0.0
        
        real_array /= real_sum
        synth_array /= synth_sum
        
        return float(entropy(real_array, qk=synth_array))
    
    def analyze_ngrams(self, texts: List[str]) -> Dict[str, List[tuple]]:
        """
        Analyze n-grams in the given texts (bigrams and trigrams) 
        and return their frequencies.
        
        Steps:
          1. Fit and transform the bigram vectorizer on the texts.
          2. Collect each bigram and its frequency, store in a list.
          3. Fit and transform the trigram vectorizer on the same texts.
          4. Collect each trigram and its frequency, store in a list.
          5. Sort both lists by frequency in descending order.
        
        Args:
            texts: A list of string documents.
        
        Returns:
            A dictionary with:
              - 'bigrams': [(bigram, freq), ...]
              - 'trigrams': [(trigram, freq), ...]
        """
        logging.info(f"Starting n-gram analysis with {len(texts)} texts...")
        
        try:
            # Process bigrams
            bigram_matrix = self.vectorizer_bigram.fit_transform(texts)
            bigram_freq = bigram_matrix.sum(axis=0).A1  # sum frequencies column-wise
            bigrams = [
                (word, bigram_freq[idx]) 
                for word, idx in self.vectorizer_bigram.vocabulary_.items()
            ]

            # Process trigrams
            trigram_matrix = self.vectorizer_trigram.fit_transform(texts)
            trigram_freq = trigram_matrix.sum(axis=0).A1
            trigrams = [
                (word, trigram_freq[idx]) 
                for word, idx in self.vectorizer_trigram.vocabulary_.items()
            ]

            # Sort each by frequency (descending)
            return {
                'bigrams': sorted(bigrams, key=lambda x: x[1], reverse=True),
                'trigrams': sorted(trigrams, key=lambda x: x[1], reverse=True)
            }
            
        except Exception as e:
            # Log some data for debugging
            logging.error(f"Error in n-gram analysis: {str(e)}")
            logging.error(f"Text sample causing error: {texts[:2]}")
            raise
