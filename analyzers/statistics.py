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
        
        try:
            # Process bigrams and trigrams separately for better memory management
            bigram_matrix = self.vectorizer_bigram.fit_transform(texts)
            bigram_freq = bigram_matrix.sum(axis=0).A1
            bigrams = [(word, bigram_freq[idx]) for word, idx in self.vectorizer_bigram.vocabulary_.items()]

            trigram_matrix = self.vectorizer_trigram.fit_transform(texts)
            trigram_freq = trigram_matrix.sum(axis=0).A1
            trigrams = [(word, trigram_freq[idx]) for word, idx in self.vectorizer_trigram.vocabulary_.items()]

            return {
                'bigrams': sorted(bigrams, key=lambda x: x[1], reverse=True),
                'trigrams': sorted(trigrams, key=lambda x: x[1], reverse=True)
            }
            
        except Exception as e:
            # Log problematic texts for debugging
            logging.error(f"Error in ngram analysis: {str(e)}")
            logging.error(f"Text sample causing error: {texts[:2]}")
            raise
    
    def analyze_sentiment_mismatch(self, review_text: str, original_sentiment: str, predicted_sentiment: str) -> Dict:
        """
        Analyzes why there might be a mismatch between original and predicted sentiment
        """
        analysis = {
            "text": review_text,
            "original_sentiment": original_sentiment,
            "predicted_sentiment": predicted_sentiment,
            "confidence": 0.0,  # Will be set by sentiment analyzer
            "analysis_factors": {
                "sentiment_indicators": [],
                "context_analysis": {
                    "has_contrasting_elements": False,
                    "contrasting_phrases": [],
                    "domain_specific_terms": [],
                    "intensity_markers": []
                },
                "linguistic_features": {
                    "contains_negation": False,
                    "contains_conditionals": False,
                    "contains_comparisons": False,
                    "contains_technical_terms": False
                },
                "explanation": ""
            }
        }
        
        # Check for contrasting elements
        contrast_markers = ["but", "however", "although", "though", "while", "despite", "yet"]
        found_contrasts = [marker for marker in contrast_markers if marker in review_text.lower()]
        if found_contrasts:
            analysis["analysis_factors"]["context_analysis"]["has_contrasting_elements"] = True
            analysis["analysis_factors"]["context_analysis"]["contrasting_phrases"] = found_contrasts
        
        # Check for negations
        negation_words = ["not", "no", "never", "neither", "nor", "none", "isn't", "aren't", "wasn't", "weren't"]
        if any(word in review_text.lower().split() for word in negation_words):
            analysis["analysis_factors"]["linguistic_features"]["contains_negation"] = True
            analysis["analysis_factors"]["sentiment_indicators"].append("contains_negation")
        
        # Check for conditionals
        conditional_words = ["if", "would", "could", "might", "may"]
        if any(word in review_text.lower().split() for word in conditional_words):
            analysis["analysis_factors"]["linguistic_features"]["contains_conditionals"] = True
            analysis["analysis_factors"]["sentiment_indicators"].append("contains_conditionals")
        
        # Check for technical terms
        technical_terms = ["functionality", "integration", "system", "configuration", "setup", "interface", "performance"]
        found_tech_terms = [term for term in technical_terms if term in review_text.lower()]
        if found_tech_terms:
            analysis["analysis_factors"]["linguistic_features"]["contains_technical_terms"] = True
            analysis["analysis_factors"]["context_analysis"]["domain_specific_terms"] = found_tech_terms
        
        # Check for intensity markers
        intensity_words = ["very", "extremely", "highly", "greatly", "significantly"]
        found_intensity = [word for word in intensity_words if word in review_text.lower()]
        if found_intensity:
            analysis["analysis_factors"]["context_analysis"]["intensity_markers"] = found_intensity
        
        # Generate explanation based on findings
        explanation_parts = []
        
        if analysis["analysis_factors"]["context_analysis"]["has_contrasting_elements"]:
            explanation_parts.append("Contains contrasting elements that may affect sentiment interpretation")
        
        if analysis["analysis_factors"]["linguistic_features"]["contains_negation"]:
            explanation_parts.append("Contains negations that could reverse sentiment")
        
        if analysis["analysis_factors"]["linguistic_features"]["contains_technical_terms"]:
            explanation_parts.append("Contains technical terms that might influence sentiment interpretation")
        
        if analysis["analysis_factors"]["linguistic_features"]["contains_conditionals"]:
            explanation_parts.append("Contains conditional statements that may modify the sentiment strength")
        
        if not explanation_parts:
            if original_sentiment == "neutral" and predicted_sentiment != "neutral":
                explanation_parts.append("Review contains positive/negative indicators despite being marked as neutral")
            elif original_sentiment != "neutral" and predicted_sentiment == "neutral":
                explanation_parts.append("Review lacks strong sentiment indicators despite non-neutral original sentiment")
        
        analysis["analysis_factors"]["explanation"] = ". ".join(explanation_parts)
        
        return analysis