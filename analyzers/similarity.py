import logging
import os
import json
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Set, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from nltk.util import ngrams

from config import (
    SIMILARITY_THRESHOLD, 
    ENABLE_TIERED_SIMILARITY, 
    SIMILARITY_TIERS
)

class SophisticatedSimilarityAnalyzer:
    """
    A class for sophisticated text similarity analysis using Sentence Transformers,
    combined with an optional n-gram overlap measure.
    """
    
    def __init__(self):
        """
        Initialize the analyzer.
        
        Attempts to load the SentenceTransformer model on the best available hardware:
        GPU or Apple’s MPS if possible. If not, falls back to CPU.
        
        Also sets:
          - batch_size: chunk size for computing embeddings
          - similarity_batch_size: chunk size when processing similarity matrix
          - exact_match_max_length: max words in a text snippet to consider for exact duplication
          - ngram_size: size of character-based n-grams used for get_ngram_similarity
          - enable_tiered_analysis: whether to perform tiered similarity checks
          - similarity_tiers: thresholds for each tier
        """
        # Attempt to specify a device if GPU or MPS is available
        self.device = "cpu"
        try:
            import torch
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
        except ImportError:
            pass  # If torch not installed, fallback to CPU only
        
        logging.info(f"Initializing SentenceTransformer on device: {self.device}")
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        
        self.batch_size = 500
        self.similarity_batch_size = 1000
        self.exact_match_max_length = 50
        self.ngram_size = 3
        self.enable_tiered_analysis = ENABLE_TIERED_SIMILARITY
        self.similarity_tiers = SIMILARITY_TIERS

    def get_ngram_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate a combined word and character n-gram based similarity between two texts.
        
        This function:
          1. Converts both texts to lowercase and strips extra whitespace.
          2. Splits into words, then collects character-level n-grams for each word,
             as well as word-level n-grams (if length of words >= ngram_size).
          3. Returns Jaccard similarity (intersection/union) over these sets of n-grams.
        
        Args:
            text1: First text to compare.
            text2: Second text to compare.
        
        Returns:
            A float in [0.0, 1.0], where higher means more overlap in n-grams.
        """
        def preprocess_text(text: str) -> str:
            text = text.lower().strip()
            # Replace multiple spaces with a single space
            return ' '.join(text.split())
            
        def get_ngrams(text: str) -> Set[str]:
            txt = preprocess_text(text)
            words = txt.split()
            word_ngrams = set()
            
            # For each word, add the whole word + any character-level n-grams
            for word in words:
                word_ngrams.add(word)
                if len(word) >= self.ngram_size:
                    for i in range(len(word) - self.ngram_size + 1):
                        word_ngrams.add(word[i:i + self.ngram_size])
            
            # Add word-level n-grams (e.g., 3 consecutive words)
            if len(words) >= self.ngram_size:
                word_ngrams.update(' '.join(gram) for gram in ngrams(words, self.ngram_size))
            
            return word_ngrams
            
        ngrams1 = get_ngrams(text1)
        ngrams2 = get_ngrams(text2)
        
        if not ngrams1 or not ngrams2:
            return 0.0
        
        intersection = len(ngrams1.intersection(ngrams2))
        union = len(ngrams1.union(ngrams2))
        return intersection / union if union > 0 else 0.0

    def process_similarity_batch(
        self, 
        similarity_matrix: np.ndarray, 
        start_idx: int, 
        texts: List[str], 
        duplicates: Set[Tuple[int, int]], 
        pbar: tqdm
    ) -> Tuple[List[Dict], Dict[str, List[Dict]]]:
        """
        Process a sub-matrix of the full similarity matrix and gather results.
        
        1. Combines embedding-based similarity with the n-gram measure.
        2. Checks if combined similarity > SIMILARITY_THRESHOLD => duplicates.
        3. Optionally performs tiered analysis and stores similarity info in the relevant tier.
        
        Args:
            similarity_matrix: A square sub-matrix of embedding similarities for a batch.
            start_idx: The global starting index of this batch in the full data.
            texts: The full list of texts being analyzed.
            duplicates: A set of pairs (i, j) already marked as duplicates, so they won't be re-processed.
            pbar: A tqdm progress bar for updates.
        
        Returns:
            similar_pairs: A list of dicts describing discovered pairs of highly similar texts.
            tiered_results: A dict with 'tier1', 'tier2', 'tier3' keys, each storing a list of pairs 
                            that meet that tier's threshold.
        """
        similar_pairs = []
        tiered_results = {
            'tier1': [],
            'tier2': [],
            'tier3': []
        }
        
        matrix_size = similarity_matrix.shape[0]
        comparisons_total = (matrix_size * (matrix_size - 1)) // 2
        
        comparison_count = 0
        for i in range(matrix_size):
            global_i = start_idx + i
            for j in range(i + 1, matrix_size):
                global_j = start_idx + j
                comparison_count += 1
                
                # Periodically update the progress bar with the number of comparisons
                if comparison_count % 1000 == 0:
                    pbar.set_postfix({'comparisons': f"{comparison_count}/{comparisons_total}"}, refresh=True)
                
                # If not already found to be duplicates
                if (global_i, global_j) not in duplicates:
                    # Embedding similarity
                    sim = float(similarity_matrix[i, j])
                    # N-gram similarity
                    ngram_sim = self.get_ngram_similarity(texts[global_i], texts[global_j])
                    # Combined measure
                    combined_sim = (sim + ngram_sim) / 2.0
                    
                    # If combined sim is above threshold => consider them duplicates
                    if combined_sim > SIMILARITY_THRESHOLD:
                        similar_pair = {
                            'index1': global_i,
                            'index2': global_j,
                            'text1': texts[global_i],
                            'text2': texts[global_j],
                            'similarity': combined_sim
                        }
                        similar_pairs.append(similar_pair)
                        duplicates.add((global_i, global_j))
                    
                    # If tiered analysis is enabled, categorize the pair
                    if self.enable_tiered_analysis:
                        similarity_info = {
                            'index1': global_i,
                            'index2': global_j,
                            'text1': texts[global_i],
                            'text2': texts[global_j],
                            'embedding_similarity': sim,
                            'ngram_similarity': ngram_sim,
                            'combined_similarity': combined_sim
                        }
                        
                        # Check tiers in descending order
                        if combined_sim >= self.similarity_tiers['tier1']:
                            tiered_results['tier1'].append(similarity_info)
                        elif combined_sim >= self.similarity_tiers['tier2']:
                            tiered_results['tier2'].append(similarity_info)
                        elif combined_sim >= self.similarity_tiers['tier3']:
                            tiered_results['tier3'].append(similarity_info)
        
        return similar_pairs, tiered_results

    def analyze_similarity(self, texts: List[str]) -> Dict:
        """
        Analyze text similarity with optional tiered analysis.
        
        Steps:
         1. Identify exact duplicates among short texts.
         2. Compute embeddings in batches.
         3. Partition embeddings for pairwise similarity in sub-batches (cosine).
         4. Combine with n-gram measure => final similarity.
         5. Collect results, optionally do tiered analysis, and optionally store results in a JSON.
        
        Args:
            texts: The list of texts to be analyzed.
        
        Returns:
            A dictionary with:
              - 'exact_duplicates': list of groups, each group containing dicts with 'text' and 'index'
              - 'similar_pairs': list of dicts with 'index1', 'index2', 'text1', 'text2', 'similarity'
        """
        duplicates = set()
        exact_duplicates = []
        similar_pairs = []
        all_tiered_results = {
            'tier1': [],
            'tier2': [],
            'tier3': []
        }
        
        logging.info("Performing exact matching for short texts...")
        with tqdm(total=len(texts), desc="Exact matching") as pbar:
            seen_texts = defaultdict(list)
            for i, text in enumerate(texts):
                cleaned_text = text.lower().strip()
                # If the text is short enough, consider exact matching
                if len(cleaned_text.split()) <= self.exact_match_max_length:
                    seen_texts[cleaned_text].append(i)
                pbar.update(1)
        
        # Identify exact duplicates
        for indices in seen_texts.values():
            if len(indices) > 1:
                group = [{'text': texts[i], 'index': i} for i in indices]
                exact_duplicates.append(group)
                # Mark them as duplicates except for the first occurrence
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        duplicates.add((indices[i], indices[j]))
        
        # Compute embeddings in batches
        embeddings = []
        logging.info("Computing embeddings in batches...")
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding Batches"):
            batch = texts[i:i + self.batch_size]
            # 'model.encode' automatically uses the device set in __init__
            batch_embeddings = self.model.encode(batch, show_progress_bar=False)
            embeddings.extend(batch_embeddings)
        embeddings = np.array(embeddings)
        
        # Partition embeddings for pairwise similarity
        logging.info("Processing similarity matrices...")
        with tqdm(total=len(texts), desc="Similarity analysis") as pbar:
            for i in range(0, len(texts), self.similarity_batch_size):
                batch_end = min(i + self.similarity_batch_size, len(texts))
                similarity_matrix = cosine_similarity(
                    embeddings[i:batch_end], 
                    embeddings[i:batch_end]
                )
                
                batch_pairs, batch_tiered = self.process_similarity_batch(
                    similarity_matrix, i, texts, duplicates, pbar
                )
                similar_pairs.extend(batch_pairs)
                
                # Accumulate tiered results if enabled
                if self.enable_tiered_analysis:
                    for tier in all_tiered_results:
                        all_tiered_results[tier].extend(batch_tiered[tier])
                
                # Update the progress bar
                pbar.update(batch_end - i)
        
        # Optionally save tiered analysis results
        if self.enable_tiered_analysis:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f'similarity_tiers_{timestamp}.json'
            
            tiered_analysis = {
                'timestamp': timestamp,
                'total_texts_analyzed': len(texts),
                'tier_thresholds': self.similarity_tiers,
                'results': {
                    tier: {
                        'count': len(results),
                        'pairs': results
                    }
                    for tier, results in all_tiered_results.items()
                }
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(tiered_analysis, f, indent=2, ensure_ascii=False)
            logging.info(f"Saved tiered similarity analysis to: {output_file}")
        
        return {
            'exact_duplicates': exact_duplicates,
            'similar_pairs': similar_pairs
        }

    def analyze_token_overlap(self, texts1: List[str], texts2: List[str]) -> Dict[str, int]:
        """
        Analyze token overlap between two sets of texts by:
          1. Merging each list of texts into one large string.
          2. Splitting on whitespace to get tokens.
          3. Computing intersection and union of token sets.
        
        Args:
            texts1: A list of strings.
            texts2: Another list of strings.
        
        Returns:
            A dict with:
              - 'overlap_count': int, number of common tokens
              - 'overlap_ratio': float, ratio = overlap / union
              - 'unique_tokens1': int, # unique tokens in texts1
              - 'unique_tokens2': int, # unique tokens in texts2
        """
        tokens1 = set(" ".join(texts1).split())
        tokens2 = set(" ".join(texts2).split())
        
        overlap = tokens1.intersection(tokens2)
        union_size = len(tokens1.union(tokens2))
        overlap_ratio = (len(overlap) / union_size) if union_size > 0 else 0.0
        
        return {
            'overlap_count': len(overlap),
            'overlap_ratio': overlap_ratio,
            'unique_tokens1': len(tokens1),
            'unique_tokens2': len(tokens2)
        }
