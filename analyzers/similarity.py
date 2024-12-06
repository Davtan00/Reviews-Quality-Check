from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging
from tqdm import tqdm
from typing import List, Dict, Set, Tuple
from config import SIMILARITY_THRESHOLD, ENABLE_TIERED_SIMILARITY, SIMILARITY_TIERS
from nltk.util import ngrams
from collections import defaultdict
import json
from datetime import datetime
import os

class SophisticatedSimilarityAnalyzer:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.batch_size = 500
        self.similarity_batch_size = 1000
        self.exact_match_max_length = 50
        self.ngram_size = 3
        self.enable_tiered_analysis = ENABLE_TIERED_SIMILARITY
        self.similarity_tiers = SIMILARITY_TIERS

    def get_ngram_similarity(self, text1: str, text2: str) -> float:
        """Calculate n-gram based similarity between two texts"""
        def preprocess_text(text: str) -> str:
            # Convert to lowercase and remove extra whitespace
            text = text.lower().strip()
            # Replace multiple spaces with single space
            text = ' '.join(text.split())
            return text
            
        def get_ngrams(text: str) -> Set[str]:
            # Preprocess text
            text = preprocess_text(text)
            # Split into words
            words = text.split()
            # Generate character-level n-grams for each word
            word_ngrams = set()
            for word in words:
                # Add the word itself
                word_ngrams.add(word)
                # Add character n-grams
                if len(word) >= self.ngram_size:
                    for i in range(len(word) - self.ngram_size + 1):
                        word_ngrams.add(word[i:i + self.ngram_size])
            # Add word-level n-grams
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

    def process_similarity_batch(self, similarity_matrix: np.ndarray, start_idx: int, texts: List[str], duplicates: Set[Tuple[int, int]], pbar: tqdm) -> Tuple[List[Dict], Dict[str, List[Dict]]]:
        """Process a batch of the similarity matrix with tiered analysis"""
        similar_pairs = []
        tiered_results = {
            'tier1': [],  # Embedding similarity > 0.98 (identical semantic content)
            'tier2': [],  # Combined similarity > 0.92 (for deduplication)
            'tier3': []   # N-gram similarity > 0.85 (structural similarity)
        }
        
        matrix_size = similarity_matrix.shape[0]
        comparisons_total = (matrix_size * (matrix_size - 1)) // 2
        
        comparison_count = 0
        for i in range(matrix_size):
            global_i = start_idx + i
            for j in range(i + 1, matrix_size):
                global_j = start_idx + j
                comparison_count += 1
                
                if comparison_count % 1000 == 0:
                    pbar.set_postfix({'comparisons': f"{comparison_count}/{comparisons_total}"}, refresh=True)
                
                if (global_i, global_j) not in duplicates:
                    embedding_sim = float(similarity_matrix[i, j])
                    ngram_sim = self.get_ngram_similarity(texts[global_i], texts[global_j])
                    combined_sim = (embedding_sim + ngram_sim) / 2
                    
                    similarity_info = {
                        'index1': global_i,
                        'index2': global_j,
                        'text1': texts[global_i],
                        'text2': texts[global_j],
                        'embedding_similarity': embedding_sim,
                        'ngram_similarity': ngram_sim,
                        'combined_similarity': combined_sim
                    }
                    
                    # Check for semantic duplicates (embedding similarity)
                    if embedding_sim >= self.similarity_tiers['tier1']:  # > 0.98
                        tiered_results['tier1'].append(similarity_info)
                        similar_pairs.append(similarity_info)
                        duplicates.add((global_i, global_j))
                    
                    # Check for combined similarity (deduplication)
                    elif combined_sim >= self.similarity_tiers['tier2']:  # > 0.92
                        tiered_results['tier2'].append(similarity_info)
                        similar_pairs.append(similarity_info)
                        duplicates.add((global_i, global_j))
                    
                    # Check for structural similarity (n-grams)
                    elif ngram_sim >= self.similarity_tiers['tier3']:  # > 0.85
                        tiered_results['tier3'].append(similarity_info)
        
        return similar_pairs, tiered_results

    def analyze_similarity(self, texts: List[str]) -> Dict:
        """Analyze text similarity with optional tiered analysis"""
        duplicates = set()
        exact_duplicates = []
        similar_pairs = []
        all_tiered_results = {
            'tier1': [],
            'tier2': [],
            'tier3': []
        }
        
        # First pass: exact matching for short texts
        logging.info("Performing exact matching for short texts...")
        with tqdm(total=len(texts), desc="Exact matching") as pbar:
            seen_texts = defaultdict(list)
            for i, text in enumerate(texts):
                cleaned_text = text.lower().strip()
                if len(cleaned_text.split()) <= self.exact_match_max_length:
                    seen_texts[cleaned_text].append(i)
                pbar.update(1)
        
        # Process exact duplicates
        exact_duplicates = []
        for indices in seen_texts.values():
            if len(indices) > 1:
                group = [{'text': texts[i], 'index': i} for i in indices]
                exact_duplicates.append(group)
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        duplicates.add((indices[i], indices[j]))
        
        # Second pass: semantic similarity with batched processing
        embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Computing embeddings"):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self.model.encode(batch, show_progress_bar=False)
            embeddings.extend(batch_embeddings)
        embeddings = np.array(embeddings)
        
        # Process similarity in batches
        logging.info("Processing similarity matrices...")
        with tqdm(total=len(texts), desc="Similarity analysis") as pbar:
            for i in range(0, len(texts), self.similarity_batch_size):
                batch_end = min(i + self.similarity_batch_size, len(texts))
                similarity_matrix = cosine_similarity(embeddings[i:batch_end], embeddings[i:batch_end])
                batch_pairs, batch_tiered = self.process_similarity_batch(
                    similarity_matrix, i, texts, duplicates, pbar
                )
                similar_pairs.extend(batch_pairs)
                
                # Accumulate tiered results
                if self.enable_tiered_analysis:
                    for tier in all_tiered_results:
                        all_tiered_results[tier].extend(batch_tiered[tier])
                
                pbar.update(batch_end - i)
        
        # Save tiered analysis results if enabled
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
        """Analyze token overlap between two sets of texts"""
        tokens1 = set(" ".join(texts1).split())
        tokens2 = set(" ".join(texts2).split())
        
        overlap = tokens1.intersection(tokens2)
        
        return {
            'overlap_count': len(overlap),
            'overlap_ratio': len(overlap) / len(tokens1.union(tokens2)),
            'unique_tokens1': len(tokens1),
            'unique_tokens2': len(tokens2)
        }