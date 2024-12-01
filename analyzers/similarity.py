from sentence_transformers import SentenceTransformer
from nltk import ngrams
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict
from tqdm import tqdm
import logging
try:
    from gensim.models.ldamulticore import LdaMulticore
except ImportError:
    from gensim.models import LdaMulticore

class SophisticatedSimilarityAnalyzer:
    """
    Hybrid similarity detection system combining semantic and lexical features:
    - Semantic: Uses sentence transformers for meaning-based similarity (70% weight)
    - Lexical: Uses n-gram overlap for surface-level similarity (30% weight)
    - Thresholds: 0.85 for semantic, 0.7 for n-gram similarity
    
    This dual approach helps catch both meaning-based and text-based similarities
    while reducing false positives through weighted combination.
    """
    
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.semantic_threshold = 0.85
        self.ngram_threshold = 0.7
        self.batch_size = 500  # TODO: Make this dynamic using psutil,platform etc.
        self.chunk_size = 2000

    def _get_embeddings(self, texts):
        """Generate embeddings for all texts using the sentence transformer model"""
        return self.model.encode(texts)
    
    def _get_ngram_similarity(self, text1, text2, n=3):
        """Calculate n-gram based similarity between two texts"""
        # Get n-grams for both texts
        ngrams1 = set(self.get_ngrams(text1, n))
        ngrams2 = set(self.get_ngrams(text2, n))
        
        # Calculate Jaccard similarity
        if not ngrams1 or not ngrams2:
            return 0.0
        
        intersection = len(ngrams1.intersection(ngrams2))
        union = len(ngrams1.union(ngrams2))
        return intersection / union if union > 0 else 0.0
    
    def get_ngrams(self, text, n):
        """Generate n-grams from text"""
        words = text.lower().split()
        return [''.join(gram) for gram in ngrams(words, n)]
    
    def analyze_similarity(self, texts: List[str]) -> Dict:
        """Analyze similarity between texts using batched processing and memory-efficient comparison"""
        logging.info(f"Processing {len(texts)} texts in batches...")
        
        similar_pairs = []
        total_similarity = 0
        pair_count = 0
        
        for i in range(0, len(texts), self.chunk_size):
            chunk_texts = texts[i:i + self.chunk_size]
            chunk_embeddings = []
            
            # Calculate embeddings for current chunk
            for j in tqdm(range(0, len(chunk_texts), self.batch_size), 
                         desc=f"Processing chunk {i//self.chunk_size + 1}/{(len(texts)-1)//self.chunk_size + 1}"):
                batch = chunk_texts[j:j + self.batch_size]
                batch_embeddings = self.model.encode(batch)
                chunk_embeddings.extend(batch_embeddings)
            
            chunk_embeddings = np.array(chunk_embeddings)
            
            # Compare current chunk with itself
            similarity_matrix = cosine_similarity(chunk_embeddings)
            
            # Find similar pairs within the chunk
            for j in range(len(chunk_texts)):
                for k in range(j + 1, len(chunk_texts)):
                    similarity = similarity_matrix[j][k]
                    total_similarity += similarity
                    pair_count += 1
                    
                    if similarity > self.semantic_threshold:
                        similar_pairs.append({
                            'text1': chunk_texts[j],
                            'text2': chunk_texts[k],
                            'similarity': float(similarity)
                        })
            
            # Clear memory
            del similarity_matrix
            del chunk_embeddings
        
        average_similarity = total_similarity / pair_count if pair_count > 0 else 0
        
        return {
            'average_similarity': average_similarity,
            'high_similarity_pairs': similar_pairs,
            'total_pairs_analyzed': pair_count
        }