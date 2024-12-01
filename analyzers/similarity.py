from sentence_transformers import SentenceTransformer
from nltk import ngrams
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
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
        self.optimal_batch_size = 2000
    
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
    
    def analyze_similarity(self, texts, batch_size=None):
        """
        Analyze similarity with hardware-optimized batching
        """
        batch_size = batch_size or self.optimal_batch_size
        
        if len(texts) > batch_size:
            print(f"Processing {len(texts)} texts in batches...")
            results = np.zeros((len(texts), len(texts)))
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                batch_result = self._process_batch(batch)
                results[i:i+batch_size, i:i+batch_size] = batch_result['similarity_matrix']
            return self._merge_results([{'similarity_matrix': results}])
        return self._process_batch(texts)
    # TODO: Stop wasting time on this and just make an approach that works for most in most cases
    def _process_batch(self, texts):
        """Process a batch with optimized numpy operations"""
        # Pre-allocate arrays for better memory usage
        embeddings = self._get_embeddings(texts)
        semantic_similarities = cosine_similarity(embeddings)
        
        # Use numpy operations instead of nested loops
        indices = np.triu_indices(len(texts), k=1)
        semantic_sims = semantic_similarities[indices]
        
        # Initialize results with numpy arrays
        high_similarity_pairs = []
        mask = semantic_sims >= self.semantic_threshold * 0.8
        
        if np.any(mask):
            for idx in np.where(mask)[0]:
                i, j = indices[0][idx], indices[1][idx]
                ngram_sim = self._get_ngram_similarity(texts[i], texts[j])
                combined_sim = (semantic_sims[idx] * 0.7) + (ngram_sim * 0.3)
                
                if combined_sim >= self.semantic_threshold:
                    high_similarity_pairs.append({
                        'index1': int(i),
                        'index2': int(j),
                        'similarity': float(combined_sim),
                        'semantic_similarity': float(semantic_sims[idx]),
                        'ngram_similarity': float(ngram_sim)
                    })
        
        return {
            'average_similarity': float(np.mean(semantic_sims)),
            'high_similarity_pairs': high_similarity_pairs,
            'similarity_matrix': semantic_similarities
        }