from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging
from tqdm import tqdm
from typing import List, Dict
from config import SIMILARITY_THRESHOLD

class SophisticatedSimilarityAnalyzer:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.batch_size = 500  # TODO: Make this dynamic using psutil,platform etc.
        self.chunk_size = 2000
        
    def analyze_similarity(self, texts: List[str]) -> Dict:
        """Analyze similarity between texts using batched processing"""
        logging.info(f"Processing {len(texts)} texts in batches...")
        
        # Calculate embeddings in batches
        embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Processing chunk 1/1"):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self.model.encode(batch)
            embeddings.extend(batch_embeddings)
            
        # Convert to numpy array for faster computation
        embeddings = np.array(embeddings)
        
        # Calculate cosine similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Find similar pairs above threshold
        similar_pairs = []
        total_similarity = 0
        pair_count = 0
        
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                similarity = float(similarity_matrix[i][j])
                total_similarity += similarity
                pair_count += 1
                
                if similarity > SIMILARITY_THRESHOLD:
                    similar_pairs.append({
                        'text1': texts[i],
                        'text2': texts[j],
                        'similarity': similarity,
                        'index1': i,
                        'index2': j
                    })
        
        average_similarity = total_similarity / pair_count if pair_count > 0 else 0
        
        return {
            'similarity_matrix': similarity_matrix.tolist(),
            'similar_pairs': similar_pairs,
            'average_similarity': average_similarity,
            'total_pairs_analyzed': pair_count
        }