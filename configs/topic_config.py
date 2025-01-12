"""
Topic modeling configuration based on research standards.
References:
- Röder, Michael & Both, Andreas & Hinneburg, Alexander. (2015). Exploring the Space of Topic Coherence Measures. WSDM 2015 - Proceedings of the 8th ACM International Conference on Web Search and Data Mining. 399-408. 10.1145/2684822.2685324. 
- David Mimno, Hanna Wallach, Edmund Talley, Miriam Leenders, and Andrew McCallum. 2011. Optimizing Semantic Coherence in Topic Models. In Proceedings of the 2011 Conference on Empirical Methods in Natural Language Processing, pages 262–272, Edinburgh, Scotland, UK.. Association for Computational Linguistics.
- Wallach, Hanna & Murray, Iain & Salakhutdinov, Ruslan & Mimno, David. (2009). Evaluation methods for topic models. Proceedings of the 26th International Conference On Machine Learning, ICML 2009. 382. 139. 10.1145/1553374.1553515. 
"""

TOPIC_CONFIG = {
    'coherence_measure': 'c_v',  # Following Röder et al. (2015)
    'min_topics': 2,
    'max_topics': 20,
    'coherence_thresholds': {
        'excellent': 0.70,  # Based on benchmark datasets
        'good': 0.55,
        'acceptable': 0.40,
        'poor': 0.0
    },
    'perplexity_thresholds': {  # Following Wallach et al. (2009)
        'excellent': -7.0,
        'good': -8.0,
        'acceptable': -9.0,
        'poor': float('-inf')
    }
}