"""
Topic modeling configuration based on research standards.
References:
- Röder et al. (2015): Topic coherence measures
- Mimno et al. (2011): Semantic coherence optimization
- Wallach et al. (2009): Topic model evaluation
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