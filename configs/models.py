from typing import Dict

class ModelConfig:
    """Configuration class for sentiment models"""
    SUPPORTED_MODELS = {
        'distilbert-sst2': {
            'name': 'distilbert-base-uncased-finetuned-sst-2-english',
            'type': 'binary',
            'mapping': {0: 'negative', 1: 'positive'},
            'description': 'DistilBERT model fine-tuned on SST-2 (fast, efficient)'
        },
        'nlptown-bert': {
            'name': 'nlptown/bert-base-multilingual-uncased-sentiment',
            'type': 'five-class',
            'mapping': {
                0: 'negative', 1: 'negative',  # Very negative & negative
                2: 'neutral',
                3: 'positive', 4: 'positive'   # Positive & very positive
            },
            'description': 'Multilingual BERT with 5-class sentiment'
        },
        'cardiffnlp-twitter': {
            'name': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
            'type': 'three-class',
            'mapping': {0: 'negative', 1: 'neutral', 2: 'positive'},
            'description': 'RoBERTa model fine-tuned on Twitter data with native neutral detection'
        }
    }

class DomainIndicators:
    """Domain-specific sentiment indicators"""
    INDICATORS = {
        'technology': {
                'positive': {'innovative', 'efficient', 'powerful', 'impressive', 'reliable'},
                'negative': {'slow', 'buggy', 'expensive', 'disappointing', 'unreliable'},
                'neutral_markers': {'average', 'standard', 'typical', 'expected'}
            },
            'software': {
                'positive': {'user-friendly', 'intuitive', 'fast', 'robust', 'feature-rich', 'versatile', 'stable', 'secure', 'efficient', 'scalable'},
                'negative': {'crashes', 'unresponsive', 'complicated', 'glitchy', 'slow', 'insecure', 'outdated', 'buggy', 'limited', 'inefficient'},
                'neutral_markers': {'adequate', 'functional', 'standard', 'acceptable', 'usable'}
            },
            'hotel': {
                'positive': {'luxurious', 'comfortable', 'clean', 'spacious', 'friendly staff', 'great service', 'convenient location', 'cozy', 'elegant', 'amenities'},
                'negative': {'dirty', 'noisy', 'uncomfortable', 'rude staff', 'poor service', 'overpriced', 'crowded', 'small rooms', 'inconvenient', 'unhygienic'},
                'neutral_markers': {'average', 'basic', 'standard', 'decent', 'adequate'}
            },
            'travel': {
                'positive': {'adventurous', 'exciting', 'breathtaking', 'relaxing', 'memorable', 'spectacular', 'unforgettable', 'scenic', 'enjoyable', 'fascinating'},
                'negative': {'boring', 'tiring', 'stressful', 'disappointing', 'dangerous', 'overrated', 'expensive', 'crowded', 'dull', 'tedious'},
                'neutral_markers': {'ordinary', 'mediocre', 'typical', 'expected', 'standard'}
            },
            'education': {
                'positive': {'informative', 'engaging', 'comprehensive', 'enlightening', 'inspirational', 'effective', 'supportive', 'innovative', 'challenging', 'rewarding'},
                'negative': {'boring', 'uninformative', 'confusing', 'ineffective', 'unhelpful', 'outdated', 'dull', 'frustrating', 'disorganized', 'stressful'},
                'neutral_markers': {'average', 'typical', 'standard', 'basic', 'mediocre'}
            },
            'ecommerce': {
                'positive': {'convenient', 'fast shipping', 'great deals', 'user-friendly', 'secure', 'reliable', 'efficient', 'wide selection', 'responsive', 'satisfactory'},
                'negative': {'delayed', 'poor customer service', 'fraudulent', 'difficult navigation', 'unreliable', 'damaged goods', 'overpriced', 'confusing', 'limited options', 'slow'},
                'neutral_markers': {'average', 'acceptable', 'standard', 'typical', 'satisfactory'}
            },
            'social media': {
                'positive': {'engaging', 'interactive', 'innovative', 'user-friendly', 'connective', 'fun', 'inspiring', 'entertaining', 'informative'},
                'negative': {'toxic', 'privacy concerns', 'cyberbullying', 'spam', 'fake news', 'unreliable', 'time-consuming', 'annoying ads', 'glitchy'},
                'neutral_markers': {'common', 'average', 'typical', 'standard', 'expected'}
            },
            'healthcare': {
                'positive': {'caring', 'professional', 'compassionate', 'knowledgeable', 'efficient', 'reliable', 'thorough', 'state-of-the-art', 'clean', 'responsive'},
                'negative': {'rude', 'unprofessional', 'inefficient', 'uncaring', 'dirty', 'long wait times', 'expensive', 'misdiagnosis', 'negligent', 'incompetent'},
                'neutral_markers': {'standard', 'average', 'typical', 'adequate', 'sufficient'}
            }
        }