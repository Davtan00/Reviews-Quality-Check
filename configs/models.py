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
    """Domain-specific sentiment indicators with expanded keyword sets."""

    INDICATORS = {
        'technology': {
            'positive': {
                'innovative', 'efficient', 'powerful', 'impressive', 'reliable',
                'cutting-edge', 'scalable', 'disruptive', 'well-designed',
                'user-centric', 'robust', 'secure', 'sleek', 'lightweight',
                'groundbreaking', 'revolutionary', 'high-performance'
            },
            'negative': {
                'slow', 'buggy', 'expensive', 'disappointing', 'unreliable',
                'outdated', 'clunky', 'inefficient', 'overhyped', 'vulnerable',
                'data privacy concerns', 'fragile', 'resource-heavy', 'obsolete'
            },
            'neutral_markers': {
                'average', 'standard', 'typical', 'expected', 'common',
                'generic', 'industry-standard', 'normal', 'ordinary'
            }
        },
        'software': {
            'positive': {
                'user-friendly', 'intuitive', 'fast', 'robust', 'feature-rich',
                'versatile', 'stable', 'secure', 'efficient', 'scalable',
                'lightweight', 'modern', 'responsive', 'well-documented',
                'high-quality', 'streamlined', 'flexible'
            },
            'negative': {
                'crashes', 'unresponsive', 'complicated', 'glitchy', 'slow',
                'insecure', 'outdated', 'buggy', 'limited', 'inefficient',
                'laggy', 'overly complex', 'resource-intensive', 'unstable'
            },
            'neutral_markers': {
                'adequate', 'functional', 'standard', 'acceptable', 'usable',
                'basic', 'typical', 'common', 'ordinary'
            }
        },
        'hotel': {
            'positive': {
                'luxurious', 'comfortable', 'clean', 'spacious', 'friendly staff',
                'great service', 'convenient location', 'cozy', 'elegant',
                'amenities', 'welcoming', 'serene', 'well-maintained',
                'beautiful decor'
            },
            'negative': {
                'dirty', 'noisy', 'uncomfortable', 'rude staff', 'poor service',
                'overpriced', 'crowded', 'small rooms', 'inconvenient',
                'unhygienic', 'neglected', 'musty', 'unfriendly atmosphere'
            },
            'neutral_markers': {
                'average', 'basic', 'standard', 'decent', 'adequate',
                'typical', 'moderate', 'ordinary'
            }
        },
        'travel': {
            'positive': {
                'adventurous', 'exciting', 'breathtaking', 'relaxing', 'memorable',
                'spectacular', 'unforgettable', 'scenic', 'enjoyable',
                'fascinating', 'refreshing', 'exotic', 'inspiring'
            },
            'negative': {
                'boring', 'tiring', 'stressful', 'disappointing', 'dangerous',
                'overrated', 'expensive', 'crowded', 'dull', 'tedious',
                'hectic', 'underwhelming'
            },
            'neutral_markers': {
                'ordinary', 'mediocre', 'typical', 'expected', 'standard',
                'common', 'average'
            }
        },
        'education': {
            'positive': {
                'informative', 'engaging', 'comprehensive', 'enlightening',
                'inspirational', 'effective', 'supportive', 'innovative',
                'challenging', 'rewarding', 'stimulating', 'thought-provoking'
            },
            'negative': {
                'boring', 'uninformative', 'confusing', 'ineffective', 'unhelpful',
                'outdated', 'dull', 'frustrating', 'disorganized', 'stressful',
                'tedious', 'demotivating'
            },
            'neutral_markers': {
                'average', 'typical', 'standard', 'basic', 'mediocre',
                'common', 'normal'
            }
        },
        'ecommerce': {
            'positive': {
                'convenient', 'fast shipping', 'great deals', 'user-friendly',
                'secure', 'reliable', 'efficient', 'wide selection',
                'responsive', 'satisfactory', 'seamless checkout', 'reputable',
                'personalized recommendations', 'competitive pricing',
                'fast delivery', 'trusted brand', 'hassle-free returns',
                'variety of payment options', 'transparent policies',
                'product variety', 'discounts', 'excellent packaging',
                'loyalty rewards'
            },
            'negative': {
                'delayed', 'poor customer service', 'fraudulent',
                'difficult navigation', 'unreliable', 'damaged goods',
                'overpriced', 'confusing', 'limited options', 'slow',
                'hidden fees', 'misleading product descriptions',
                'unresponsive seller', 'compromised data', 'faulty items',
                'no tracking', 'unexpected charges', 'clunky checkout'
            },
            'neutral_markers': {
                'average', 'acceptable', 'standard', 'typical', 'satisfactory',
                'typical shipping times', 'average offers', 'standard selection',
                'expected product range', 'common disclaimers', 'routine process'
            }
        },
        'social media': {
            'positive': {
                'engaging', 'interactive', 'innovative', 'user-friendly',
                'connective', 'fun', 'inspiring', 'entertaining', 'informative',
                'immersive', 'community-driven'
            },
            'negative': {
                'toxic', 'privacy concerns', 'cyberbullying', 'spam', 'fake news',
                'unreliable', 'time-consuming', 'annoying ads', 'glitchy',
                'data harvesting'
            },
            'neutral_markers': {
                'common', 'average', 'typical', 'standard', 'expected',
                'everyday', 'usual'
            }
        },
        'healthcare': {
            'positive': {
                'caring', 'professional', 'compassionate', 'knowledgeable',
                'efficient', 'reliable', 'thorough', 'state-of-the-art', 'clean',
                'responsive', 'holistic', 'empathetic', 'organized',
                'well-staffed', 'advanced equipment', 'pain-free', 'life-saving',
                'top-notch', 'well-maintained', 'safe', 'reassuring',
                'transparent', 'trustworthy', 'cutting-edge procedures',
                'patient-centered', 'comprehensive care'
            },
            'negative': {
                'rude', 'unprofessional', 'inefficient', 'uncaring', 'dirty',
                'long wait times', 'expensive', 'misdiagnosis', 'negligent',
                'incompetent', 'traumatic', 'haphazard', 'unattentive staff',
                'exorbitant costs', 'lack of resources', 'harsh environment',
                'slow response', 'impersonal treatment'
            },
            'neutral_markers': {
                'standard', 'average', 'typical', 'adequate', 'sufficient',
                'routine', 'regulated', 'regular check-ups', 'common practice',
                'ordinary', 'normal procedures'
            }
        }
    }
