from transformers import AutoModelForSequenceClassification, AutoTokenizer
from nltk.tokenize import word_tokenize

class SentimentValidator:
    """
    Advanced sentiment validation system that combines:
    - BERT-based sentiment predictions
    - Domain-specific sentiment indicators
    - Contrast marker detection
    - Confidence thresholding
    
    Designed to minimize false positives by only flagging high-confidence mismatches
    while accounting for domain context and linguistic nuances.
    """
    
    def __init__(self):
        # Load a lightweight BERT model fine-tuned for sentiment(or load a big boy one if you can run it)
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # High confidence threshold for flagging mismatches
        self.confidence_threshold = 0.95
        
        # Common contrast markers that often indicate neutral sentiment
        self.contrast_markers = {'but', 'however', 'although', 'though', 'while', 'yet'}
        
        # Add more neutral indicators
        self.neutral_indicators = {
            'adequate', 'adequately', 'average', 'basic', 'decent', 'fair', 'moderate', 
            'normal', 'ordinary', 'reasonable', 'standard', 'typical', 'usual',
            'performs adequately', 'works fine', 'meets expectations', 'as expected',
            'suitable for', 'acceptable', 'sufficient', 'satisfactory'
        }
        
        # Expand neutral indicators with more nuanced patterns
        self.neutral_indicators.update({
            'mixed feelings', 'balanced', 'middle ground', 'somewhat', 
            'relatively', 'fairly', 'neither', 'nor', 'compared to',
            'while', 'although', 'however', 'on one hand', 'on the other hand',
            'pros and cons', 'trade-off', 'trade off', 'compromise',
            'limited compared', 'basic but', 'simple but', 'decent but',
            'good enough', 'not great but', 'not bad but'
        })
        
        # Add neutral sentiment patterns
        self.neutral_patterns = {
            'performance_comparison': r'(?i)(compared|relative|versus|vs).*(?:newer|other|previous|similar)',
            'balanced_opinion': r'(?i)(while|although|however).*but',
            'moderate_intensity': r'(?i)(somewhat|fairly|relatively|quite|rather)\s\w+',
            'explicit_neutral': r'(?i)(neutral|mixed|balanced|middle ground|average|moderate)',
            'pros_cons': r'(?i)(pros.*cons|advantages.*disadvantages|benefits.*drawbacks)',
        }
        
        # Common domain-specific positive/negative indicators(gpt , no clue if these are actually good enough????)
        self.domain_indicators = {
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
    
    def _preprocess_text(self, text: str):
        """Basic text preprocessing"""
        return text.lower().strip()
    
    def _check_domain_indicators(self, text: str, domain: str):
        """Check for domain-specific sentiment indicators"""
        if not domain or domain not in self.domain_indicators:
            return None
        
        text = self._preprocess_text(text)
        domain_info = self.domain_indicators[domain]
        
        positive_count = sum(1 for word in domain_info['positive'] if word in text)
        negative_count = sum(1 for word in domain_info['negative'] if word in text)
        neutral_count = sum(1 for word in domain_info['neutral_markers'] if word in text)
        
        if positive_count > negative_count and positive_count > neutral_count:
            return 'positive'
        elif negative_count > positive_count and negative_count > neutral_count:
            return 'negative'
        elif neutral_count > 0:
            return 'neutral'
        return None
    
    def _detect_neutral_patterns(self, text: str) -> bool:
        """Check for linguistic patterns indicating neutral sentiment"""
        import re
        
        text = text.lower()
        
        # Check for balanced statements with positive and negative aspects
        has_positive = any(word in text for word in self.domain_indicators.get('general', {}).get('positive', set()))
        has_negative = any(word in text for word in self.domain_indicators.get('general', {}).get('negative', set()))
        has_contrast = any(marker in text for marker in self.contrast_markers)
        
        if has_positive and has_negative and has_contrast:
            return True
            
        # Check for neutral patterns
        for pattern in self.neutral_patterns.values():
            if re.search(pattern, text):
                return True
                
        return False
    
    def validate_sentiment(self, text: str, labeled_sentiment: str, domain: str = None):
        """Enhanced sentiment validation with better neutral detection"""
        domain_sentiment = self._check_domain_indicators(text, domain)
        
        # Get BERT model prediction
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.model(**inputs)
        probs = outputs.logits.softmax(dim=-1)
        predicted_class = outputs.logits.argmax().item()
        confidence = probs[0][predicted_class].item()
        
        # Check for neutral indicators
        has_neutral_indicators = any(indicator in text.lower() for indicator in self.neutral_indicators)
        has_neutral_patterns = self._detect_neutral_patterns(text)
        
        # Determine predicted sentiment with neutral consideration
        if has_neutral_indicators or has_neutral_patterns:
            predicted_sentiment = "neutral"
            # Adjust confidence for neutral predictions
            confidence = min(confidence, 0.8)  # Cap confidence for neutral predictions
        else:
            predicted_sentiment = "positive" if predicted_class == 1 else "negative"
        
        # Determine if there's a mismatch
        is_mismatch = False
        if labeled_sentiment == "neutral":
            # For neutral labeled sentiments, only flag as mismatch if we're very confident
            # it's strongly positive or negative
            is_mismatch = (confidence > self.confidence_threshold and 
                          not has_neutral_indicators and 
                          not has_neutral_patterns)
        else:
            # For positive/negative labeled sentiments, flag if prediction differs
            is_mismatch = (predicted_sentiment != labeled_sentiment and 
                          confidence >= self.confidence_threshold)
        
        return {
            'is_mismatch': is_mismatch,
            'predicted': predicted_sentiment,
            'confidence': confidence,
            'domain_sentiment': domain_sentiment,
            'has_neutral_indicators': has_neutral_indicators,
            'has_neutral_patterns': has_neutral_patterns
        }