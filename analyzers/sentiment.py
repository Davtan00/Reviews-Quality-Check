from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Dict, Union, List
import logging
from configs.models import ModelConfig, DomainIndicators
from nltk import word_tokenize
import re

class SentimentValidator:
    """Advanced sentiment validation system with configurable models"""
    
    def __init__(self, model_key: str = 'distilbert-sst2'):
        """
        Initialize the sentiment validator with a specific model.
        
        Args:
            model_key: Key from ModelConfig.SUPPORTED_MODELS
        """
        if model_key not in ModelConfig.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model key. Choose from: {list(ModelConfig.SUPPORTED_MODELS.keys())}")
            
        self.model_config = ModelConfig.SUPPORTED_MODELS[model_key]
        self.model_type = self.model_config['type']
        self.label_mapping = self.model_config['mapping']
        
        logging.info(f"Loading sentiment model: {self.model_config['name']}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config['name'])
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_config['name'])
        
        self.confidence_thresholds = self._get_default_thresholds()
        self.domain_indicators = DomainIndicators.INDICATORS
        
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

    def _get_default_thresholds(self) -> Dict[str, float]:
        """Get default confidence thresholds based on model type"""
        if self.model_type == 'binary':
            return {
                "neutral": 0.85,
                "positive": 0.90,
                "negative": 0.90,
                "default": 0.95
            }
        elif self.model_type == 'three-class':
            return {
                "neutral": 0.75,  # Lower threshold for native neutral detection
                "positive": 0.85,
                "negative": 0.85,
                "default": 0.90
            }
        else:  # five-class
            return {
                "neutral": 0.70,
                "positive": 0.80,
                "negative": 0.80,
                "default": 0.85
            }

    def _map_model_output(self, predicted_class: int, confidence: float) -> Dict[str, Union[str, float]]:
        """Map model output to standardized sentiment with confidence"""
        predicted_sentiment = self.label_mapping.get(predicted_class, 'neutral')
        
        # Adjust confidence for multi-class models
        if self.model_type == 'five-class':
            # Combine confidences for merged classes
            if predicted_class in [1, 2] or predicted_class in [4, 5]:
                confidence = confidence * 0.9  # Penalty for merged classes
        
        return {
            'sentiment': predicted_sentiment,
            'confidence': confidence
        }

    def _check_domain_indicators(self, text: str, domain: str = None) -> Dict[str, Union[bool, str]]:
        """
        Check text for domain-specific sentiment indicators.
        
        Args:
            text: The text to analyze
            domain: The domain to check indicators against (e.g., 'technology', 'software')
        
        Returns:
            Dict containing domain sentiment analysis results
        """
        if not domain or domain not in self.domain_indicators:
            return {
                'has_indicators': False,
                'sentiment': None,
                'domain': domain
            }
        
        text_lower = text.lower()
        domain_indicators = self.domain_indicators[domain]
        
        # Check for positive indicators
        positive_matches = sum(1 for indicator in domain_indicators['positive'] 
                             if indicator in text_lower)
        
        # Check for negative indicators
        negative_matches = sum(1 for indicator in domain_indicators['negative'] 
                             if indicator in text_lower)
        
        # Check for neutral markers
        neutral_matches = sum(1 for marker in domain_indicators['neutral_markers'] 
                            if marker in text_lower)
        
        # Determine domain sentiment
        if neutral_matches > 0 and (positive_matches + negative_matches) <= neutral_matches:
            domain_sentiment = 'neutral'
        elif positive_matches > negative_matches:
            domain_sentiment = 'positive'
        elif negative_matches > positive_matches:
            domain_sentiment = 'negative'
        else:
            domain_sentiment = None
        
        return {
            'has_indicators': bool(positive_matches + negative_matches + neutral_matches),
            'sentiment': domain_sentiment,
            'domain': domain,
            'indicator_counts': {
                'positive': positive_matches,
                'negative': negative_matches,
                'neutral': neutral_matches
            }
        }

    def _analyze_context(self, text: str) -> Dict[str, bool]:
        """
        Analyze the context of the text for sentiment modifiers.
        
        Args:
            text: The text to analyze
        
        Returns:
            Dict containing context analysis results
        """
        text_lower = text.lower()
        words = word_tokenize(text_lower)
        
        return {
            'has_contrast': any(marker in text_lower for marker in self.contrast_markers),
            'has_neutral_indicators': any(indicator in text_lower for indicator in self.neutral_indicators),
            'word_count': len(words),
            'has_multiple_sentences': len(text.split('.')) > 1
        }

    def _detect_neutral_patterns(self, text: str) -> bool:
        """
        Detect neutral sentiment patterns using regex.
        
        Args:
            text: The text to analyze
            
        Returns:
            bool: True if neutral patterns are detected
        """
        text_lower = text.lower()
        
        # Check each pattern
        for pattern in self.neutral_patterns.values():
            if re.search(pattern, text_lower):
                return True
                
        return False

    def _get_confidence_threshold(self, sentiment_type: str) -> float:
        """
        Get confidence threshold for a specific sentiment type.
        
        Args:
            sentiment_type: The type of sentiment (positive, negative, neutral)
            
        Returns:
            float: The confidence threshold
        """
        return self.confidence_thresholds.get(sentiment_type, 
                                            self.confidence_thresholds['default'])

    def _adjust_confidence_for_neutral(self, confidence: float, 
                                     context: Dict[str, bool], 
                                     text: str) -> float:
        """
        Adjust confidence score for neutral predictions based on context.
        
        Args:
            confidence: Original confidence score
            context: Context analysis results
            text: Original text
            
        Returns:
            float: Adjusted confidence score
        """
        # Reduce confidence if there are contrasting statements
        if context['has_contrast']:
            confidence *= 0.9
            
        # Adjust based on text length and complexity
        if context['has_multiple_sentences'] and context['word_count'] > 20:
            confidence *= 0.95
            
        # Stronger confidence if multiple neutral indicators are present
        neutral_count = sum(1 for indicator in self.neutral_indicators 
                          if indicator in text.lower())
        if neutral_count > 1:
            confidence = min(confidence * 1.1, 1.0)
            
        return confidence

    def validate_sentiment(self, text: str, labeled_sentiment: str, domain: str = None):
        """Enhanced sentiment validation with model-specific handling"""
        domain_sentiment = self._check_domain_indicators(text, domain)
        
        # Get context analysis
        context = self._analyze_context(text)
        
        # Get model prediction
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.model(**inputs)
        probs = outputs.logits.softmax(dim=-1)
        predicted_class = outputs.logits.argmax().item()
        confidence = probs[0][predicted_class].item()
        
        # Map model output to standardized format
        prediction = self._map_model_output(predicted_class, confidence)
        predicted_sentiment = prediction['sentiment']
        confidence = prediction['confidence']
        
        # Apply neutral detection logic
        has_neutral_indicators = any(indicator in text.lower() for indicator in self.neutral_indicators)
        has_neutral_patterns = self._detect_neutral_patterns(text)
        
        # Determine predicted sentiment with neutral consideration
        if has_neutral_indicators or has_neutral_patterns:
            predicted_sentiment = "neutral"
            confidence = self._adjust_confidence_for_neutral(confidence, context, text)
        
        # Determine if there's a mismatch using dynamic thresholds
        is_mismatch = False
        if labeled_sentiment == "neutral":
            threshold = self._get_confidence_threshold("neutral")
            is_mismatch = (confidence > threshold and 
                          not has_neutral_indicators and 
                          not has_neutral_patterns)
        else:
            threshold = self._get_confidence_threshold(predicted_sentiment)
            is_mismatch = (predicted_sentiment != labeled_sentiment and 
                          confidence >= threshold)
        
        return {
            'is_mismatch': is_mismatch,
            'predicted': predicted_sentiment,
            'confidence': confidence,
            'domain_sentiment': domain_sentiment,
            'has_neutral_indicators': has_neutral_indicators,
            'has_neutral_patterns': has_neutral_patterns,
            'model_type': self.model_type,
            'model_name': self.model_config['name']
        }

    @classmethod
    def list_available_models(cls) -> Dict[str, str]:
        """List available sentiment models with descriptions"""
        return {k: v['description'] for k, v in ModelConfig.SUPPORTED_MODELS.items()}