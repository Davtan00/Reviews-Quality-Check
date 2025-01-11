from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Dict, Union, List
import logging
from configs.models import ModelConfig, DomainIndicators
from nltk import word_tokenize
import re

class SentimentValidator:
    """
    Advanced sentiment validation system with configurable models.
    
    This class uses a Hugging Face Transformers-based model to predict 
    sentiment and then refines the result by checking:
      - Domain-specific indicators (positive, negative, neutral markers)
      - Context markers and neutral patterns (e.g., contrast words, 
        balanced statements, comparisons)
      - Confidence thresholds (which differ per model type)
    """
    
    def __init__(self, model_key: str = 'distilbert-sst2'):
        """
        Initialize the sentiment validator with a specific model.
        
        Args:
            model_key: Key from ModelConfig.SUPPORTED_MODELS.
                      Defaults to 'distilbert-sst2'.
        Raises:
            ValueError: If the model_key is not found in SUPPORTED_MODELS.
        """
        if model_key not in ModelConfig.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model key. Choose from: {list(ModelConfig.SUPPORTED_MODELS.keys())}"
            )
        
        self.model_config = ModelConfig.SUPPORTED_MODELS[model_key]
        self.model_type = self.model_config['type']
        self.label_mapping = self.model_config['mapping']
        
        logging.info(f"Loading sentiment model: {self.model_config['name']}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config['name'])
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_config['name'])
        
        self.confidence_thresholds = self._get_default_thresholds()
        self.domain_indicators = DomainIndicators.INDICATORS
        
        # Common contrast markers that often indicate neutral or balanced sentiment
        self.contrast_markers = {'but', 'however', 'although', 'though', 'while', 'yet'}
        
        # Base set of neutral indicators
        self.neutral_indicators = {
            'adequate', 'adequately', 'average', 'basic', 'decent', 'fair', 'moderate',
            'normal', 'ordinary', 'reasonable', 'standard', 'typical', 'usual',
            'performs adequately', 'works fine', 'meets expectations', 'as expected',
            'suitable for', 'acceptable', 'sufficient', 'satisfactory'
        }
        
        # Additional nuanced indicators and phrases that might suggest neutrality
        self.neutral_indicators.update({
            'mixed feelings', 'balanced', 'middle ground', 'somewhat',
            'relatively', 'fairly', 'neither', 'nor', 'compared to',
            'while', 'although', 'however', 'on one hand', 'on the other hand',
            'pros and cons', 'trade-off', 'trade off', 'compromise',
            'limited compared', 'basic but', 'simple but', 'decent but',
            'good enough', 'not great but', 'not bad but'
        })
        
        # Regex-based patterns that typically reflect neutral/balanced language
        self.neutral_patterns = {
            'performance_comparison': r'(?i)(compared|relative|versus|vs).*(?:newer|other|previous|similar)',
            'balanced_opinion': r'(?i)(while|although|however).*but',
            'moderate_intensity': r'(?i)(somewhat|fairly|relatively|quite|rather)\s\w+',
            'explicit_neutral': r'(?i)(neutral|mixed|balanced|middle ground|average|moderate)',
            'pros_cons': r'(?i)(pros.*cons|advantages.*disadvantages|benefits.*drawbacks)',
        }

    def _get_default_thresholds(self) -> Dict[str, float]:
        """
        Get default confidence thresholds based on the model type.
        
        Returns:
            A dictionary mapping sentiment type ('neutral', 'positive', 'negative')
            or 'default' to a float threshold.
        """
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
        """
        Map model output to a standardized sentiment label and adjust confidence if needed.
        
        Args:
            predicted_class: The class index predicted by the model.
            confidence: The probability or confidence value for that class.
        
        Returns:
            A dict containing 'sentiment' (str) and 'confidence' (float).
        """
        predicted_sentiment = self.label_mapping.get(predicted_class, 'neutral')
        
        # Example: In a five-class model, some classes might merge, so we penalize confidence slightly.
        if self.model_type == 'five-class':
            if predicted_class in [1, 2] or predicted_class in [4, 5]:
                confidence *= 0.9  # Penalty for merged classes

        return {
            'sentiment': predicted_sentiment,
            'confidence': confidence
        }

    def _check_domain_indicators(self, text: str, domain: str = None) -> Dict[str, Union[bool, str]]:
        """
        Check text for domain-specific sentiment indicators.
        
        Args:
            text: The text to analyze.
            domain: The domain to check indicators against (e.g., 'technology', 'software').
        
        Returns:
            A dictionary containing:
              - 'has_indicators': bool, whether domain indicators were found
              - 'sentiment': str or None, the domain-driven sentiment
              - 'domain': str, the domain used
              - 'indicator_counts': dict with positive, negative, and neutral counts
        """
        if not domain or domain not in self.domain_indicators:
            return {
                'has_indicators': False,
                'sentiment': None,
                'domain': domain
            }
        
        text_lower = text.lower()
        domain_indicators = self.domain_indicators[domain]
        
        positive_matches = sum(1 for indicator in domain_indicators['positive'] 
                               if indicator in text_lower)
        negative_matches = sum(1 for indicator in domain_indicators['negative'] 
                               if indicator in text_lower)
        neutral_matches = sum(1 for marker in domain_indicators['neutral_markers'] 
                              if marker in text_lower)
        
        # Basic logic: if neutral markers outnumber positive+negative, treat it as neutral
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
        Analyze the context of the text for sentiment modifiers (contrast markers, 
        neutral indicators, sentence counts, etc.).
        
        Args:
            text: The text to analyze.
        
        Returns:
            A dictionary indicating presence of contrast, neutral indicators, 
            word count, and whether multiple sentences exist.
        """
        text_lower = text.lower()
        words = word_tokenize(text_lower)
        
        return {
            'has_contrast': any(marker in text_lower for marker in self.contrast_markers),
            'has_neutral_indicators': any(indicator in text_lower for indicator in self.neutral_indicators),
            'word_count': len(words),
            'has_multiple_sentences': (len(text.split('.')) > 1)
        }

    def _detect_neutral_patterns(self, text: str) -> bool:
        """
        Detect neutral sentiment patterns using pre-defined regex expressions.
        
        Args:
            text: The text to analyze.
        
        Returns:
            True if any neutral pattern regex matches, False otherwise.
        """
        text_lower = text.lower()
        
        for pattern in self.neutral_patterns.values():
            if re.search(pattern, text_lower):
                return True
        return False

    def _check_neutral_indicators(self, text: str) -> Dict[str, bool]:
        """
        Check for various indicators of neutral sentiment in the text.
        
        Args:
            text: The text to analyze.
        
        Returns:
            A dictionary of booleans summarizing the presence of certain patterns 
            or indicators (contrast markers, moderate words, etc.), plus a 
            'strong_neutral_indication' key if multiple signals are found.
        """
        text_lower = text.lower()
        words = set(word_tokenize(text_lower))
        
        indicators = {
            'has_contrast_markers': bool(words.intersection(self.contrast_markers)),
            'has_neutral_words': bool(words.intersection(self.neutral_indicators)),
            'has_balanced_opinion': any(
                re.search(pattern, text_lower) for pattern in [
                    r'(?i)(while|although|however).*but',
                    r'(?i)on(\s+the)?\s+one\s+hand.*on(\s+the)?\s+other(\s+hand)?',
                    r'(?i)pros.*cons',
                    r'(?i)advantages.*disadvantages'
                ]
            ),
            'has_moderate_intensity': any(
                re.search(pattern, text_lower) for pattern in [
                    r'(?i)(somewhat|fairly|relatively|quite|rather)\s\w+',
                    r'(?i)not\s+(too|very|that|particularly|especially)\s+\w+',
                    r'(?i)(good|bad)\s+enough'
                ]
            ),
            'has_comparison': bool(
                re.search(r'(?i)(compared|relative|versus|vs|than).*(?:other|previous|similar|different)',
                          text_lower)
            )
        }
        
        # Check each of the known neutral patterns
        for pattern_name, pattern in self.neutral_patterns.items():
            indicators[f'matches_{pattern_name}'] = bool(re.search(pattern, text_lower))
        
        # If multiple signals are True, we consider it strong evidence of neutral
        neutral_signals = sum(1 for v in indicators.values() if v)
        indicators['strong_neutral_indication'] = (neutral_signals >= 2)
        
        return indicators

    def _get_confidence_threshold(self, sentiment_type: str) -> float:
        """
        Get confidence threshold for a specific sentiment type.
        
        Args:
            sentiment_type: The type of sentiment ('positive', 'negative', 'neutral').
        
        Returns:
            The configured confidence threshold (float).
        """
        return self.confidence_thresholds.get(sentiment_type, self.confidence_thresholds['default'])

    def _adjust_confidence_for_neutral(self, confidence: float, 
                                       context: Dict[str, bool], 
                                       text: str) -> float:
        """
        Adjust the confidence score when the predicted sentiment is neutral, 
        factoring in context about contrast markers, text length, etc.
        
        Args:
            confidence: Original confidence score.
            context: Context analysis results.
            text: Original text (for additional checks if needed).
        
        Returns:
            The adjusted confidence score (float).
        """
        if context['has_contrast']:
            confidence *= 0.9
        
        # If the text is long and has multiple sentences, we slightly lower confidence
        if context['has_multiple_sentences'] and context['word_count'] > 20:
            confidence *= 0.95
        
        # If there are multiple neutral indicators present, we slightly increase 
        # neutral confidence, capped at 1.0
        neutral_count = sum(
            1 for indicator in self.neutral_indicators 
            if indicator in text.lower()
        )
        if neutral_count > 1:
            confidence = min(confidence * 1.1, 1.0)
        
        return confidence

    def validate_sentiment(self, text: str, labeled_sentiment: str, domain: str = None) -> Dict[str, Union[str, bool, float, Dict]]:
        """
        Perform sentiment validation on a single piece of text against a labeled sentiment.
        
        Steps:
         1. Check domain-based indicators (if domain is provided).
         2. Analyze context (contrast words, multiple sentences, etc.).
         3. Run the model to predict sentiment and confidence.
         4. Check for neutral indicators and patterns; possibly override sentiment to 'neutral'.
         5. Determine if there's a mismatch using dynamic thresholds.
        
        Args:
            text: The review text to analyze.
            labeled_sentiment: The sentiment label assigned externally (e.g., 'positive').
            domain: Optional domain for specialized checks (e.g., 'software').
        
        Returns:
            A dictionary with:
              - 'is_mismatch': bool
              - 'predicted': str
              - 'confidence': float
              - 'domain_sentiment': dict (details about domain-based indicators)
              - 'has_neutral_indicators': bool
              - 'has_neutral_patterns': bool
              - 'model_type': str
              - 'model_name': str
        """
        logging.debug(f"Starting sentiment validation for text: {text[:100]}...")
        
        # 1. Check domain indicators
        logging.debug("Checking domain indicators...")
        domain_sentiment = self._check_domain_indicators(text, domain)
        if domain_sentiment['has_indicators']:
            logging.debug(f"Found domain indicators: {domain_sentiment['indicator_counts']}")
        
        # 2. Analyze context
        logging.debug("Analyzing context...")
        context = self._analyze_context(text)
        if context['has_contrast']:
            logging.debug("Found contrasting statements in text")
        
        # 3. Run the model prediction
        logging.debug("Running model prediction...")
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.model(**inputs)
        probs = outputs.logits.softmax(dim=-1)
        predicted_class = outputs.logits.argmax().item()
        confidence = probs[0][predicted_class].item()
        
        # Map output to standardized format
        prediction = self._map_model_output(predicted_class, confidence)
        predicted_sentiment = prediction['sentiment']
        confidence = prediction['confidence']
        logging.debug(f"Initial model prediction: {predicted_sentiment} (confidence: {confidence:.3f})")
        
        # 4. Check for neutral indicators/patterns and possibly override sentiment
        logging.debug("Checking for neutral indicators...")
        has_neutral_indicators = any(indicator in text.lower() for indicator in self.neutral_indicators)
        has_neutral_patterns = self._detect_neutral_patterns(text)
        
        if has_neutral_indicators:
            logging.debug("Found neutral indicators in text")
        if has_neutral_patterns:
            logging.debug("Found neutral patterns in text")
        
        # If either are present, we override to neutral and adjust confidence
        if has_neutral_indicators or has_neutral_patterns:
            predicted_sentiment = "neutral"
            confidence = self._adjust_confidence_for_neutral(confidence, context, text)
            logging.debug(f"Overriding sentiment to neutral (confidence: {confidence:.3f})")
        
        # 5. Determine mismatch based on dynamic thresholds
        is_mismatch = False
        if labeled_sentiment == "neutral":
            # If labeled is neutral but model strongly suggests otherwise
            threshold = self._get_confidence_threshold("neutral")
            is_mismatch = (
                confidence > threshold 
                and not has_neutral_indicators
                and not has_neutral_patterns
            )
        else:
            threshold = self._get_confidence_threshold(predicted_sentiment)
            is_mismatch = (
                predicted_sentiment != labeled_sentiment 
                and confidence >= threshold
            )
        
        if is_mismatch:
            logging.info(
                f"Found sentiment mismatch - Labeled: {labeled_sentiment}, "
                f"Predicted: {predicted_sentiment} (confidence: {confidence:.3f})"
            )
        
        result = {
            'is_mismatch': is_mismatch,
            'predicted': predicted_sentiment,
            'confidence': confidence,
            'domain_sentiment': domain_sentiment,
            'has_neutral_indicators': has_neutral_indicators,
            'has_neutral_patterns': has_neutral_patterns,
            'model_type': self.model_type,
            'model_name': self.model_config['name']
        }
        
        logging.debug("Sentiment validation completed")
        return result

    def validate_sentiments_batch(self, reviews: List[Dict], domain_override: str = None) -> List[Dict]:
        """
        Validate sentiments for a batch of reviews and return those that mismatch.
        
        Args:
            reviews: List of review dictionaries with 'text' and 'sentiment' keys.
            domain_override: Optional domain to use for all reviews.
        
        Returns:
            A list of mismatched reviews, each item containing:
              - 'text'
              - 'original_sentiment'
              - 'predicted_sentiment'
              - 'confidence'
              - 'domain_indicators'
              - 'neutral_indicators'
        """
        mismatches = []
        total = len(reviews)
        batch_size = 32  # Process in smaller batches to show more frequent progress
        
        logging.info(f"Validating sentiments for {total} reviews...")
        
        try:
            from tqdm import tqdm
            with tqdm(total=total, desc="Validating sentiments") as pbar:
                for i in range(0, total, batch_size):
                    batch = reviews[i:i + batch_size]
                    
                    for review in batch:
                        try:
                            text = review['text']
                            original_sentiment = review['sentiment']
                            
                            # Run model prediction in the same way as validate_sentiment does
                            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                            outputs = self.model(**inputs)
                            predicted_class = outputs.logits.argmax().item()
                            confidence_val = outputs.logits.softmax(dim=1).max().item()
                            
                            # Map to standardized sentiment
                            prediction = self._map_model_output(predicted_class, confidence_val)
                            predicted_sentiment = prediction['sentiment']
                            confidence_val = prediction['confidence']
                            
                            # Check domain-specific indicators
                            domain_check = self._check_domain_indicators(text, domain_override)
                            
                            # Check for neutral indicators
                            neutral_check = self._check_neutral_indicators(text)
                            
                            # If strong neutral signals exist but model is not neutral, lower confidence
                            if neutral_check['strong_neutral_indication'] and predicted_sentiment != 'neutral':
                                confidence_val *= 0.8
                            
                            # If confidence is high enough and sentiments don't match, record a mismatch
                            threshold = self.confidence_thresholds.get(predicted_sentiment, 
                                                                       self.confidence_thresholds['default'])
                            if confidence_val > threshold and predicted_sentiment != original_sentiment:
                                mismatches.append({
                                    'text': text,
                                    'original_sentiment': original_sentiment,
                                    'predicted_sentiment': predicted_sentiment,
                                    'confidence': confidence_val,
                                    'domain_indicators': domain_check,
                                    'neutral_indicators': neutral_check
                                })
                                
                        except Exception as e:
                            logging.warning(f"Error processing review: {str(e)}")
                            continue
                        
                        # Update progress bar
                        pbar.update(1)
                        
        except ImportError:
            # Fallback if tqdm is not available
            for i, review in enumerate(reviews):
                if (i + 1) % 100 == 0:
                    logging.info(f"Processed {i + 1}/{total} reviews")
                    
                try:
                    text = review['text']
                    original_sentiment = review['sentiment']
                    
                    # Run model prediction
                    inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                    outputs = self.model(**inputs)
                    predicted_class = outputs.logits.argmax().item()
                    confidence_val = outputs.logits.softmax(dim=1).max().item()
                    
                    # Map to standardized sentiment
                    prediction = self._map_model_output(predicted_class, confidence_val)
                    predicted_sentiment = prediction['sentiment']
                    confidence_val = prediction['confidence']
                    
                    # Check domain-specific indicators
                    domain_check = self._check_domain_indicators(text, domain_override)
                    
                    # Check for neutral indicators
                    neutral_check = self._check_neutral_indicators(text)
                    
                    if neutral_check['strong_neutral_indication'] and predicted_sentiment != 'neutral':
                        confidence_val *= 0.8
                    
                    # If confidence is high enough and sentiments don't match
                    threshold = self.confidence_thresholds.get(predicted_sentiment, 
                                                               self.confidence_thresholds['default'])
                    if confidence_val > threshold and predicted_sentiment != original_sentiment:
                        mismatches.append({
                            'text': text,
                            'original_sentiment': original_sentiment,
                            'predicted_sentiment': predicted_sentiment,
                            'confidence': confidence_val,
                            'domain_indicators': domain_check,
                            'neutral_indicators': neutral_check
                        })
                        
                except Exception as e:
                    logging.warning(f"Error processing review: {str(e)}")
                    continue
        
        logging.info(f"Found {len(mismatches)} sentiment mismatches")
        return mismatches

    @classmethod
    def list_available_models(cls) -> Dict[str, str]:
        """
        List available sentiment models with descriptions.
        
        Returns:
            A dictionary where each key is a model key and each value is 
            a short descriptive string for that model.
        """
        return {
            key: val['description'] 
            for key, val in ModelConfig.SUPPORTED_MODELS.items()
        }
