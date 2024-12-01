import unicodedata
import re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from typing import Union

def sanitize_text(text: str) -> str:
    """
    Sanitize text by removing non-ASCII characters and normalizing Unicode.
    
    Args:
        text (str): Input text to sanitize
        
    Returns:
        str: Sanitized text
    """
    # Normalize Unicode characters
    normalized = unicodedata.normalize('NFKD', text)
    # Convert to ASCII, removing non-ASCII characters
    ascii_text = normalized.encode('ascii', 'ignore').decode('ascii')
    return ascii_text

def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace and normalizing line endings.
    
    Args:
        text (str): Input text to clean
        
    Returns:
        str: Cleaned text
    """
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    return text

def truncate_text(text: str, max_length: int = 100, ellipsis: str = '...') -> str:
    """
    Truncate text to specified length while preserving word boundaries.
    
    Args:
        text (str): Input text to truncate
        max_length (int): Maximum length of output text
        ellipsis (str): String to append to truncated text
        
    Returns:
        str: Truncated text
    """
    if len(text) <= max_length:
        return text
        
    truncated = text[:max_length]
    # Find last space to preserve word boundary
    last_space = truncated.rfind(' ')
    if last_space > 0:
        truncated = truncated[:last_space]
    return truncated + ellipsis

def calculate_flesch_reading_ease(text: str) -> Union[float, int]:
    """
    Calculate the Flesch Reading Ease score for the given text.
    
    The score indicates how easy the text is to read:
    90-100: Very easy to read
    80-89: Easy to read
    70-79: Fairly easy to read
    60-69: Standard
    50-59: Fairly difficult to read
    30-49: Difficult to read
    0-29: Very difficult to read
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        float: Flesch Reading Ease score
    """
    if not text or not isinstance(text, str):
        return 0
    
    try:
        # Tokenize into sentences and words
        sentences = sent_tokenize(text.strip())
        words = word_tokenize(text.lower())
        
        if not sentences or not words:
            return 0
        
        # Count syllables
        def count_syllables(word):
            # Basic syllable counting - count vowel groups
            vowels = 'aeiouy'
            word = word.lower().strip(".:,!?")
            count = 0
            prev_char_is_vowel = False
            
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_char_is_vowel:
                    count += 1
                prev_char_is_vowel = is_vowel
                
            # Handle some common cases
            if word.endswith('e'):
                count -= 1
            if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
                count += 1
            if count == 0:
                count = 1
                
            return count
        
        total_syllables = sum(count_syllables(word) for word in words)
        
        # Calculate metrics
        words_per_sentence = len(words) / len(sentences)
        syllables_per_word = total_syllables / len(words)
        
        # Flesch Reading Ease formula
        score = 206.835 - (1.015 * words_per_sentence) - (84.6 * syllables_per_word)
        
        # Clamp score between 0 and 100
        return max(0, min(100, score))
        
    except Exception as e:
        print(f"Error calculating Flesch score: {str(e)}")
        return 0