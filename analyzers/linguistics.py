import spacy
from textblob import TextBlob
from textstat import flesch_reading_ease, dale_chall_readability_score
from nltk.tokenize import sent_tokenize
import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    # Try loading the larger model first
    nlp = spacy.load('en_core_web_md')  # Medium-sized model with word vectors
except OSError:
    print("Installing required model...")
    from spacy.cli import download
    download('en_core_web_md')
    nlp = spacy.load('en_core_web_md')

# Add a warning if falling back to small model
def check_model_capabilities():
    if not nlp.has_pipe('vectors'):
        logger.warning(
            "Using a model without word vectors. Consider using 'en_core_web_md' or 'en_core_web_lg' "
            "for better similarity calculations."
        )

class SophisticatedLinguisticAnalyzer:
    """
    A class for comprehensive linguistic analysis using spaCy, TextBlob, and textstat.
    
    Includes functionality to:
      - Analyze sentence structure complexity (sentence length variety and dependency depth).
      - Assess vocabulary sophistication (lexical density, infrequent words).
      - Measure coherence across sentence boundaries (entity overlap, semantic similarity).
      - Compute readability scores (Flesch, Dale-Chall).
      - Perform simple grammar checks using TextBlob.
    """
    
    def __init__(self):
        """
        Initialize the linguistic analyzer.
        
        Attempts to run spaCy on the best available hardware (GPU, MPS) via spacy.prefer_gpu().
        If no compatible hardware is found, it silently falls back to CPU.
        """
        try:
            spacy.prefer_gpu()  # Attempt to use GPU/MPS if available
        except Exception as e:
            # If spaCy can't use GPU/MPS, it'll remain on CPU
            pass
        
        # Load spaCy model
        self.nlp = spacy.load('en_core_web_sm')
        
        # Optional thresholds for sophisticated usage (not directly used, but can be expanded)
        self.sophistication_threshold = {
            'basic': 0.3,
            'intermediate': 0.6,
            'advanced': 0.8
        }
    
    def analyze_sentence_structure(self, doc) -> float:
        """
        Analyze sentence structure complexity and variety within a spaCy Doc.
        
        1. Calculates length variety as the standard deviation of sentence lengths 
           divided by the mean length (helps measure consistency of sentence size).
        2. Calculates a simple syntactic complexity score by combining:
           - The maximum depth of the dependency tree per sentence.
           - The number of ccomp, xcomp, and advcl tokens per sentence.
        
        The final sentence structure score is a weighted combination of these two aspects.
        
        Args:
            doc: A spaCy Doc object.
        
        Returns:
            A float score between 0.0 and ~1.0 indicating sentence structural complexity.
        """
        sentences = list(doc.sents)
        if not sentences:
            return 0.0
        
        # Analyze sentence lengths
        lengths = [len(sent) for sent in sentences]
        mean_length = max(np.mean(lengths), 1.0)
        length_variety = np.std(lengths) / mean_length
        
        # Analyze syntactic complexity
        complexity_scores = []
        for sent in sentences:
            # Count depth of dependency tree
            max_depth = max(len(list(token.ancestors)) for token in sent)
            # Count clauses (ccomp, xcomp, advcl)
            num_clauses = sum(1 for token in sent if token.dep_ in {'ccomp', 'xcomp', 'advcl'})
            complexity_scores.append(max_depth + num_clauses)
        
        complexity_score = np.mean(complexity_scores) if complexity_scores else 0.0
        
        # Combine them (lightly scaled by 1/10 for complexity_score)
        # and clamp complexity to 1.0 if it's high
        return (length_variety + min(complexity_score / 10.0, 1.0)) / 2.0
    
    def analyze_vocabulary(self, doc) -> float:
        """
        Analyze vocabulary sophistication and diversity.
        
        1. Lexical density = (# of content words) / (total tokens).
        2. Sophistication score based on frequency rank (token.rank).
        
        Args:
            doc: A spaCy Doc object.
        
        Returns:
            A float score combining lexical density and sophistication, ~0.0–1.0 range.
        """
        if not doc or len(doc) == 0:
            return 0.0
        
        # Identify content words
        content_pos = {'NOUN', 'VERB', 'ADJ', 'ADV'}
        content_tokens = [token for token in doc if not token.is_stop and token.pos_ in content_pos]
        
        # Count unique lemmas among content words
        content_lemmas = set(token.lemma_ for token in content_tokens)
        
        # Calculate lexical density
        lexical_density = len(content_tokens) / len(doc) if len(doc) else 0.0
        
        # Count how many content tokens are "sophisticated" (rank >= 10000 => less frequent)
        sophisticated_words = sum(1 for token in content_tokens if token.rank >= 10000)
        sophistication_score = sophisticated_words / max(len(content_lemmas), 1)
        
        # Combine them with minimal weighting (just a simple average)
        return (lexical_density + min(sophistication_score, 1.0)) / 2.0
    
    def analyze_coherence(self, doc) -> float:
        """
        Analyze text coherence by measuring entity overlap and semantic similarity 
        between consecutive sentences.
        
        Args:
            doc: A spaCy Doc object.
        
        Returns:
            A float ~0.0–1.0, where higher indicates stronger coherence.
            If the text has only 1 sentence, defaults to 1.0.
        """
        sentences = list(doc.sents)
        if len(sentences) <= 1:
            return 1.0  # A single sentence is trivially considered coherent
        
        coherence_scores = []
        for i in range(1, len(sentences)):
            prev_sent = sentences[i - 1]
            curr_sent = sentences[i]
            
            # Overlap of named entities
            prev_entities = set(ent.root.lemma_ for ent in prev_sent.ents)
            curr_entities = set(ent.root.lemma_ for ent in curr_sent.ents)
            
            # Weighted overlap ratio
            union_size = len(prev_entities | curr_entities)
            if union_size == 0:
                entity_overlap = 0.0
            else:
                entity_overlap = len(prev_entities & curr_entities) / union_size
            
            # Check for semantic similarity (spaCy vector-based)
            similarity = prev_sent.similarity(curr_sent)
            
            # Combine them
            coherence_scores.append((entity_overlap + similarity) / 2.0)
        
        return float(np.mean(coherence_scores)) if coherence_scores else 0.0
    
    def _calculate_sentence_flow(self, sentences: list) -> float:
        """
        Calculate how well sentences flow together based on punctuation and capitalization.
        This is a placeholder approach.
        
        Args:
            sentences: A list of strings, each representing a sentence.
        
        Returns:
            A float ~0.0–1.0 where higher is smoother flow.
        """
        if len(sentences) <= 1:
            return 1.0
        
        flow_scores = []
        for i in range(1, len(sentences)):
            prev_end = sentences[i - 1][-1] if sentences[i - 1] else ''
            curr_start = sentences[i][0] if sentences[i] else ''
            
            # Very naive check: if the previous sentence ends with punctuation 
            # and the current sentence starts capitalized => +1
            if prev_end in {'.', '!', '?'} and curr_start.isupper():
                flow_scores.append(1.0)
            else:
                flow_scores.append(0.5)
        
        return float(np.mean(flow_scores))
    
    def analyze_quality(self, text: str) -> dict:
        """
        Perform a comprehensive quality analysis of text, returning multiple metrics.
        
        1. If the text is empty or blank, returns default zeros.
        2. Otherwise, uses spaCy to create a Doc, then calls methods for:
           - Sentence structure
           - Vocabulary sophistication
           - Coherence
           - Readability scores (Flesch, Dale-Chall)
           - Grammar checks (via TextBlob)
        
        Args:
            text: The raw string to analyze.
        
        Returns:
            A dict containing keys:
              - 'quality_score': overall average of structure, vocab, coherence, 
                                 readability, grammar
              - 'readability_score'
              - 'structure_score'
              - 'vocabulary_score'
              - 'coherence_score'
              - 'grammar_score'
              - 'grammar_issues': list of strings describing potential grammar problems
        """
        if not text or not text.strip():
            return {
                'quality_score': 0.0,
                'readability_score': 0.0,
                'structure_score': 0.0,
                'vocabulary_score': 0.0,
                'coherence_score': 0.0,
                'grammar_score': 1.0
            }
        
        # Create spaCy Doc and TextBlob
        doc = self.nlp(text)
        blob = TextBlob(text)
        
        # Extract scores
        structure_score = self.analyze_sentence_structure(doc)
        vocabulary_score = self.analyze_vocabulary(doc)
        coherence_score = self.analyze_coherence(doc)
        
        # Combine Flesch and Dale-Chall readings into a single normalized value
        readability_score = (
            (flesch_reading_ease(text) / 100.0) 
            + (dale_chall_readability_score(text) / 10.0)
        ) / 2.0
        
        # Grammar checks
        grammar_issues = self._check_grammar_with_textblob(blob)
        # A simple approach: 1 - (# issues / total words). Clamped between 0.0 and 1.0.
        grammar_score = 1.0 - (len(grammar_issues) / len(text.split()))
        grammar_score = max(0.0, min(grammar_score, 1.0))
        
        # Average all main metrics for a final "quality_score"
        quality_score = np.mean([
            structure_score,
            vocabulary_score,
            coherence_score,
            readability_score,
            grammar_score
        ])
        
        return {
            'quality_score': quality_score,
            'readability_score': readability_score,
            'structure_score': structure_score,
            'vocabulary_score': vocabulary_score,
            'coherence_score': coherence_score,
            'grammar_score': grammar_score,
            'grammar_issues': grammar_issues
        }

    def _check_grammar_with_textblob(self, blob: TextBlob) -> list:
        """
        Basic grammar checking using TextBlob, focusing on a few patterns:
          - Subject-verb agreement for singular/plural forms
          - Indefinite article usage (a/an)
        
        Args:
            blob: A TextBlob object of the text to analyze.
        
        Returns:
            A list of strings describing potential grammar issues.
        """
        issues = []
        
        for sentence in blob.sentences:
            tags = sentence.tags
            for i in range(len(tags) - 1):
                word, tag = tags[i]
                next_word, next_tag = tags[i + 1]
                
                # Simple subject-verb agreement checks
                if (tag == 'NN' and next_tag == 'VBP') or (tag == 'NNS' and next_tag == 'VBZ'):
                    issues.append(f"Possible subject-verb agreement error: '{word} {next_word}'")
            
            # Check indefinite article usage
            for i, (word, tag) in enumerate(tags):
                if tag == 'DT' and word.lower() == 'a' and i < len(tags) - 1:
                    next_word = sentence.words[i + 1].lower()
                    if next_word and next_word[0] in 'aeiou':
                        issues.append(
                            f"Consider using 'an' instead of 'a' before '{next_word}'"
                        )
        
        return issues
