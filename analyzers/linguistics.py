import spacy
from textblob import TextBlob
from textstat import flesch_reading_ease, dale_chall_readability_score
from nltk.tokenize import sent_tokenize
import numpy as np

class SophisticatedLinguisticAnalyzer:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.sophistication_threshold = {
            'basic': 0.3,
            'intermediate': 0.6,
            'advanced': 0.8
        }
    
    def analyze_sentence_structure(self, doc):
        """Analyze sentence structure complexity and variety"""
        sentences = [sent for sent in doc.sents]
        if not sentences:
            return 0.0
        
        # Analyze sentence lengths
        lengths = [len(sent) for sent in sentences]
        length_variety = np.std(lengths) / max(np.mean(lengths), 1)
        
        # Analyze syntactic complexity
        complexity_scores = []
        for sent in sentences:
            # Count depth of dependency tree
            max_depth = max(len(list(token.ancestors)) for token in sent)
            # Count number of clauses
            num_clauses = len([token for token in sent if token.dep_ in {'ccomp', 'xcomp', 'advcl'}])
            complexity_scores.append(max_depth + num_clauses)
        
        complexity_score = np.mean(complexity_scores) if complexity_scores else 0
        
        return (length_variety + min(complexity_score / 10, 1)) / 2
    
    def analyze_vocabulary(self, doc):
        """Analyze vocabulary sophistication and diversity"""
        if not doc:
            return 0.0
        
        # Count unique lemmas for content words
        content_lemmas = set(token.lemma_ for token in doc 
                           if not token.is_stop and token.pos_ in {'NOUN', 'VERB', 'ADJ', 'ADV'})
        
        # Calculate lexical density
        num_content_words = len([token for token in doc 
                               if not token.is_stop and token.pos_ in {'NOUN', 'VERB', 'ADJ', 'ADV'}])
        lexical_density = num_content_words / len(doc) if len(doc) > 0 else 0
        
        # Calculate vocabulary sophistication based on word frequency
        sophisticated_words = len([token for token in doc 
                                 if not token.is_stop and token.pos_ in {'NOUN', 'VERB', 'ADJ', 'ADV'} 
                                 and token.rank >= 10000]) # Higher rank means less frequent word
        
        sophistication_score = sophisticated_words / max(len(content_lemmas), 1)
        
        return (lexical_density + min(sophistication_score, 1)) / 2
    
    def analyze_coherence(self, doc):
        """Analyze text coherence and flow"""
        sentences = list(doc.sents)
        if len(sentences) <= 1:
            return 1.0  # Single sentence is considered coherent
        
        # Analyze referential coherence
        coherence_scores = []
        for i in range(1, len(sentences)):
            prev_sent = sentences[i-1]
            curr_sent = sentences[i]
            
            
            prev_entities = set(ent.root.lemma_ for ent in prev_sent.ents)
            curr_entities = set(ent.root.lemma_ for ent in curr_sent.ents)
            entity_overlap = len(prev_entities & curr_entities) / max(len(prev_entities | curr_entities), 1)
            
            # Check for semantic similarity
            similarity = prev_sent.similarity(curr_sent)
            
            coherence_scores.append((entity_overlap + similarity) / 2)
        
        return np.mean(coherence_scores) if coherence_scores else 0
    
    def _calculate_sentence_flow(self, sentences):
        """Calculate how well sentences flow together"""
        if len(sentences) <= 1:
            return 1.0
        
        flow_scores = []
        for i in range(1, len(sentences)):
            prev_end = sentences[i-1][-1] if sentences[i-1] else ''
            curr_start = sentences[i][0] if sentences[i] else ''
            
           
            if prev_end in {'.', '!', '?'} and curr_start.isupper():
                flow_scores.append(1.0)
            else:
                flow_scores.append(0.5)
        
        return np.mean(flow_scores)
    
    def analyze_quality(self, text):
        """Comprehensive quality analysis of text"""
        if not text or not text.strip():
            return {
                'quality_score': 0.0,
                'readability_score': 0.0,
                'structure_score': 0.0,
                'vocabulary_score': 0.0,
                'coherence_score': 0.0,
                'grammar_score': 1.0
            }
        
        
        doc = self.nlp(text)
        blob = TextBlob(text)
        
      
        structure_score = self.analyze_sentence_structure(doc)
        vocabulary_score = self.analyze_vocabulary(doc)
        coherence_score = self.analyze_coherence(doc)
        
      
        readability_score = (flesch_reading_ease(text) / 100 + 
                           dale_chall_readability_score(text) / 10) / 2
        
       
        grammar_issues = self._check_grammar_with_textblob(blob)
        grammar_score = 1.0 - (len(grammar_issues) / len(text.split()))
        grammar_score = max(0.0, min(grammar_score, 1.0))
        
      
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
        """Basic grammar checking using TextBlob"""
        issues = []
        
        for sentence in blob.sentences:
            tags = sentence.tags
            for i in range(len(tags) - 1):
                word, tag = tags[i]
                next_word, next_tag = tags[i + 1]
                
                
                if (tag == 'NN' and next_tag == 'VBP') or (tag == 'NNS' and next_tag == 'VBZ'):
                    issues.append(f"Possible subject-verb agreement error: '{word} {next_word}'")
            
           
            for i, (word, tag) in enumerate(tags):
                if tag == 'DT' and i < len(tags) - 1:
                    next_word = sentence.words[i + 1].lower()
                    if word.lower() == 'a' and next_word[0] in 'aeiou':
                        issues.append(f"Consider using 'an' instead of 'a' before '{next_word}'")
        
        return issues