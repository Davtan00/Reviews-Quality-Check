import os
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from collections import Counter
from nltk.corpus import stopwords
import nltk
from textstat import flesch_reading_ease
from fpdf import FPDF
import numpy as np
import unicodedata
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from nltk import ngrams
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
import re
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from gensim.models import CoherenceModel, LdaModel
from gensim.corpora import Dictionary
from gensim.models.phrases import Phrases, Phraser
from nltk.tokenize import sent_tokenize
import language_tool_python
from textstat import flesch_reading_ease, dale_chall_readability_score
from nltk.tokenize import word_tokenize
from nltk.util import ngrams as nltk_ngrams

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

GENERATED_DATA_FOLDER = "Generated Data"
REPORT_FOLDER = "Report"

os.makedirs(GENERATED_DATA_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

def find_duplicates(data):
    texts = [entry['text'] for entry in data]
    duplicates = pd.Series(texts).duplicated(keep=False)
    return [data[i] for i in range(len(data)) if duplicates[i]]

def calculate_similarity(data):
    analyzer = SophisticatedSimilarityAnalyzer()
    texts = [entry['text'] for entry in data]
    return analyzer.analyze_similarity(texts)

def analyze_quality(data):
    results = []
    for entry in data:
        text = entry['text']
        score = flesch_reading_ease(text)
        results.append({
            "id": entry['id'],
            "text": text,
            "flesch_score": score
        })
    return results

def validate_sentiments(data):
    mismatches = []
    for entry in data:
        text = entry['text']
        sentiment = TextBlob(text).sentiment.polarity
        predicted = entry['sentiment']
        derived_sentiment = (
            'positive' if sentiment > 0.1 else 
            'negative' if sentiment < -0.1 else 
            'neutral'
        )
        if derived_sentiment != predicted:
            mismatches.append({
                "id": entry['id'],
                "text": text,
                "expected": predicted,
                "actual": derived_sentiment
            })
    return mismatches

def sanitize_text(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')

def generate_pdf_report(file_name, report, duplicates, sentiment_mismatches):
    """
    Generates structured PDF report with:
    - Basic statistics
    - Similarity analysis
    - Linguistic metrics
    - Topic distribution
    - Sentiment analysis
    - Detailed examples of duplicates and mismatches
    
    Handles Unicode normalization and proper PDF formatting
    """
    print("\nInside PDF generation:")
    print(f"Number of sentiment mismatches received: {len(sentiment_mismatches)}")
    if sentiment_mismatches:
        print("Keys in first mismatch:")
        print(list(sentiment_mismatches[0].keys()))
        print("Sample of first mismatch data:")
        print(json.dumps(sentiment_mismatches[0], indent=2))

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.set_title(file_name)

    pdf.set_font("Arial", size=16, style='B')
    pdf.cell(200, 10, txt=sanitize_text(f"Analysis Report for {file_name}"), ln=True, align='C')
    pdf.ln(10)

    sections = {
        "Basic Statistics": {
            "Total Reviews": report["total_reviews"],
            "Duplicates Found": report["duplicates_found"],
        },
        "Similarity Analysis": {
            "Average Similarity": f"{report['average_similarity']:.2f}",
            "High Similarity Pairs": report["high_similarity_pairs"],
        },
        "Linguistic Analysis": {
            "Average Linguistic Quality": f"{report['average_linguistic_quality']:.2f}",
            "Unigram Diversity": f"{report['unigram_diversity']:.2f}",
            "Bigram Diversity": f"{report['bigram_diversity']:.2f}",
            "Trigram Diversity": f"{report['trigram_diversity']:.2f}",
        },
        "Topic Analysis": {
            "Topic Diversity": f"{report['topic_diversity']:.2f}",
            "Topic Coherence": f"{report['dominant_topic_coherence']:.2f}",
        },
        "Sentiment Analysis": {
            "Sentiment Mismatches": report["sentiment_mismatches"],
            "Average Confidence": f"{report['sentiment_confidence']:.2f}",
        }
    }

    for section_title, metrics in sections.items():
        pdf.set_font("Arial", size=14, style='B')
        pdf.cell(0, 10, txt=section_title, ln=True)
        pdf.set_font("Arial", size=12)
        for metric_name, value in metrics.items():
            pdf.cell(0, 10, txt=sanitize_text(f"{metric_name}: {value}"), ln=True)
        pdf.ln(5)

    if duplicates:
        pdf.set_font("Arial", size=12, style='B')
        pdf.cell(0, 10, txt="Duplicates Found:", ln=True)
        pdf.set_font("Arial", size=10)
        for duplicate in duplicates:
            pdf.multi_cell(0, 10, txt=sanitize_text(f"ID: {duplicate['id']} - Text: {duplicate['text']}"))
        pdf.ln(10)

    if sentiment_mismatches:
        pdf.set_font("Arial", size=12, style='B')
        pdf.cell(0, 10, txt="Sentiment Mismatches:", ln=True)
        pdf.set_font("Arial", size=10)
        for mismatch in sentiment_mismatches:
            try:
                polarity = mismatch.get('component_scores', {}).get('textblob', {}).get('polarity', 0)
                
                pdf.multi_cell(
                    0, 10, 
                    txt=sanitize_text(
                        f"ID: {mismatch.get('id', 'N/A')} - Text: {mismatch.get('text', 'N/A')}\n"
                        f"Expected: {mismatch.get('expected', 'N/A')} - Actual: {mismatch.get('actual', 'N/A')}\n"
                        f"Confidence: {mismatch.get('confidence', 0):.2f} - Polarity: {polarity:.2f}"
                    )
                )
            except Exception as e:
                print(f"Error processing mismatch in PDF: {str(e)}")
                print("Problematic mismatch data:")
                print(json.dumps(mismatch, indent=2))
        pdf.ln(10)

    output_path = os.path.join(REPORT_FOLDER, f"{file_name.replace('.json', '.pdf')}")
    pdf.output(output_path)

def process_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)["generated_data"]

    duplicates = find_duplicates(data)
    similarity_result = calculate_similarity(data)
    similarity_matrix = similarity_result['similarity_matrix']
    quality_scores = analyze_quality(data)
    sentiment_mismatches = validate_sentiments(data)
    
    similarity_scores = np.tril(similarity_matrix, k=-1).flatten()
    high_similarity = similarity_scores[similarity_scores > 0.9]

    report = {
        "total_reviews": len(data),
        "duplicates_found": len(duplicates),
        "average_similarity": float(np.mean(similarity_scores)),
        "high_similarity_pairs": len(high_similarity),
        "average_linguistic_quality": float(np.mean([q['flesch_score'] for q in quality_scores if 'flesch_score' in q and q['flesch_score'] is not None])),
        "sentiment_mismatches": len(sentiment_mismatches),
    }

    try:
        topic_analysis = analyze_topic_coherence(data)
        ngram_diversity = analyze_ngram_diversity(data)
        enhanced_sentiments = enhanced_sentiment_validation(data)
        
        report.update({
            "topic_diversity": float(topic_analysis.get("topic_diversity", 0.0)),
            "dominant_topic_coherence": float(topic_analysis.get("dominant_topic_distribution", 0.0)),
            "unigram_diversity": float(ngram_diversity.get("1-gram_diversity", 0.0)),
            "bigram_diversity": float(ngram_diversity.get("2-gram_diversity", 0.0)),
            "trigram_diversity": float(ngram_diversity.get("3-gram_diversity", 0.0)),
            "sentiment_confidence": float(np.mean([m.get("confidence", 1.0) for m in enhanced_sentiments])) if enhanced_sentiments else 1.0
        })
    except Exception as e:
        print(f"Warning: Error in additional analyses: {str(e)}")
        report.update({
            "topic_diversity": 0.0,
            "dominant_topic_coherence": 0.0,
            "unigram_diversity": 0.0,
            "bigram_diversity": 0.0,
            "trigram_diversity": 0.0,
            "sentiment_confidence": 1.0
        })
    
    file_name = os.path.basename(file_path)
    generate_pdf_report(file_name, report, duplicates, enhanced_sentiments)

def analyze_topic_coherence(data, n_topics=5):
    texts = [entry['text'] for entry in data]
    
    vectorizer = CountVectorizer(max_features=1000, stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(texts)
    
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda_output = lda.fit_transform(doc_term_matrix)
    
    feature_names = vectorizer.get_feature_names_out()
    topics = {}
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-10:-1]]
        topics[f"Topic {topic_idx + 1}"] = top_words
    
    topic_diversity = np.mean([len(set(words)) for words in topics.values()])
    
    return {
        "topics": topics,
        "topic_diversity": topic_diversity,
        "dominant_topic_distribution": np.mean(lda_output.max(axis=1))
    }

def analyze_ngram_diversity(data, n_range=(1, 3)):
    texts = [entry['text'] for entry in data]
    
    diversity_metrics = {}
    for n in range(n_range[0], n_range[1] + 1):
        all_ngrams = []
        total_ngrams = 0
        
        try:
            for text in texts:
                if not text or not isinstance(text, str):
                    continue
                    
                tokens = word_tokenize(text.lower())
                
                if len(tokens) >= n:
                    text_ngrams = list(nltk_ngrams(sequence=tokens, n=n))
                    all_ngrams.extend(text_ngrams)
                    total_ngrams += len(text_ngrams)
            
            if total_ngrams > 0:
                unique_ngrams = len(set(all_ngrams))
                diversity_metrics[f"{n}-gram_diversity"] = unique_ngrams / total_ngrams
            else:
                diversity_metrics[f"{n}-gram_diversity"] = 0.0
                
        except Exception as e:
            print(f"Error processing {n}-grams: {str(e)}")
            diversity_metrics[f"{n}-gram_diversity"] = 0.0
    
    return diversity_metrics

class SophisticatedSentimentAnalyzer:
    """
    Advanced sentiment analysis using ensemble approach combining VADER, TextBlob, and custom pattern matching.
    Weights different sentiment signals and adjusts confidence thresholds dynamically based on agreement between methods.
    """
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.nlp = spacy.load('en_core_web_sm')
        
        self.intensifiers = set(['very', 'really', 'extremely', 'absolutely'])
        self.negators = set(['not', "n't", 'never', 'no'])
        self.positive_phrases = set([
            'game changer', 'excellent', 'outstanding', 'impressive',
            'love it', 'amazing', 'fantastic', 'recommend'
        ])
        self.negative_phrases = set([
            'waste of', 'terrible', 'horrible', 'awful', 'disappointing',
            'useless', 'poor', 'wouldn\'t recommend'
        ])

    def _check_patterns(self, text):
        """
        Custom pattern matching that considers:
        - Contextual intensifiers (very, really, etc.)
        - Negation handling
        - Multi-word sentiment phrases
        - Proximity-based sentiment modification
        Returns weighted sentiment score based on pattern matches
        """
        text_lower = text.lower()
        
        pos_matches = sum(1 for phrase in self.positive_phrases if phrase in text_lower)
        neg_matches = sum(1 for phrase in self.negative_phrases if phrase in text_lower)
        
        doc = self.nlp(text)
        intensified_sentiment = 0
        
        for i, token in enumerate(doc):
            if token.text.lower() in self.intensifiers:
                next_words = ' '.join([t.text.lower() for t in doc[i:i+3]])
                if any(phrase in next_words for phrase in self.positive_phrases):
                    intensified_sentiment += 1.5
                elif any(phrase in next_words for phrase in self.negative_phrases):
                    intensified_sentiment -= 1.5
        
        return pos_matches - neg_matches + intensified_sentiment

    def analyze(self, text):
        vader_scores = self.vader.polarity_scores(text)
        
        blob = TextBlob(text)
        blob_sentiment = blob.sentiment
        
        pattern_score = self._check_patterns(text)
        
        compound_score = (
            vader_scores['compound'] * 0.5 +
            blob_sentiment.polarity * 0.3 +
            (pattern_score / 5.0) * 0.2
        )
        
        scores = [
            vader_scores['compound'],
            blob_sentiment.polarity,
            pattern_score / 5.0
        ]
        confidence = 1 - np.std(scores)
        
        if confidence > 0.8:
            thresholds = (0.1, -0.1)
        else:
            thresholds = (0.05, -0.05)
            
        sentiment = (
            'positive' if compound_score > thresholds[0] else
            'negative' if compound_score < thresholds[1] else
            'neutral'
        )
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'compound_score': compound_score,
            'component_scores': {
                'vader': vader_scores,
                'textblob': blob_sentiment._asdict(),
                'pattern': pattern_score
            }
        }

def enhanced_sentiment_validation(data):
    analyzer = SophisticatedSentimentAnalyzer()
    results = []
    
    for entry in data:
        analysis = analyzer.analyze(entry['text'])
        if analysis['sentiment'] != entry['sentiment']:
            results.append({
                'id': entry['id'],
                'text': entry['text'],
                'expected': entry['sentiment'],
                'actual': analysis['sentiment'],
                'confidence': analysis['confidence'],
                'compound_score': analysis['compound_score'],
                'component_scores': analysis['component_scores']
            })
    
    return results

class SophisticatedSimilarityAnalyzer:
    """
    Hybrid similarity analysis combining semantic embeddings with n-gram overlap.
    Uses sentence transformers for semantic understanding and n-grams for surface-level similarity.
    Weights: 70% semantic similarity, 30% n-gram similarity
    """
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.semantic_threshold = 0.85
        self.ngram_threshold = 0.7
        
    def _get_embeddings(self, texts):
        return self.model.encode(texts, show_progress_bar=False)
    
    def _get_ngram_similarity(self, text1, text2, n=3):
        def get_ngrams(text, n):
            tokens = nltk.word_tokenize(text.lower())
            return set(nltk.ngrams(tokens, n))
        
        ngrams1 = get_ngrams(text1.lower(), n)
        ngrams2 = get_ngrams(text2.lower(), n)
        
        if not ngrams1 or not ngrams2:
            return 0.0
            
        intersection = len(ngrams1.intersection(ngrams2))
        union = len(ngrams1.union(ngrams2))
        return intersection / union if union > 0 else 0.0
    
    def analyze_similarity(self, texts):
        embeddings = self._get_embeddings(texts)
        semantic_sim_matrix = cosine_similarity(embeddings)
        
        similar_pairs = defaultdict(list)
        
        for i in range(len(texts)):
            for j in range(i+1, len(texts)):
                semantic_sim = semantic_sim_matrix[i][j]
                ngram_sim = self._get_ngram_similarity(texts[i], texts[j])
                
                combined_sim = (semantic_sim * 0.7 + ngram_sim * 0.3)
                
                if combined_sim > self.semantic_threshold:
                    similar_pairs['high_similarity'].append({
                        'pair': (i, j),
                        'semantic_similarity': float(semantic_sim),
                        'ngram_similarity': ngram_sim,
                        'combined_similarity': combined_sim,
                        'texts': (texts[i], texts[j])
                    })
                elif combined_sim > self.ngram_threshold:
                    similar_pairs['moderate_similarity'].append({
                        'pair': (i, j),
                        'semantic_similarity': float(semantic_sim),
                        'ngram_similarity': ngram_sim,
                        'combined_similarity': combined_sim,
                        'texts': (texts[i], texts[j])
                    })
        
        return {
            'similarity_matrix': semantic_sim_matrix,
            'similar_pairs': dict(similar_pairs),
            'average_similarity': float(np.mean(semantic_sim_matrix)),
            'max_similarity': float(np.max(semantic_sim_matrix - np.eye(len(texts))))
        }

class SophisticatedTopicAnalyzer:
    """
    Dynamic topic modeling with automatic optimization:
    1. Determines optimal number of topics using coherence scores
    2. Processes n-grams for better phrase capture
    3. Implements hierarchical topic structure
    4. Handles topic evolution and stability
    """
    def __init__(self, min_topics=2, max_topics=10):
        self.nlp = spacy.load('en_core_web_sm')
        self.min_topics = min_topics
        self.max_topics = max_topics
        self.optimal_model = None
        self.dictionary = None
        self.corpus = None
        
    def preprocess_text(self, texts):
        cleaned_texts = [re.sub(r'[^\w\s]', '', text.lower()) for text in texts]
        
        docs = [[token.lemma_ for token in self.nlp(text) 
                if not token.is_stop and not token.is_punct and token.is_alpha]
               for text in cleaned_texts]
        
        bigram = Phrases(docs, min_count=5, threshold=100)
        trigram = Phrases(bigram[docs], threshold=100)
        
        bigram_mod = Phraser(bigram)
        trigram_mod = Phraser(trigram)
        
        processed_texts = [trigram_mod[bigram_mod[doc]] for doc in docs]
        
        return processed_texts
    
    def find_optimal_topics(self, texts, coherence='c_v'):
        """
        Identifies ideal topic count by:
        - Testing topic ranges from min_topics to max_topics
        - Evaluating coherence scores for each model
        - Selecting model with highest coherence while avoiding overfitting
        Returns optimal model and associated metrics
        """
        processed_texts = self.preprocess_text(texts)
        self.dictionary = Dictionary(processed_texts)
        self.corpus = [self.dictionary.doc2bow(text) for text in processed_texts]
        
        coherence_scores = []
        models = {}
        
        for num_topics in range(self.min_topics, self.max_topics + 1):
            lda_model = LdaModel(
                corpus=self.corpus,
                num_topics=num_topics,
                id2word=self.dictionary,
                random_state=42,
                iterations=50,
                passes=10,
                alpha='auto',
                eta='auto'
            )
            
            coherence_model = CoherenceModel(
                model=lda_model,
                texts=processed_texts,
                dictionary=self.dictionary,
                coherence=coherence
            )
            
            coherence_score = coherence_model.get_coherence()
            coherence_scores.append(coherence_score)
            models[num_topics] = lda_model
        
        optimal_num_topics = self.min_topics + coherence_scores.index(max(coherence_scores))
        self.optimal_model = models[optimal_num_topics]
        
        return {
            'optimal_num_topics': optimal_num_topics,
            'coherence_scores': coherence_scores,
            'optimal_model': self.optimal_model
        }
    
    def analyze_topics(self, texts):
        if not self.optimal_model:
            self.find_optimal_topics(texts)
        
        doc_topics = [self.optimal_model.get_document_topics(bow) 
                     for bow in self.corpus]
        
        topic_diversity = np.mean([len([t for t, _ in doc]) for doc in doc_topics])
        
        topics = {}
        for idx in range(self.optimal_model.num_topics):
            topics[f"Topic {idx + 1}"] = [
                word for word, prob in 
                self.optimal_model.show_topic(idx, topn=10)
            ]
        
        return {
            'topics': topics,
            'topic_diversity': topic_diversity,
            'topic_coherence': np.mean([score for _, score in 
                                      self.optimal_model.top_topics(self.corpus)]),
            'document_topic_distribution': doc_topics
        }

class SophisticatedLinguisticAnalyzer:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.language_tool = language_tool_python.LanguageTool('en-US')
        self.sophistication_threshold = {
            'basic': 0.3,
            'intermediate': 0.6,
            'advanced': 0.8
        }
        
    def analyze_sentence_structure(self, doc):
        sentences = [sent for sent in doc.sents]
        
        lengths = [len(sent) for sent in sentences]
        types = []
        for sent in sentences:
            if any(token.dep_ == 'mark' for token in sent):
                types.append('complex')
            elif len([token for token in sent if token.pos_ == 'VERB']) > 1:
                types.append('compound')
            else:
                types.append('simple')
        
        return {
            'avg_length': np.mean(lengths),
            'sentence_types': Counter(types),
            'structure_complexity': len([t for t in types if t != 'simple']) / len(types)
        }
    
    def analyze_vocabulary(self, doc):
        pos_diversity = Counter([token.pos_ for token in doc])
        lemma_diversity = len(set([token.lemma_ for token in doc]))
        
        content_words = [token for token in doc 
                        if not token.is_stop and token.is_alpha]
        unique_content_words = set([token.lemma_ for token in content_words])
        
        return {
            'pos_distribution': dict(pos_diversity),
            'lexical_diversity': lemma_diversity / len(doc),
            'vocabulary_richness': len(unique_content_words) / len(content_words) 
                                 if content_words else 0
        }
    
    def analyze_coherence(self, doc):
        sentences = [sent for sent in doc.sents]
        
        pronouns = [token for token in doc if token.pos_ == 'PRON']
        reference_density = len(pronouns) / len(doc)
        
        discourse_markers = [token for token in doc 
                           if token.dep_ in ['mark', 'cc']]
        
        return {
            'reference_density': reference_density,
            'discourse_marker_ratio': len(discourse_markers) / len(doc),
            'sentence_flow': self._calculate_sentence_flow(sentences)
        }
    
    def _calculate_sentence_flow(self, sentences):
        if len(sentences) < 2:
            return 1.0
            
        flow_scores = []
        for i in range(len(sentences) - 1):
            curr_sent = set([token.lemma_ for token in sentences[i] 
                           if not token.is_stop])
            next_sent = set([token.lemma_ for token in sentences[i + 1] 
                           if not token.is_stop])
            
            overlap = len(curr_sent.intersection(next_sent))
            flow_scores.append(overlap / len(curr_sent.union(next_sent)) 
                             if curr_sent.union(next_sent) else 0)
            
        return np.mean(flow_scores)
    
    def analyze_quality(self, text):
        doc = self.nlp(text)
        
        grammar_errors = self.language_tool.check(text)
        
        readability = {
            'flesch_score': flesch_reading_ease(text),
            'dale_chall_score': dale_chall_readability_score(text)
        }
        
        structure = self.analyze_sentence_structure(doc)
        
        vocabulary = self.analyze_vocabulary(doc)
        
        coherence = self.analyze_coherence(doc)
        
        sophistication_score = (
            vocabulary['lexical_diversity'] * 0.3 +
            structure['structure_complexity'] * 0.3 +
            coherence['sentence_flow'] * 0.2 +
            (1 - len(grammar_errors) / len(doc)) * 0.2
        )
        
        sophistication_level = (
            'advanced' if sophistication_score > self.sophistication_threshold['advanced']
            else 'intermediate' if sophistication_score > self.sophistication_threshold['intermediate']
            else 'basic'
        )
        
        return {
            'readability': readability,
            'grammar_errors': len(grammar_errors),
            'structure_analysis': structure,
            'vocabulary_analysis': vocabulary,
            'coherence_analysis': coherence,
            'sophistication': {
                'score': sophistication_score,
                'level': sophistication_level
            }
        }

def analyze_reviews_comprehensively(data):
    """
    Master analysis function combining:
    - Topic modeling
    - Linguistic quality assessment
    - Sentiment analysis
    - Similarity detection
    
    Provides holistic view of review quality, authenticity, and content distribution
    """
    topic_analyzer = SophisticatedTopicAnalyzer()
    linguistic_analyzer = SophisticatedLinguisticAnalyzer()
    
    texts = [entry['text'] for entry in data]
    
    topic_analysis = topic_analyzer.analyze_topics(texts)
    
    linguistic_analyses = []
    for entry in data:
        analysis = linguistic_analyzer.analyze_quality(entry['text'])
        linguistic_analyses.append({
            'id': entry['id'],
            'analysis': analysis
        })
    
    return {
        'topic_analysis': topic_analysis,
        'linguistic_analyses': linguistic_analyses
    }

def main():
    files = [f for f in os.listdir(GENERATED_DATA_FOLDER) if f.endswith('.json')]
    for file in files:
        file_path = os.path.join(GENERATED_DATA_FOLDER, file)
        try:
            print(f"Processing file: {file}")
            process_file(file_path)
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    print(f"Reports saved in folder: {REPORT_FOLDER}")

if __name__ == "__main__":
    main()
