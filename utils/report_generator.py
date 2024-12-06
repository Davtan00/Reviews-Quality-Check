from fpdf import FPDF
from typing import Dict, List
import os
from config import PDF_FONT_SIZE, PDF_TITLE_SIZE, PDF_MARGIN, MAX_TEXT_LENGTH
from utils.text_processing import sanitize_text, clean_text, truncate_text
from utils.visualization import VisualizationGenerator
from analyzers.statistics import StatisticalAnalyzer
from pathlib import Path
import tempfile
import shutil
from contextlib import contextmanager
import logging
import json

@contextmanager
def managed_temp_directory():
    """Context manager for temporary directory"""
    temp_dir = Path(tempfile.mkdtemp())
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def generate_pdf_report(
    file_name: str,          # Output PDF path
    report: Dict,            # Analysis results including metrics and statistics
    duplicates: List[Dict],  # List of identified duplicate reviews
    sentiment_mismatches: List[Dict],  # Reviews with sentiment analysis discrepancies
    similarity_pairs: List[Dict]       # Highly similar review pairs
) -> None:
    """Generate a structured PDF report with analysis results and visualizations"""
    
    with managed_temp_directory() as temp_dir:
        try:
            # Initialize visualization and statistical analyzers
            viz_gen = VisualizationGenerator(temp_dir)
            stat_analyzer = StatisticalAnalyzer()
            
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=10)
            pdf.add_page()
            
            # Main title
            pdf.set_font("Arial", size=16, style='B')
            main_title = "Analysis Report"
            main_title_w = pdf.get_string_width(main_title)
            page_w = pdf.w - 2 * PDF_MARGIN
            pdf.set_x((page_w - main_title_w) / 2 + PDF_MARGIN)
            pdf.cell(main_title_w, 10, main_title, ln=True, align='C')
            
            # Subtitle (filename without extension)
            clean_filename = sanitize_text(Path(file_name).stem)
            pdf.set_font("Arial", size=10)
            pdf.ln(2)
            subtitle_w = pdf.get_string_width(clean_filename)
            pdf.set_x((page_w - subtitle_w) / 2 + PDF_MARGIN)
            pdf.cell(subtitle_w, 6, clean_filename, ln=True, align='C')
            pdf.ln(4)

            # Two-column layout for metrics
            col_width = page_w / 2
            
            # Left column - Basic Statistics
            pdf.set_font("Arial", size=10, style='B')
            pdf.set_x(PDF_MARGIN)
            pdf.cell(col_width, 6, "Basic Statistics", ln=True)
            pdf.set_font("Arial", size=9)
            
            stats = [
                ("Total Reviews", report.get('total_reviews', 0)),
                ("Duplicates Found", sum(len(group) - 1 for group in duplicates)),  
                ("Similar Pairs", len(similarity_pairs)),
                ("Sentiment Mismatches", len(sentiment_mismatches)),
                ("Reviews After Cleaning", report.get('total_reviews', 0) - 
                    (sum(len(group) - 1 for group in duplicates) + len(similarity_pairs)))
            ]
            
            for label, value in stats:
                pdf.set_x(PDF_MARGIN)
                pdf.cell(col_width, 5, f"{label}: {value}", ln=True)
            
            # Right column - Quality Metrics
            y_position = pdf.get_y()
            pdf.set_xy(PDF_MARGIN + col_width, y_position - (len(stats) * 5))
            pdf.set_font("Arial", size=10, style='B')
            pdf.cell(col_width, 6, "Quality Metrics", ln=True)
            pdf.set_font("Arial", size=9)
            
            metrics = [
                ("Avg Linguistic Quality", f"{report.get('average_linguistic_quality', 0.0):.2f}"),
                ("Topic Diversity", f"{report.get('topic_diversity', 0.0):.2f}"),
                ("Topic Coherence (C_v)", f"{report.get('topic_coherence_cv', 0.0):.2f}"),
                ("Topic Coherence (UMass)", f"{report.get('topic_coherence_umass', 0.0):.2f}"),
                ("Avg Sentiment Confidence", f"{report.get('sentiment_confidence', 0.0):.2f}")
            ]
            
            for label, value in metrics:
                pdf.set_x(PDF_MARGIN + col_width)
                pdf.cell(col_width, 5, f"{label}: {value}", ln=True)
            
            # Text Diversity section below the columns
            pdf.ln(6)
            pdf.set_x(PDF_MARGIN)
            pdf.set_font("Arial", size=10, style='B')
            pdf.cell(0, 6, "Text Diversity", ln=True)
            pdf.set_font("Arial", size=9)
            
            diversity = [
                ("Unigram Diversity", f"{report.get('unigram_diversity', 0.0):.2f}"),
                ("Bigram Diversity", f"{report.get('bigram_diversity', 0.0):.2f}"),
                ("Trigram Diversity", f"{report.get('trigram_diversity', 0.0):.2f}")
            ]
            
            for label, value in diversity:
                pdf.set_x(PDF_MARGIN)
                pdf.cell(0, 5, f"{label}: {value}", ln=True)
            
            # Visualizations section with larger images
            pdf.ln(6)
            pdf.set_font("Arial", size=10, style='B')
            pdf.cell(0, 6, "Visualizations", ln=True)
            
            # Add wordcloud if data exists
            if duplicates or sentiment_mismatches:
                # Extract texts properly based on data structure
                texts = []
                # For duplicates (list of groups), get text from each review in each group
                for group in duplicates:
                    texts.extend(item['text'] for item in group)
                # For sentiment mismatches (list of dicts), get text directly
                texts.extend(item['text'] for item in sentiment_mismatches)
                
                wordcloud_path = viz_gen.generate_wordcloud(texts)
                pdf.image(wordcloud_path, x=20, w=170)  # Increased width and adjusted x position
            
            # Add ngram plots side by side
            if duplicates:
                # Flatten duplicate groups and extract texts
                duplicate_texts = []
                for group in duplicates:
                    duplicate_texts.extend(item['text'] for item in group)
                ngram_path = viz_gen.generate_ngram_plot(
                    stat_analyzer.analyze_ngrams(duplicate_texts)
                )
                pdf.image(ngram_path, x=10, w=190)  # Increased width and adjusted x position
            
            if 'real_distribution' in report:
                kl_div_path = viz_gen.generate_kl_divergence_plot(
                    report['real_distribution'],
                    report['synthetic_distribution'],
                    report['kl_divergence']
                )
                pdf.image(kl_div_path, x=30, w=150)
            
            # Save the report
            pdf.output(file_name)
            logging.info(f"Successfully generated PDF report: {file_name}")
            
        except Exception as e:
            logging.error(f"Error generating PDF: {str(e)}")
            logging.error(f"Current state - duplicates length: {len(duplicates)}")
            raise

def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to specified length and add ellipsis if needed"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."

def format_sentiment_mismatch(mismatch):
    """Format a single sentiment mismatch entry with better spacing and structure"""
    return (
        f"ID: {mismatch['id']}\n"
        f"Text: {mismatch['text']}\n"
        f"Existing Sentiment: {mismatch['expected']}\n"
        f"Analysed Sentiment: {mismatch['actual']}\n"
        f"Confidence: {mismatch['confidence']:.2f}\n"
        f"{'_' * 80}\n\n"  
    )