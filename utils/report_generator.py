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
    output_path: str,
    analysis_results: Dict,
    original_data: List[Dict],
    duplicates: List[Dict],
    sentiment_mismatches: List[Dict],
    similarity_analysis: Dict
) -> None:
    """
    Generate a comprehensive PDF report containing all analysis results.
    All visualizations are generated and embedded directly in the PDF.
    """
    with managed_temp_directory() as temp_dir:
        try:
            # Initialize visualization and statistical analyzers
            viz_gen = VisualizationGenerator(temp_dir)
            stat_analyzer = StatisticalAnalyzer()
            
            pdf = FPDF()
            pdf.add_page()
            
            # Title
            pdf.set_font("Arial", size=16, style='B')
            pdf.cell(0, 10, "Review Quality Analysis Report", ln=True, align='C')
            pdf.ln(10)
            
            # Two-column layout for metrics
            col_width = (pdf.w - 2 * PDF_MARGIN) / 2
            
            # Left column - Basic Statistics
            pdf.set_font("Arial", size=12, style='B')
            pdf.set_x(PDF_MARGIN)
            pdf.cell(col_width, 10, "Basic Statistics", ln=True)
            pdf.set_font("Arial", size=10)
            
            stats = [
                ("Total Original Reviews", len(original_data)),
                ("Reviews After Cleaning", len(original_data) - similarity_analysis['total_removed']),
                ("Semantic Duplicates", len(similarity_analysis['semantic_duplicates'])),
                ("Combined Duplicates", len(similarity_analysis['combined_duplicates'])),
                ("Structural Similarities", len(similarity_analysis['structural_similars'])),
                ("Sentiment Mismatches", len(sentiment_mismatches))
            ]
            
            for label, value in stats:
                pdf.set_x(PDF_MARGIN)
                pdf.cell(col_width, 8, f"{label}: {value}", ln=True)
            
            # Right column - Quality Metrics
            y_position = pdf.get_y()
            pdf.set_xy(PDF_MARGIN + col_width, y_position - (len(stats) * 8))
            pdf.set_font("Arial", size=12, style='B')
            pdf.cell(col_width, 10, "Quality Metrics", ln=True)
            pdf.set_font("Arial", size=10)
            
            metrics = [
                ("Avg Linguistic Quality", f"{analysis_results.get('linguistic_quality', 0.0):.2f}"),
                ("Topic Diversity", f"{analysis_results.get('topic_diversity', 0.0):.2f}"),
                ("Topic Coherence (C_v)", f"{analysis_results.get('topic_coherence_cv', 0.0):.2f}"),
                ("Topic Coherence (UMass)", f"{analysis_results.get('topic_coherence_umass', 0.0):.2f}"),
                ("Avg Sentiment Confidence", f"{analysis_results.get('sentiment_confidence', 0.0):.2f}")
            ]
            
            for label, value in metrics:
                pdf.set_x(PDF_MARGIN + col_width)
                pdf.cell(col_width, 8, f"{label}: {value}", ln=True)
            
            # Visualizations
            pdf.add_page()
            pdf.set_font("Arial", size=12, style='B')
            pdf.cell(0, 10, "Text Analysis Visualizations", ln=True)
            
            # Word Cloud
            if original_data:
                texts = [review['text'] for review in original_data]
                wordcloud_path = viz_gen.generate_wordcloud(texts)
                pdf.image(str(wordcloud_path), x=20, w=170)
            
            # N-gram Analysis
            if original_data:
                texts = [review['text'] for review in original_data]
                ngram_data = stat_analyzer.analyze_ngrams(texts)
                ngram_path = viz_gen.generate_ngram_plot(ngram_data)
                pdf.image(str(ngram_path), x=10, w=190)
            
            # Sentiment Distribution
            if 'sentiment_distribution' in analysis_results:
                sentiment_path = viz_gen.generate_sentiment_distribution(
                    analysis_results['sentiment_distribution'],
                    [len(review['text']) for review in original_data]
                )
                pdf.image(str(sentiment_path), x=20, w=170)
            
            # Similarity Analysis
            pdf.add_page()
            pdf.set_font("Arial", size=12, style='B')
            pdf.cell(0, 10, "Similarity Analysis", ln=True)
            pdf.set_font("Arial", size=10)
            
            if similarity_analysis['semantic_duplicates']:
                pdf.ln(5)
                pdf.set_font("Arial", size=11, style='B')
                pdf.cell(0, 10, "Example Semantic Duplicates:", ln=True)
                pdf.set_font("Arial", size=9)
                
                for i, pair in enumerate(similarity_analysis['semantic_duplicates'][:3], 1):
                    pdf.set_font("Arial", size=9, style='B')
                    pdf.cell(0, 8, f"Pair {i}:", ln=True)
                    pdf.set_font("Arial", size=9)
                    pdf.multi_cell(0, 6, f"Review 1: {truncate_text(pair['text1'], MAX_TEXT_LENGTH)}")
                    pdf.multi_cell(0, 6, f"Review 2: {truncate_text(pair['text2'], MAX_TEXT_LENGTH)}")
                    pdf.cell(0, 6, f"Similarity Score: {pair['similarity']:.3f}", ln=True)
                    pdf.ln(2)
            
            # Sentiment Mismatch Analysis
            pdf.add_page()
            pdf.set_font("Arial", size=12, style='B')
            pdf.cell(0, 10, "Sentiment Mismatch Analysis", ln=True)
            pdf.set_font("Arial", size=10)
            
            if sentiment_mismatches:
                for i, mismatch in enumerate(sentiment_mismatches[:5], 1):
                    pdf.ln(5)
                    pdf.set_font("Arial", size=10, style='B')
                    pdf.cell(0, 8, f"Example {i}:", ln=True)
                    pdf.set_font("Arial", size=9)
                    pdf.multi_cell(0, 6, f"Review: {truncate_text(mismatch['text'], MAX_TEXT_LENGTH)}")
                    pdf.multi_cell(0, 6, f"Original Sentiment: {mismatch['original_sentiment']}")
                    pdf.multi_cell(0, 6, f"Predicted Sentiment: {mismatch['predicted_sentiment']} (Confidence: {mismatch['confidence']:.3f})")
                    if 'analysis_factors' in mismatch and 'explanation' in mismatch['analysis_factors']:
                        pdf.multi_cell(0, 6, f"Explanation: {mismatch['analysis_factors']['explanation']}")
                    pdf.ln(2)
            
            # Save the report
            pdf.output(output_path)
            logging.info(f"Successfully generated PDF report: {output_path}")
            
        except Exception as e:
            logging.error(f"Error generating PDF: {str(e)}")
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