from fpdf import FPDF
from typing import Dict, List, Any
import logging
import os
import json

from config import PDF_FONT_SIZE, PDF_TITLE_SIZE, PDF_MARGIN, MAX_TEXT_LENGTH
from utils.text_processing import sanitize_text, clean_text, truncate_text
from utils.visualization import VisualizationGenerator
from analyzers.statistics import StatisticalAnalyzer
from pathlib import Path
import tempfile
import shutil
from contextlib import contextmanager

@contextmanager
def managed_temp_directory():
    temp_dir = Path(tempfile.mkdtemp())
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def generate_pdf_report(
    output_path: str,
    analysis_results: Dict[str, Any],
    original_data: List[Dict[str, Any]],
    duplicates: List[List[Dict[str, Any]]],
    sentiment_mismatches: List[Dict[str, Any]],
    similarity_analysis: List[Dict[str, Any]]
) -> None:
    """
    Generate a comprehensive PDF report containing all analysis results.
    ### TODO: If no similarities simply remove the section, dont produce empty page.
    Args:
        output_path (str): Where to save the final PDF file.
        analysis_results (dict): A dictionary with aggregated metrics 
            (e.g., 'average_linguistic_quality', 'topic_diversity', etc.).
        original_data (list of dict): The reviews prior to deduplication.
        duplicates (list of lists): Each inner list is a group of duplicated reviews.
        sentiment_mismatches (list of dict): Detailed mismatch data.
        similarity_analysis (list of dict): Each dict has 'index1', 'index2', 'text1', 'text2', 'similarity'.
    """
    with managed_temp_directory() as temp_dir:
        try:
            # Initialize visualization
            viz_gen = VisualizationGenerator(temp_dir)
            stat_analyzer = StatisticalAnalyzer()
            
            pdf = FPDF()
            pdf.add_page()
            
            # Title
            pdf.set_font("Arial", size=16, style='B')
            pdf.cell(0, 10, "Review Quality Analysis Report", ln=True, align='C')
            pdf.ln(10)
            
            # Column widths
            col_width = (pdf.w - 2 * PDF_MARGIN) / 2
            
            # Left column: Basic Stats
            pdf.set_font("Arial", size=12, style='B')
            pdf.set_x(PDF_MARGIN)
            pdf.cell(col_width, 10, "Basic Statistics", ln=True)
            pdf.set_font("Arial", size=10)
            
            # We use 'results' fields directly:
            stats = [
                ("Total Original Reviews", len(original_data)),
                ("Duplicates Found", len(duplicates)),  # how many groups; 2 same sentences => 1 
                ("Sentiment Mismatches", len(sentiment_mismatches)),
                ("High-Similarity Pairs", len(similarity_analysis))
            ]
            for label, value in stats:
                pdf.set_x(PDF_MARGIN)
                pdf.cell(col_width, 8, f"{label}: {value}", ln=True)
            
            # Right column: Quality Metrics
            y_position = pdf.get_y()
            pdf.set_xy(PDF_MARGIN + col_width, y_position - (len(stats) * 8))
            pdf.set_font("Arial", size=12, style='B')
            pdf.cell(col_width, 10, "Quality Metrics", ln=True)
            pdf.set_font("Arial", size=10)
            
           
            # average_linguistic_quality, topic_diversity, topic_coherence_cv etc.
            metrics = [
                ("Avg Linguistic Quality", f"{analysis_results.get('average_linguistic_quality', 0.0):.2f}"),
                ("Topic Diversity", f"{analysis_results.get('topic_diversity', 0.0):.2f}"),
                ("Topic Coherence (C_v)", f"{analysis_results.get('topic_coherence_cv', 0.0):.2f}"),
                ("Topic Coherence (UMass)", f"{analysis_results.get('topic_coherence_umass', 0.0):.2f}"),
                ("Sentiment Confidence", f"{analysis_results.get('sentiment_confidence', 0.0):.2f}")
            ]
            for label, value in metrics:
                pdf.set_x(PDF_MARGIN + col_width)
                pdf.cell(col_width, 8, f"{label}: {value}", ln=True)
            
            # Visualizations (word cloud, n-grams)
            pdf.add_page()
            pdf.set_font("Arial", size=12, style='B')
            pdf.cell(0, 10, "Text Analysis Visualizations", ln=True)
            
            # Wordcloud
            if original_data:
                texts = [rev['text'] for rev in original_data if 'text' in rev]
                wc_path = viz_gen.generate_wordcloud(texts)
                pdf.image(wc_path, x=20, w=170)
            
            # N-gram analysis
            if original_data:
                texts = [rev['text'] for rev in original_data if 'text' in rev]
                ngram_data = stat_analyzer.analyze_ngrams(texts)
                ngram_path = viz_gen.generate_ngram_plot(ngram_data)
                pdf.add_page()
                pdf.set_font("Arial", size=12, style='B')
                pdf.cell(0, 10, "N-gram Analysis", ln=True)
                pdf.image(ngram_path, x=10, w=190)
            
            # Similarity Analysis (list)
            pdf.add_page()
            pdf.set_font("Arial", size=12, style='B')
            pdf.cell(0, 10, "Similarity Analysis", ln=True)
            pdf.set_font("Arial", size=10)
            
            if similarity_analysis:
                pdf.ln(5)
                pdf.set_font("Arial", size=11, style='B')
                pdf.cell(0, 8, "Example High-Similarity Pairs:", ln=True)
                pdf.set_font("Arial", size=9)
                
                # Sort descending by similarity; show top 3
                top_pairs = sorted(similarity_analysis, key=lambda x: x.get('similarity', 0), reverse=True)[:3]
                for i, pair in enumerate(top_pairs, 1):
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
                pdf.ln(3)
                pdf.set_font("Arial", size=11, style='B')
                pdf.cell(0, 8, "Examples of Mismatches:", ln=True)
                pdf.set_font("Arial", size=9)
                
                for i, mismatch in enumerate(sentiment_mismatches[:5], 1):
                    pdf.ln(2)
                    pdf.set_font("Arial", size=9, style='B')
                    pdf.cell(0, 6, f"Mismatch {i}:", ln=True)
                    pdf.set_font("Arial", size=9)
                    
                    text_val = mismatch.get("text", "")
                    expected = mismatch.get("expected", "N/A")
                    actual = mismatch.get("actual", "N/A")
                    confidence = mismatch.get("confidence", 0.0)
                    
                    pdf.multi_cell(0, 6, f"Text: {truncate_text(text_val, MAX_TEXT_LENGTH)}")
                    pdf.multi_cell(0, 6, f"Expected: {expected}")
                    pdf.multi_cell(0, 6, f"Actual: {actual} (Confidence: {confidence:.3f})")
                    pdf.ln(2)
            
            # Topic Analysis (if present)
            pdf.add_page()
            pdf.set_font("Arial", size=12, style='B')
            pdf.cell(0, 10, "Topic Analysis Details", ln=True)
            pdf.set_font("Arial", size=10)
            
            topic_analysis = analysis_results.get("topic_analysis_details", {})
            if "topics" in topic_analysis:
                for topic_id, words in topic_analysis["topics"].items():
                    pdf.multi_cell(0, 6, f"Topic {topic_id}: {', '.join(words)}")
                    pdf.ln(2)
            
            # Linguistic Analysis (if present)
            pdf.add_page()
            pdf.set_font("Arial", size=12, style='B')
            pdf.cell(0, 10, "Linguistic Analysis", ln=True)
            pdf.set_font("Arial", size=10)
            
            ling_analysis = analysis_results.get("linguistic_analysis", {})
            if "individual_scores" in ling_analysis:
                for i, score_data in enumerate(ling_analysis["individual_scores"][:5], 1):
                    overall = score_data.get('overall_score', 0.0)
                    coherence = score_data.get('coherence_score', 0.0)
                    sophistication = score_data.get('sophistication_score', 0.0)
                    pdf.multi_cell(0, 6,
                                   f"Sentence {i}: "
                                   f"Overall={overall:.2f}, "
                                   f"Coherence={coherence:.2f}, "
                                   f"Sophistication={sophistication:.2f}")
                    pdf.ln(2)
            
            # Finally, save the PDF
            pdf.output(output_path)
            logging.info(f"PDF report generated at: {output_path}")
            
        except Exception as e:
            logging.error(f"Error generating PDF: {str(e)}")
            raise

def format_sentiment_mismatch(mismatch: Dict[str, Any]) -> str:
    """
    Example helper if you wanted to format a mismatch as text.
    """
    return (
        f"ID: {mismatch.get('id')}\n"
        f"Text: {mismatch.get('text')}\n"
        f"Expected: {mismatch.get('expected')}\n"
        f"Actual: {mismatch.get('actual')}\n"
        f"Confidence: {mismatch.get('confidence', 0.0):.2f}\n"
        + ("_" * 80) + "\n\n"
    )
