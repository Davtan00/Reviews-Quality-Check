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

@contextmanager
def managed_temp_directory():
    """Context manager for temporary directory"""
    temp_dir = Path(tempfile.mkdtemp())
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def generate_pdf_report(
    file_name: str,
    report: Dict,
    duplicates: List[Dict],
    sentiment_mismatches: List[Dict],
    similarity_pairs: List[Dict]
) -> None:
    """Generate a structured PDF report with analysis results and visualizations"""
    
    with managed_temp_directory() as temp_dir:
        try:
            # Initialize visualization and statistical analyzers
            viz_gen = VisualizationGenerator(temp_dir)
            stat_analyzer = StatisticalAnalyzer()
            
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            
            # Add a page before writing any content
            pdf.add_page()
            
            # Sanitize the file name for the title
            clean_filename = sanitize_text(os.path.basename(file_name))
            title = f"Analysis Report for {clean_filename}"
            
            # Improved title formatting
            pdf.set_font("Arial", size=16, style='B')
            # Calculate title width and center it
            title_w = pdf.get_string_width(title)
            page_w = pdf.w - 2 * PDF_MARGIN
            pdf.set_x((page_w - title_w) / 2 + PDF_MARGIN)
            pdf.cell(title_w, 10, title, ln=True, align='C')
            pdf.ln(5)

            # Add domain information
            pdf.set_font("Arial", size=12, style='B')
            domain = report.get('domain', 'general')
            pdf.cell(0, 10, f"Domain: {domain.capitalize()}", ln=True)
            pdf.ln(5)

            # Basic Statistics section
            pdf.set_font("Arial", size=14, style='B')
            pdf.cell(0, 10, "Basic Statistics", ln=True)
            pdf.set_font("Arial", size=10)
            
            stats = [
                ("Total Reviews", report.get('total_reviews', 0)),
                ("Duplicates Found", report.get('duplicates_found', 0)),
                ("Sentiment Mismatches", len(sentiment_mismatches)),
                ("High Similarity Pairs", len(similarity_pairs))
            ]
            
            for label, value in stats:
                pdf.cell(0, 8, f"{label}: {value}", ln=True)
            pdf.ln(5)

            # Quality Metrics section
            pdf.set_font("Arial", size=14, style='B')
            pdf.cell(0, 10, "Quality Metrics", ln=True)
            pdf.set_font("Arial", size=10)
            
            metrics = [
                ("Average Linguistic Quality", f"{report.get('average_linguistic_quality', 0.0):.2f}"),
                ("Topic Diversity", f"{report.get('topic_diversity', 0.0):.2f}"),
                ("Topic Coherence", f"{report.get('dominant_topic_coherence', 0.0):.2f}"),
                ("Average Sentiment Confidence", f"{report.get('sentiment_confidence', 0.0):.2f}")
            ]
            
            for label, value in metrics:
                pdf.cell(0, 8, f"{label}: {value}", ln=True)
            pdf.ln(5)

            # Text Diversity section
            pdf.set_font("Arial", size=14, style='B')
            pdf.cell(0, 10, "Text Diversity", ln=True)
            pdf.set_font("Arial", size=10)
            
            diversity = [
                ("Unigram Diversity", f"{report.get('unigram_diversity', 0.0):.2f}"),
                ("Bigram Diversity", f"{report.get('bigram_diversity', 0.0):.2f}"),
                ("Trigram Diversity", f"{report.get('trigram_diversity', 0.0):.2f}")
            ]
            
            for label, value in diversity:
                pdf.cell(0, 8, f"{label}: {value}", ln=True)
            pdf.ln(10)

            # Sentiment Mismatches section with improved formatting
            if sentiment_mismatches:
                pdf.add_page()
                pdf.set_font("Arial", size=14, style='B')
                pdf.cell(0, 10, "Sentiment Mismatches", ln=True)
                pdf.set_font("Arial", size=10)
                
                for mismatch in sentiment_mismatches:
                    # Sanitize all text fields
                    clean_id = sanitize_text(str(mismatch['id']))
                    clean_text = sanitize_text(truncate_text(mismatch['text'], MAX_TEXT_LENGTH))
                    clean_expected = sanitize_text(str(mismatch['expected']))
                    clean_actual = sanitize_text(str(mismatch['actual']))
                    
                    pdf.multi_cell(0, 8, f"ID: {clean_id}")
                    pdf.multi_cell(0, 8, f"Text: {clean_text}")
                    pdf.multi_cell(0, 8, f"Existing Sentiment: {clean_expected}")
                    pdf.multi_cell(0, 8, f"Analysed Sentiment: {clean_actual}")
                    pdf.multi_cell(0, 8, f"Confidence: {mismatch['confidence']:.2f}")
                    pdf.multi_cell(0, 8, "_" * 80)
                    pdf.ln(5)
  
            
            # Add wordcloud
            wordcloud_path = viz_gen.generate_wordcloud(
                [item['text'] for item in duplicates + sentiment_mismatches]
            )
            pdf.add_page()
            pdf.image(wordcloud_path, x=30, w=150)
            
            # Add KL divergence and token overlap analysis
            if 'real_distribution' in report:
                kl_div_path = viz_gen.generate_kl_divergence_plot(
                    report['real_distribution'],
                    report['synthetic_distribution'],
                    report['kl_divergence']
                )
                pdf.add_page()
                pdf.image(kl_div_path, x=30, w=150)
            
            # Add n-gram analysis
            ngram_path = viz_gen.generate_ngram_plot(
                stat_analyzer.analyze_ngrams([item['text'] for item in duplicates])
            )
            pdf.add_page()
            pdf.image(ngram_path, x=30, w=150)
            
            # Save the report
            pdf.output(file_name)
            
        except Exception as e:
            print(f"Error generating PDF: {str(e)}")
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