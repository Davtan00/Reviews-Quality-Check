from fpdf import FPDF
from typing import Dict, List
import os
from config import PDF_FONT_SIZE, PDF_TITLE_SIZE, PDF_MARGIN, MAX_TEXT_LENGTH
from utils.text_processing import sanitize_text, clean_text, truncate_text


def generate_pdf_report(
    file_name: str,
    report: Dict,
    duplicates: List[Dict],
    sentiment_mismatches: List[Dict],
    similarity_pairs: List[Dict]
) -> None:
    """Generate a structured PDF report with analysis results."""
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
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
            ("Total Reviews", report['total_reviews']),
            ("Duplicates Found", report['duplicates_found']),
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
            ("Average Linguistic Quality", f"{report['average_linguistic_quality']:.2f}"),
            ("Topic Diversity", f"{report['topic_diversity']:.2f}"),
            ("Topic Coherence", f"{report['dominant_topic_coherence']:.2f}"),
            ("Average Sentiment Confidence", f"{report['sentiment_confidence']:.2f}")
        ]
        
        for label, value in metrics:
            pdf.cell(0, 8, f"{label}: {value}", ln=True)
        pdf.ln(5)

        # Text Diversity section
        pdf.set_font("Arial", size=14, style='B')
        pdf.cell(0, 10, "Text Diversity", ln=True)
        pdf.set_font("Arial", size=10)
        
        diversity = [
            ("Unigram Diversity", f"{report['unigram_diversity']:.2f}"),
            ("Bigram Diversity", f"{report['bigram_diversity']:.2f}"),
            ("Trigram Diversity", f"{report['trigram_diversity']:.2f}")
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
                pdf.multi_cell(0, 8, f"Expected: {clean_expected}")
                pdf.multi_cell(0, 8, f"Actual: {clean_actual}")
                pdf.multi_cell(0, 8, f"Confidence: {mismatch['confidence']:.2f}")
                pdf.multi_cell(0, 8, "_" * 80)
                pdf.ln(5)

        pdf.output(file_name)
        print(f"PDF report generated successfully: {file_name}")
        
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
        f"Expected: {mismatch['expected']}\n"
        f"Actual: {mismatch['actual']}\n"
        f"Confidence: {mismatch['confidence']:.2f}\n"
        f"{'_' * 80}\n\n"  
    )