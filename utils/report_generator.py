from fpdf import FPDF
from typing import Dict, List
import os
from config import PDF_FONT_SIZE, PDF_TITLE_SIZE, PDF_MARGIN, MAX_TEXT_LENGTH
from utils.text_processing import sanitize_text


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
        pdf.set_margins(PDF_MARGIN, PDF_MARGIN)
        pdf.add_page()
        pdf.set_font("Arial", size=PDF_FONT_SIZE)
        pdf.set_title(os.path.basename(file_name))

        # Title
        pdf.set_font("Arial", size=PDF_TITLE_SIZE, style='B')
        pdf.cell(200, 10, txt=sanitize_text(f"Analysis Report for {file_name}"), ln=True, align='C')
        pdf.ln(10)

        # Define sections for the report
        sections = {
            "Basic Statistics": {
                "Total Reviews": report["total_reviews"],
                "Duplicates Found": len(duplicates),
                "Sentiment Mismatches": len(sentiment_mismatches),
                "High Similarity Pairs": len(similarity_pairs)
            },
            "Quality Metrics": {
                "Average Linguistic Quality": f"{report['average_linguistic_quality']:.2f}",
                "Topic Diversity": f"{report['topic_diversity']:.2f}",
                "Topic Coherence": f"{report['dominant_topic_coherence']:.2f}",
                "Average Sentiment Confidence": f"{report['sentiment_confidence']:.2f}"
            },
            "Text Diversity": {
                "Unigram Diversity": f"{report['unigram_diversity']:.2f}",
                "Bigram Diversity": f"{report['bigram_diversity']:.2f}",
                "Trigram Diversity": f"{report['trigram_diversity']:.2f}"
            }
        }

        # Add sections to report
        for section_title, metrics in sections.items():
            pdf.set_font("Arial", size=14, style='B')
            pdf.cell(0, 10, txt=section_title, ln=True)
            pdf.set_font("Arial", size=12)
            for metric_name, value in metrics.items():
                pdf.cell(0, 10, txt=sanitize_text(f"{metric_name}: {value}"), ln=True)
            pdf.ln(5)

        # Add detailed sections if data is available
        if duplicates:
            pdf.add_page()
            pdf.set_font("Arial", size=14, style='B')
            pdf.cell(0, 10, txt="Duplicate Reviews", ln=True)
            pdf.set_font("Arial", size=10)
            for duplicate in duplicates:
                pdf.multi_cell(0, 10, txt=sanitize_text(
                    f"ID: {duplicate['id']}\n"
                    f"Text: {duplicate['text']}\n"
                ))

        if sentiment_mismatches:
            pdf.add_page()
            pdf.set_font("Arial", size=14, style='B')
            pdf.cell(0, 10, txt="Sentiment Mismatches", ln=True)
            pdf.set_font("Arial", size=10)
            for mismatch in sentiment_mismatches:
                formatted_text = format_sentiment_mismatch(mismatch)
                pdf.multi_cell(0, 10, txt=sanitize_text(formatted_text))
                pdf.ln(5)  # Add extra space between entries

        if similarity_pairs:
            pdf.add_page()
            pdf.set_font("Arial", size=14, style='B')
            pdf.cell(0, 10, txt="Similar Review Pairs", ln=True)
            pdf.set_font("Arial", size=10)
            for pair in similarity_pairs:
                pdf.multi_cell(0, 10, txt=sanitize_text(
                    f"Pair {pair['pair'][0]} - {pair['pair'][1]}\n"
                    f"Similarity: {pair['combined_similarity']:.2f}\n"
                    f"Text 1: {pair['texts'][0]}\n"
                    f"Text 2: {pair['texts'][1]}\n"
                ))

        # Save the PDF
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