from fpdf import FPDF
from typing import Dict, List
import os
from config import (
    PDF_FONT_SIZE,
    PDF_TITLE_SIZE,
    PDF_MARGIN,
    MAX_TEXT_LENGTH
)
from utils.text_processing import sanitize_text

def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to specified length and add ellipsis if needed"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."

def generate_pdf_report(
    file_name: str,
    report: Dict,
    duplicates: List[Dict],
    sentiment_mismatches: List[Dict],
    similarity_pairs: List[Dict]
) -> None:
    """
    Generate a structured PDF report with analysis results.
    
    Args:
        file_name (str): Output file path for the PDF
        report (Dict): Dictionary containing analysis metrics
        duplicates (List[Dict]): List of duplicate reviews
        sentiment_mismatches (List[Dict]): List of sentiment mismatch cases
        similarity_pairs (List[Dict]): List of high-similarity review pairs
    """
    try:
        pdf = FPDF()
        pdf.set_margins(PDF_MARGIN, PDF_MARGIN)
        pdf.add_page()
        pdf.set_font("Arial", size=PDF_FONT_SIZE)
        pdf.set_title(os.path.basename(file_name))

        # Add title
        pdf.set_font("Arial", size=PDF_TITLE_SIZE, style='B')
        pdf.cell(0, 10, txt=sanitize_text("Analysis Report"), ln=True, align='C')
        pdf.ln(10)

        # Define sections for the report
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
            _add_duplicates_section(pdf, duplicates)
        
        if sentiment_mismatches:
            _add_sentiment_mismatches_section(pdf, sentiment_mismatches)
        
        if similarity_pairs:
            _add_similarity_pairs_section(pdf, similarity_pairs)
        
        if report.get('topic_analysis_details'):
            _add_topic_analysis_section(pdf, report['topic_analysis_details'])
        
        if report.get('linguistic_analysis'):
            _add_linguistic_analysis_section(pdf, report['linguistic_analysis'])

        # Save the PDF
        pdf.output(file_name)
        print(f"PDF saved to {file_name}")
        
    except Exception as e:
        print(f"Error generating PDF: {str(e)}")
        raise

def _add_duplicates_section(pdf: FPDF, duplicates: List[Dict]) -> None:
    """Add duplicates section to the report"""
    pdf.add_page()
    pdf.set_font("Arial", size=14, style='B')
    pdf.cell(0, 10, txt="Duplicate Reviews", ln=True)
    pdf.set_font("Arial", size=10)
    
    for duplicate in duplicates:
        text = truncate_text(duplicate['text'], MAX_TEXT_LENGTH)
        pdf.multi_cell(0, 10, txt=sanitize_text(f"ID: {duplicate['id']}\nText: {text}\n"))

def _add_sentiment_mismatches_section(pdf: FPDF, mismatches: List[Dict]) -> None:
    """Add sentiment mismatches section to the report"""
    pdf.add_page()
    pdf.set_font("Arial", size=14, style='B')
    pdf.cell(0, 10, txt="Sentiment Mismatches", ln=True)
    pdf.set_font("Arial", size=10)
    
    for mismatch in mismatches:
        text = truncate_text(mismatch['text'], MAX_TEXT_LENGTH)
        pdf.multi_cell(0, 10, txt=sanitize_text(
            f"ID: {mismatch['id']}\n"
            f"Text: {text}\n"
            f"Expected: {mismatch['expected']}\n"
            f"Actual: {mismatch['actual']}\n"
            f"Confidence: {mismatch['confidence']:.2f}\n"
        ))

def _add_similarity_pairs_section(pdf: FPDF, pairs: List[Dict]) -> None:
    """Add similarity pairs section to the report"""
    pdf.add_page()
    pdf.set_font("Arial", size=14, style='B')
    pdf.cell(0, 10, txt="High Similarity Pairs", ln=True)
    pdf.set_font("Arial", size=10)
    
    for pair in pairs:
        text1 = truncate_text(pair['texts'][0], MAX_TEXT_LENGTH)
        text2 = truncate_text(pair['texts'][1], MAX_TEXT_LENGTH)
        pdf.multi_cell(0, 10, txt=sanitize_text(
            f"Pair {pair['pair'][0]} - {pair['pair'][1]}\n"
            f"Similarity Score: {pair['combined_similarity']:.2f}\n"
            f"Text 1: {text1}\n"
            f"Text 2: {text2}\n"
        ))

def _add_topic_analysis_section(pdf: FPDF, topic_analysis: Dict) -> None:
    """Add topic analysis section to the report"""
    pdf.add_page()
    pdf.set_font("Arial", size=14, style='B')
    pdf.cell(0, 10, txt="Topic Analysis Details", ln=True)
    pdf.set_font("Arial", size=10)
    
    for topic_id, words in topic_analysis.get('topics', {}).items():
        pdf.multi_cell(0, 10, txt=sanitize_text(
            f"{topic_id}:\n"
            f"Top Words: {', '.join(words)}\n"
        ))

def _add_linguistic_analysis_section(pdf: FPDF, linguistic_analysis: Dict) -> None:
    """Add linguistic analysis section to the report"""
    pdf.add_page()
    pdf.set_font("Arial", size=14, style='B')
    pdf.cell(0, 10, txt="Linguistic Analysis Details", ln=True)
    pdf.set_font("Arial", size=10)
    
    for score in linguistic_analysis.get('individual_scores', []):
        pdf.multi_cell(0, 10, txt=sanitize_text(
            f"Quality Score: {score['overall_score']:.2f}\n"
            f"Coherence Score: {score['coherence_score']:.2f}\n"
            f"Sophistication Score: {score['sophistication_score']:.2f}\n"
        ))