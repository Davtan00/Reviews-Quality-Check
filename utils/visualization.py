import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from matplotlib_venn import venn2
import numpy as np
from typing import Dict, List, Any
import os
from pathlib import Path

class VisualizationGenerator:
    def __init__(self, temp_dir: Path):
        self.temp_dir = temp_dir
        # Set matplotlib to use Agg backend for better memory management
        plt.switch_backend('Agg')
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        plt.close('all')  # Ensure all figures are closed
        
    def generate_sentiment_distribution(self, sentiment_data: Dict[str, int], lengths: List[int]) -> str:
        """Generates side-by-side plots for sentiment distribution and review length analysis"""
        output_path = self.temp_dir / "sentiment_distribution.png"
        
        try:
            # Clear any existing figures
            plt.close('all')
            
            fig, ax = plt.subplots(1, 2, figsize=(14, 6))
            
            # Color-coded sentiment bars for intuitive interpretation
            sentiments = list(sentiment_data.keys())
            counts = list(sentiment_data.values())
            ax[0].bar(sentiments, counts, color=['#6cba6b', '#f16a6a', '#d1d1d1'])
            ax[0].set_title('Sentiment Distribution')
            ax[0].set_xlabel('Sentiment')
            ax[0].set_ylabel('Frequency')
            
            # KDE plot with memory optimization
            hist_data = np.array(lengths)
            ax[1].hist(hist_data, bins='auto', density=True, alpha=0.7, color='#f79c42')
            
            if len(hist_data) > 1000:
                # Use subset for KDE to reduce memory usage
                sample_size = min(1000, len(hist_data))
                sample_data = np.random.choice(hist_data, size=sample_size, replace=False)
                sns.kdeplot(data=sample_data, ax=ax[1], color='#f79c42', label='Density')
            else:
                sns.kdeplot(data=hist_data, ax=ax[1], color='#f79c42', label='Density')
            
            ax[1].set_title('Review Length Distribution')
            ax[1].set_xlabel('Review Length')
            ax[1].set_ylabel('Density')
            ax[1].legend()

            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            return str(output_path)
            
        finally:
            plt.close('all')
    
    def generate_wordcloud(self, texts: List[str]) -> str:
        """Generate wordcloud visualization with memory optimization"""
        output_path = self.temp_dir / "wordcloud.png"
        
        try:
            plt.close('all')  # Clear any existing figures
            
            # Process text in chunks if too large
            max_text_size = 100000  # 100KB of text at a time
            processed_text = ""
            
            for text in texts:
                if len(processed_text) + len(text) > max_text_size:
                    break
                processed_text += " " + text
            
            wordcloud = WordCloud(
                width=800, 
                height=400,
                background_color='white',
                max_words=200,  # Limit number of words
                prefer_horizontal=0.7
            ).generate(processed_text)
            
            plt.figure(figsize=(10, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            return str(output_path)
            
        finally:
            plt.close('all')
    
    def generate_kl_divergence_plot(self, real_dist: Dict[str, float], 
                                  synthetic_dist: Dict[str, float],
                                  kl_div: float) -> str:
        """Visualizes distribution differences between real and synthetic data"""
        output_path = self.temp_dir / "kl_divergence.png"
        
        try:
            # Clear any existing figures
            plt.close('all')
            
            # Side-by-side bars for easy comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(real_dist))
            width = 0.35
            
            ax.bar(x - width/2, real_dist.values(), width, label='Real Data', color='#4a90e2')
            ax.bar(x + width/2, synthetic_dist.values(), width, label='Synthetic Data', color='#f5a623')
            
            ax.set_xticks(x)
            ax.set_xticklabels(real_dist.keys())
            ax.set_title(f'Sentiment Distribution Comparison (KL Divergence: {kl_div:.4f})')
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            return str(output_path)
            
        finally:
            plt.close('all')
    
    def generate_token_overlap_venn(self, overlap_data: Dict[str, int]) -> str:
        """Generate Venn diagram for token overlap"""
        output_path = self.temp_dir / "token_overlap.png"
        
        try:
            # Clear any existing figures
            plt.close('all')
            
            plt.figure(figsize=(8, 8))
            venn2(subsets=(
                overlap_data['unique_tokens1'] - overlap_data['overlap_count'],
                overlap_data['unique_tokens2'] - overlap_data['overlap_count'],
                overlap_data['overlap_count']
            ), set_labels=('Real Data', 'Synthetic Data'))
            
            plt.title('Token Overlap Analysis')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            return str(output_path)
            
        finally:
            plt.close('all')
    
    def generate_ngram_plot(self, ngram_data: Dict[str, Any]) -> str:
        """Creates side-by-side frequency plots for top bigrams and trigrams"""
        output_path = self.temp_dir / "ngram_analysis.png"
        
        try:
            # Clear any existing figures
            plt.close('all')
            
            # Limit to top 10 for readability
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            bigrams = ngram_data['bigrams'][:10]
            axes[0].bar(range(len(bigrams)), [b[1] for b in bigrams], color='#9b59b6')
            axes[0].set_xticks(range(len(bigrams)))
            axes[0].set_xticklabels([b[0] for b in bigrams], rotation=45, ha='right')
            axes[0].set_title('Top 10 Bigrams')
            
            trigrams = ngram_data['trigrams'][:10]
            axes[1].bar(range(len(trigrams)), [t[1] for t in trigrams], color='#f39c12')
            axes[1].set_xticks(range(len(trigrams)))
            axes[1].set_xticklabels([t[0] for t in trigrams], rotation=45, ha='right')
            axes[1].set_title('Top 10 Trigrams')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            return str(output_path)
            
        finally:
            plt.close('all')