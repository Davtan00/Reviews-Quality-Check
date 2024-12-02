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
        
    def generate_sentiment_distribution(self, sentiment_data: Dict[str, int], lengths: List[int]) -> str:
        """Generates side-by-side plots for sentiment distribution and review length analysis"""
        output_path = self.temp_dir / "sentiment_distribution.png"
        
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        
        # Color-coded sentiment bars for intuitive interpretation
        sentiments = list(sentiment_data.keys())
        counts = list(sentiment_data.values())
        ax[0].bar(sentiments, counts, color=['#6cba6b', '#f16a6a', '#d1d1d1'])
        ax[0].set_title('Sentiment Distribution')
        ax[0].set_xlabel('Sentiment')
        ax[0].set_ylabel('Frequency')
        
        # KDE plot for better visualization of length distribution patterns
        sns.histplot(lengths, kde=True, label='Reviews', color='#f79c42', 
                    stat='density', ax=ax[1])
        ax[1].set_title('Review Length Distribution')
        ax[1].set_xlabel('Review Length')
        ax[1].set_ylabel('Density')
        ax[1].legend()

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        return str(output_path)
    
    def generate_wordcloud(self, texts: List[str]) -> str:
        """Generate wordcloud visualization"""
        output_path = self.temp_dir / "wordcloud.png"
        
        wordcloud = WordCloud(width=800, height=400, 
                            background_color='white').generate(" ".join(texts))
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(output_path)
        plt.close()
        
        return str(output_path)
    
    def generate_kl_divergence_plot(self, real_dist: Dict[str, float], 
                                  synthetic_dist: Dict[str, float],
                                  kl_div: float) -> str:
        """Visualizes distribution differences between real and synthetic data"""
        output_path = self.temp_dir / "kl_divergence.png"
        
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
        plt.savefig(output_path)
        plt.close()
        
        return str(output_path)
    
    def generate_token_overlap_venn(self, overlap_data: Dict[str, int]) -> str:
        """Generate Venn diagram for token overlap"""
        output_path = self.temp_dir / "token_overlap.png"
        
        plt.figure(figsize=(8, 8))
        venn2(subsets=(
            overlap_data['unique_tokens1'] - overlap_data['overlap_count'],
            overlap_data['unique_tokens2'] - overlap_data['overlap_count'],
            overlap_data['overlap_count']
        ), set_labels=('Real Data', 'Synthetic Data'))
        
        plt.title('Token Overlap Analysis')
        plt.savefig(output_path)
        plt.close()
        
        return str(output_path)
    
    def generate_ngram_plot(self, ngram_data: Dict[str, Any]) -> str:
        """Creates side-by-side frequency plots for top bigrams and trigrams"""
        output_path = self.temp_dir / "ngram_analysis.png"
        
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
        plt.savefig(output_path)
        plt.close()
        
        return str(output_path)