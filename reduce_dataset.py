import json
import random
from typing import Dict, List
import logging

def reduce_dataset(input_file: str, output_file: str, target_size: int = 5000) -> None:
    """
    Reduce dataset while maintaining sentiment distribution ratios.
    Handles Unicode properly and renumbers IDs sequentially.
    
    Args:
        input_file: Path to original JSON file
        output_file: Path to save reduced JSON file
        target_size: Desired size of reduced dataset (default: 1000)
    """
    logging.basicConfig(level=logging.INFO)
    
    # Load original data with proper encoding
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    original_reviews = data['generated_data']
    original_dist = data['summary']['sentiment_distribution']
    
    # Calculate ratios for new distribution
    total = sum(original_dist.values())
    ratios = {
        sentiment: int((count / total) * target_size)
        for sentiment, count in original_dist.items()
    }
    
    logging.info(f"Original distribution: {original_dist}")
    logging.info(f"Target distribution: {ratios}")
    
    # Group reviews by sentiment
    sentiment_groups = {
        'positive': [],
        'negative': [],
        'neutral': []
    }
    
    for review in original_reviews:
        sentiment_groups[review['sentiment']].append(review)
    
    # Sample from each group according to ratios
    reduced_reviews = []
    for sentiment, count in ratios.items():
        sampled = random.sample(sentiment_groups[sentiment], count)
        reduced_reviews.extend(sampled)
    
    # Shuffle the combined results
    random.shuffle(reduced_reviews)
    
    # Renumber IDs sequentially
    for i, review in enumerate(reduced_reviews, 1):
        review['id'] = i
    
    # Create new JSON with same structure
    reduced_data = {
        'domain': data['domain'],
        'generated_data': reduced_reviews,
        'summary': {
            'total_generated': len(reduced_reviews),
            'sentiment_distribution': {
                sentiment: len([r for r in reduced_reviews if r['sentiment'] == sentiment])
                for sentiment in ['positive', 'negative', 'neutral']
            }
        }
    }
    
    # Save reduced dataset with proper encoding
    with open(output_file, 'w', encoding='utf-8', errors='ignore') as f:
        json.dump(reduced_data, f, indent=2, ensure_ascii=False)
    
    logging.info(f"Reduced dataset saved to {output_file}")
    logging.info(f"Final distribution: {reduced_data['summary']['sentiment_distribution']}")

if __name__ == "__main__":
    input_file = "HC_Jan_summary.json"
    output_file = "Generated Data/HC_Jan_5k.json"
    reduce_dataset(input_file, output_file) 