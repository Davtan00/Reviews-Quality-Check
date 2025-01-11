"""
Script to evaluate multiple Hugging Face sentiment-analysis models 
against a set of pre-labeled reviews (assumed perfectly labeled).
We compare each model's predictions to the gold labels (positive, negative, neutral)
and compute Accuracy and F1 (macro average) for each model.

Usage:
    python evaluate_models.py --input-file=/path/to/reviews.json
"""

import argparse
import json
import logging
import os
import sys
from typing import List, Dict

import pandas as pd
from transformers import pipeline
from sklearn.metrics import accuracy_score, f1_score
import torch

#Only 3way
MODEL_CHECKPOINTS = [
    "cardiffnlp/twitter-roberta-base-sentiment-latest",  # 3-way
    "finiteautomata/bertweet-base-sentiment-analysis",  # 3-way
    "siebert/sentiment-roberta-large-english"           # 3-way
]

def parse_arguments():
    """Parse command-line arguments for this script."""
    parser = argparse.ArgumentParser(description="Evaluate multiple HF models on JSON reviews.")
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to the JSON file containing labeled reviews. Must have a 'generated_data' field."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable more verbose logging."
    )
    return parser.parse_args()

def map_prediction_to_label(model_output: Dict) -> str:
    """
    Convert model pipeline output to one of: 'positive', 'negative', 'neutral'.
    For many 3-class sentiment models, the label is typically returned as 
    'positive', 'negative', or 'neutral', but sometimes it can be 'LABEL_0', etc.
    """
    raw_label = model_output["label"].lower()
    
    if "neg" in raw_label or raw_label.endswith("0"):
        return "negative"
    elif "neu" in raw_label or raw_label.endswith("1"):
        return "neutral"
    elif "pos" in raw_label or raw_label.endswith("2"):
        return "positive"
    else:
        return "neutral"

def evaluate_model_on_reviews(
    model_name_or_path: str,
    reviews: List[Dict[str, str]]
) -> Dict[str, float]:
    """
    Loads a sentiment-analysis pipeline for the given model checkpoint
    and evaluates it on the provided reviews.

    Returns:
        Dictionary with 'model_name', 'accuracy', and 'f1' (macro).
    """
    try:
        logging.info(f"Loading pipeline for model: {model_name_or_path}")
        sentiment_pipe = pipeline(
            "text-classification",
            model=model_name_or_path,
            tokenizer=model_name_or_path,
            truncation=True,
            max_length=512,
            device=0 if torch.cuda.is_available() else -1  # Use GPU if available
        )

        # Add more detailed logging
        logging.info(f"Processing {len(reviews)} reviews...")
        
        gold_labels = []
        predicted_labels = []
        total = len(reviews)

        for idx, review in enumerate(reviews, 1):
            if idx % 100 == 0:  # Log progress every 100 reviews
                logging.info(f"Processing review {idx}/{total} ({(idx/total)*100:.1f}%)")
            
            gold = review["sentiment"].lower()
            gold_labels.append(gold)
            
            # Run the pipeline
            output = sentiment_pipe(review["text"])[0]
            predicted = map_prediction_to_label(output)
            predicted_labels.append(predicted)

        logging.info(f"Completed evaluation for {model_name_or_path}")

        # Compute accuracy and macro-F1
        accuracy = accuracy_score(gold_labels, predicted_labels)
        f1 = f1_score(gold_labels, predicted_labels, average="macro")

        return {
            "model_name": model_name_or_path,
            "accuracy": accuracy,
            "f1": f1
        }

    except Exception as e:
        logging.error(f"Error processing model {model_name_or_path}: {str(e)}")
        return {
            "model_name": model_name_or_path,
            "accuracy": 0.0,
            "f1": 0.0,
            "error": str(e)
        }

def evaluate_all_models(reviews: List[Dict[str, str]]) -> pd.DataFrame:
    """
    Evaluate all models in MODEL_CHECKPOINTS on the given reviews 
    and return a DataFrame summarizing each model's accuracy and F1 score.
    """
    results = []
    for checkpoint in MODEL_CHECKPOINTS:
        metrics = evaluate_model_on_reviews(checkpoint, reviews)
        results.append(metrics)

    df = pd.DataFrame(results)
    return df

def main():
    args = parse_arguments()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # 1. Load the JSON file
    if not os.path.isfile(args.input_file):
        logging.error(f"Input file not found: {args.input_file}")
        sys.exit(1)

    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 2. Extract the "generated_data" list of reviews
    #    Each review dict must have "text" and "sentiment" keys
    reviews = data.get("generated_data", [])
    if not reviews:
        logging.error("No 'generated_data' found or it's empty. Exiting.")
        sys.exit(1)

    logging.info(f"Loaded {len(reviews)} reviews from {args.input_file}")

    # 3. Evaluate all models
    logging.info("Evaluating all models...")
    results_df = evaluate_all_models(reviews)

    # 4. Print results
    logging.info("\nEvaluation Results:")
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    main()
