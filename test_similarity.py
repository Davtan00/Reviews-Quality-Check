from analyzers.similarity import SophisticatedSimilarityAnalyzer
import logging

def test_ngram_similarity():
    analyzer = SophisticatedSimilarityAnalyzer()
    
    # Test pairs with varying similarity levels
    test_pairs = [
        # Pair 1 - High Similarity (Should be well above 0.736)
        (
            "The phone's battery life is excellent, lasting all day with heavy usage.",
            "The phone has excellent battery life that lasts all day under heavy use."
        ),
        
        # Pair 2 - Borderline Similarity (Should be around 0.736)
        (
            "The screen quality is good with vibrant colors, but viewing angles could be better.",
            "Display shows nice vibrant colors, though the viewing angles aren't perfect."
        ),
        
        # Pair 3 - Lower Similarity (Should be below 0.736)
        (
            "The camera takes great photos in daylight with natural colors.",
            "While the photo quality is decent, the colors seem a bit oversaturated."
        ),
        
        # Pair 4 - Very Different (Should have very low similarity)
        (
            "The build quality is excellent with premium materials.",
            "The software needs more updates to fix various bugs."
        )
    ]
    
    print("\nN-gram Similarity Test Results (threshold = 0.736):\n")
    for i, (text1, text2) in enumerate(test_pairs, 1):
        similarity = analyzer.get_ngram_similarity(text1, text2)
        print(f"Pair {i} (similarity = {similarity:.3f}):")
        print(f"Text 1: {text1}")
        print(f"Text 2: {text2}")
        print("-" * 80)

if __name__ == "__main__":
    test_ngram_similarity()
