# Adding Models and Domains to the Sentiment Analysis System

## Model Configuration Guide

### Basic Structure


class ModelConfig:
    SUPPORTED_MODELS = {
        'your-model-key': {
            'name': 'org/model-name',       # HuggingFace model path
            'type': 'three-class',         # Model type
            'mapping': {                   # Class mapping
                0: 'negative',
                1: 'neutral',
                2: 'positive'
            },
            'description': 'Model description'  # Brief description
        }
    }

'binary-example': {
    'name': 'org/binary-sentiment',
    'type': 'binary',
    'mapping': {0: 'negative', 1: 'positive'},
    'description': 'Fast binary classifier with rule-based neutral detection'
}

'three-class-example': {
    'name': 'org/three-class-sentiment',
    'type': 'three-class',
    'mapping': {
        0: 'negative',
        1: 'neutral',
        2: 'positive'
    },
    'description': 'Native three-way sentiment classifier'
}

'five-class-example': {
    'name': 'org/five-class-sentiment',
    'type': 'five-class',
    'mapping': {
        0: 'negative',  # Very negative
        1: 'negative',  # Negative
        2: 'neutral',   # Neutral
        3: 'positive',  # Positive
        4: 'positive'   # Very positive
    },
    'description': 'Fine-grained sentiment mapped to three classes'
}

# Model Requirements
1.Must be available on HuggingFace Hub.
2.Compatible with AutoModelForSequenceClassification.
3.Maximum sequence length: 512 tokens.
4.Must output logits for sentiment classification.
5.Output must match mapping dictionary keys.

By implementing more logic these constraints could be adapted.


## Domain Configuration Guide

### Basic Structure

class DomainIndicators:
    INDICATORS = {
        'your-domain': {
            'positive': {'term1', 'term2', ...},      # Positive indicators
            'negative': {'term1', 'term2', ...},      # Negative indicators
            'neutral_markers': {'term1', 'term2', ...}  # Neutral indicators
        }
    }


'e-commerce': {
    'positive': {
        'fast-shipping',
        'great-value',
        'high-quality',
        'excellent-service',
        'reliable-delivery'
    },
    'negative': {
        'poor-packaging',
        'delayed-shipping',
        'damaged-item',
        'overpriced',
        'unreliable-seller'
    },
    'neutral_markers': {
        'standard-shipping',
        'average-quality',
        'as-described',
        'basic-packaging'
    }
}

# Domain Requirements

## Minimum Indicators

- **Positive terms**: ≥ 5  
- **Negative terms**: ≥ 5  
- **Neutral markers**: ≥ 3  

---

## Term Guidelines

1. Use domain-specific terminology.
2. Include compound terms (hyphenated).
3. Consider common variations.
4. Focus on unambiguous indicators.
5. Include contextual markers.

---

## Best Practices

- Use lowercase terms.
- Hyphenate multi-word terms.
- Avoid ambiguous terms.
- Include domain-specific jargon.
- Consider regional variations.
