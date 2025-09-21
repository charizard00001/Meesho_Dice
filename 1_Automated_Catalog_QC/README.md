# Automated Catalog Quality Check

## Pain Point
Mis-categorized products harm user experience and efficiency. Sellers often incorrectly categorize their products, leading to poor search results and reduced discoverability.

## Proposed Solution
An ML model that automatically validates and corrects product categories using computer vision and natural language processing. The system analyzes product images and descriptions to predict the correct category and flags mismatches for seller review.

## Architecture
- **Computer Vision**: Analyzes product images to identify visual features
- **NLP**: Processes product titles and descriptions for semantic understanding
- **Fusion Model**: Combines image and text features for accurate categorization
- **API Integration**: Real-time category validation during product upload

## Files
- `src/ml_model.py`: Core ML model for category prediction
- `src/api.py`: Flask API for category validation endpoint
- `requirements.txt`: Python dependencies

## Usage
The system automatically validates product categories during upload and provides suggestions for corrections, improving overall catalog quality and user experience.
