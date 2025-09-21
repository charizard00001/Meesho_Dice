"""
Flask API for Automated Catalog Quality Check
Provides endpoint for product category validation
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
import requests
from ml_model import ProductCategorizationModel
import logging

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the ML model
try:
    category_model = ProductCategorizationModel()
    logger.info("ML model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load ML model: {e}")
    category_model = None

@app.route('/check_category', methods=['POST'])
def check_category():
    """
    API endpoint to validate and predict product category
    
    Expected JSON payload:
    {
        "image_url": "https://example.com/product.jpg",
        "title": "Product Title",
        "description": "Product description",
        "seller_category": "Electronics"  # Category selected by seller
    }
    
    Returns:
    {
        "predicted_category": "Electronics",
        "seller_category": "Electronics", 
        "match": true,
        "confidence": 0.95,
        "suggestion": "Category matches prediction"
    }
    """
    try:
        # Validate request data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        required_fields = ['image_url', 'title', 'description', 'seller_category']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Extract data from request
        image_url = data['image_url']
        title = data['title']
        description = data['description']
        seller_category = data['seller_category']
        
        # Combine title and description for ML processing
        full_description = f"{title} {description}"
        
        # Download and process image
        image_path = download_image(image_url)
        if not image_path:
            return jsonify({"error": "Failed to download image"}), 400
        
        # Get ML model prediction
        if category_model:
            prediction_result = category_model.predict_category(image_path, full_description)
            predicted_category = prediction_result['predicted_category']
            confidence = prediction_result['confidence']
        else:
            # Fallback when model is not available
            predicted_category = "Unknown"
            confidence = 0.0
        
        # Check if seller's category matches prediction
        category_match = predicted_category.lower() == seller_category.lower()
        
        # Generate suggestion based on match
        if category_match:
            suggestion = "Category matches AI prediction"
        else:
            suggestion = f"Consider changing to '{predicted_category}' for better discoverability"
        
        # Clean up temporary image file
        if os.path.exists(image_path):
            os.remove(image_path)
        
        # Return response
        response = {
            "predicted_category": predicted_category,
            "seller_category": seller_category,
            "match": category_match,
            "confidence": confidence,
            "suggestion": suggestion,
            "status": "success"
        }
        
        logger.info(f"Category check completed: {predicted_category} vs {seller_category}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in check_category endpoint: {e}")
        return jsonify({"error": "Internal server error", "status": "error"}), 500

def download_image(image_url: str) -> str:
    """
    Download image from URL to temporary file
    
    Args:
        image_url: URL of the image to download
        
    Returns:
        Path to downloaded image file, or None if failed
    """
    try:
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        temp_path = temp_file.name
        temp_file.close()
        
        # Download image
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        
        # Save image to temporary file
        with open(temp_path, 'wb') as f:
            f.write(response.content)
        
        return temp_path
        
    except Exception as e:
        logger.error(f"Failed to download image from {image_url}: {e}")
        return None

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    return jsonify({
        "status": "healthy",
        "model_loaded": category_model is not None
    })

@app.route('/', methods=['GET'])
def root():
    """
    Root endpoint with API information
    """
    return jsonify({
        "service": "Automated Catalog Quality Check API",
        "version": "1.0.0",
        "endpoints": {
            "POST /check_category": "Validate product category",
            "GET /health": "Health check"
        }
    })

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
