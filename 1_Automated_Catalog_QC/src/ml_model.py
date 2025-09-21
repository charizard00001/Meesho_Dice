"""
Product Categorization ML Model
Combines computer vision and NLP to predict product categories
"""

import torch
import torchvision.transforms as transforms
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import numpy as np
from typing import Dict, Tuple, Optional

class ProductCategorizationModel:
    """
    ML model for automatic product categorization using image and text features
    """
    
    def __init__(self, model_path: str = "pretrained_models/category_classifier.pth"):
        """
        Initialize the model with pre-trained weights and tokenizer
        
        Args:
            model_path: Path to the pre-trained model weights
        """
        # Load pre-trained computer vision model for image feature extraction
        # This would typically be a ResNet or Vision Transformer fine-tuned on product images
        self.image_model = torch.load(model_path, map_location='cpu')
        self.image_model.eval()
        
        # Load pre-trained NLP model for text feature extraction
        # Using a transformer model fine-tuned on product descriptions
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.text_model = AutoModel.from_pretrained("bert-base-uncased")
        
        # Define image preprocessing pipeline
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Category mapping (simplified for prototype)
        self.category_mapping = {
            0: "Electronics",
            1: "Clothing",
            2: "Home & Garden",
            3: "Beauty & Health",
            4: "Sports & Fitness",
            5: "Books & Media"
        }
    
    def predict_category(self, image_path: str, description: str) -> Dict[str, any]:
        """
        Predict product category from image and description
        
        Args:
            image_path: Path to product image
            description: Product title and description text
            
        Returns:
            Dictionary containing predicted category and confidence score
        """
        try:
            # Computer Vision Processing
            # Load and preprocess the image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.image_transform(image).unsqueeze(0)
            
            # Extract visual features using CNN
            with torch.no_grad():
                image_features = self.image_model(image_tensor)
                # Apply global average pooling to get fixed-size feature vector
                image_features = torch.mean(image_features, dim=[2, 3])
            
            # NLP Processing
            # Tokenize and encode the text description
            text_inputs = self.tokenizer(
                description, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=512
            )
            
            # Extract text features using transformer
            with torch.no_grad():
                text_outputs = self.text_model(**text_inputs)
                # Use [CLS] token representation as text features
                text_features = text_outputs.last_hidden_state[:, 0, :]
            
            # Feature Fusion
            # Combine image and text features for final prediction
            # In practice, this would use attention mechanisms or concatenation
            combined_features = torch.cat([image_features, text_features], dim=1)
            
            # Category Classification
            # Apply final classification layer to predict category
            # This would be a trained linear layer mapping features to categories
            category_logits = self._classify_category(combined_features)
            predicted_category_id = torch.argmax(category_logits, dim=1).item()
            confidence = torch.softmax(category_logits, dim=1).max().item()
            
            return {
                "predicted_category": self.category_mapping[predicted_category_id],
                "confidence": confidence,
                "category_id": predicted_category_id
            }
            
        except Exception as e:
            # Fallback to text-only prediction if image processing fails
            return self._text_only_prediction(description)
    
    def _classify_category(self, features: torch.Tensor) -> torch.Tensor:
        """
        Classify features into product categories
        
        Args:
            features: Combined image and text features
            
        Returns:
            Category logits
        """
        # This would be a trained classification head
        # For prototype, return random logits
        num_categories = len(self.category_mapping)
        return torch.randn(1, num_categories)
    
    def _text_only_prediction(self, description: str) -> Dict[str, any]:
        """
        Fallback prediction using only text when image processing fails
        
        Args:
            description: Product description text
            
        Returns:
            Dictionary with text-based category prediction
        """
        # Simple keyword-based fallback for prototype
        description_lower = description.lower()
        
        if any(word in description_lower for word in ['phone', 'laptop', 'electronic']):
            category_id = 0
        elif any(word in description_lower for word in ['shirt', 'dress', 'clothing']):
            category_id = 1
        else:
            category_id = 2  # Default to Home & Garden
        
        return {
            "predicted_category": self.category_mapping[category_id],
            "confidence": 0.7,  # Lower confidence for text-only
            "category_id": category_id
        }
