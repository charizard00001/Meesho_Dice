"""
FastAPI backend for AI-Powered Smart Cataloging
Generates product details from image uploads
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional
import uvicorn
import logging
import asyncio
from PIL import Image
import io

# Initialize FastAPI app
app = FastAPI(
    title="AI Smart Cataloging API",
    description="Generate product details from images using AI",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductDetails(BaseModel):
    """Response model for generated product details"""
    category: str
    title: str
    description: str
    confidence: float
    suggested_tags: list[str]

class AIResponse(BaseModel):
    """Response model for AI service calls"""
    success: bool
    data: Optional[ProductDetails]
    error: Optional[str]

async def call_ai_service(image: bytes) -> AIResponse:
    """
    Simulate calling an external AI service for product detail generation
    
    Args:
        image: Image bytes from uploaded file
        
    Returns:
        AIResponse with generated product details
    """
    try:
        # Simulate AI processing delay
        await asyncio.sleep(1.5)
        
        # Load and analyze image using PIL
        image_obj = Image.open(io.BytesIO(image))
        width, height = image_obj.size
        
        # Mock AI analysis based on image characteristics
        # In production, this would call actual AI services like:
        # - Google Vision API for object detection
        # - OpenAI GPT for text generation
        # - Custom trained models for product classification
        
        # Simulate category prediction based on image size and format
        if width > height:  # Landscape orientation
            category = "Home & Garden"
            title = "Premium Home Decor Item"
            description = "Beautiful home decoration piece that adds elegance to any space. Perfect for modern interiors and contemporary design."
        else:  # Portrait orientation
            category = "Fashion & Accessories"
            title = "Stylish Fashion Accessory"
            description = "Trendy and fashionable accessory that complements your style. High-quality materials and contemporary design."
        
        # Simulate confidence score
        confidence = 0.85
        
        # Generate suggested tags based on category
        tag_mapping = {
            "Home & Garden": ["decor", "home", "interior", "modern", "elegant"],
            "Fashion & Accessories": ["fashion", "style", "trendy", "accessory", "quality"]
        }
        suggested_tags = tag_mapping.get(category, ["product", "quality", "new"])
        
        product_details = ProductDetails(
            category=category,
            title=title,
            description=description,
            confidence=confidence,
            suggested_tags=suggested_tags
        )
        
        return AIResponse(
            success=True,
            data=product_details,
            error=None
        )
        
    except Exception as e:
        logger.error(f"Error in AI service: {e}")
        return AIResponse(
            success=False,
            data=None,
            error=str(e)
        )

@app.post("/generate-details", response_model=AIResponse)
async def generate_product_details(file: UploadFile = File(...)):
    """
    Generate product details from uploaded image
    
    Args:
        file: Uploaded image file
        
    Returns:
        Generated product details including category, title, and description
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400, 
                detail="File must be an image"
            )
        
        # Validate file size (max 10MB)
        file_size = 0
        content = await file.read()
        file_size = len(content)
        
        if file_size > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(
                status_code=400,
                detail="File size must be less than 10MB"
            )
        
        logger.info(f"Processing image: {file.filename}, size: {file_size} bytes")
        
        # Call AI service to generate product details
        ai_response = await call_ai_service(content)
        
        if not ai_response.success:
            raise HTTPException(
                status_code=500,
                detail=f"AI service error: {ai_response.error}"
            )
        
        logger.info(f"Successfully generated details for {file.filename}")
        return ai_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in generate-details: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "service": "AI Smart Cataloging API",
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    """
    Root endpoint with API information
    """
    return {
        "service": "AI Smart Cataloging API",
        "version": "1.0.0",
        "endpoints": {
            "POST /generate-details": "Generate product details from image",
            "GET /health": "Health check"
        }
    }

if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
