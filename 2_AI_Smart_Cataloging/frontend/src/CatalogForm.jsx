/**
 * React component for AI-powered catalog creation
 * Allows sellers to upload images and review AI-generated product details
 */

import React, { useState, useRef } from 'react';
import axios from 'axios';

const CatalogForm = () => {
  // State management for form data
  const [selectedFile, setSelectedFile] = useState(null);
  const [aiGeneratedData, setAiGeneratedData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Form state for editable fields
  const [formData, setFormData] = useState({
    category: '',
    title: '',
    description: '',
    tags: []
  });
  
  // File input reference
  const fileInputRef = useRef(null);
  
  // API base URL (configure for your backend)
  const API_BASE_URL = 'http://localhost:8000';
  
  /**
   * Handle file selection from input
   */
  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setError(null);
      setAiGeneratedData(null);
      
      // Reset form data when new file is selected
      setFormData({
        category: '',
        title: '',
        description: '',
        tags: []
      });
    }
  };
  
  /**
   * Send image to backend API for AI processing
   */
  const generateProductDetails = async () => {
    if (!selectedFile) {
      setError('Please select an image file first');
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      // Create FormData for file upload
      const formData = new FormData();
      formData.append('file', selectedFile);
      
      // Send request to backend API
      const response = await axios.post(
        `${API_BASE_URL}/generate-details`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
          timeout: 30000, // 30 second timeout
        }
      );
      
      if (response.data.success) {
        const aiData = response.data.data;
        setAiGeneratedData(aiData);
        
        // Pre-populate form with AI-generated data
        setFormData({
          category: aiData.category,
          title: aiData.title,
          description: aiData.description,
          tags: aiData.suggested_tags
        });
        
        console.log('AI generated details:', aiData);
      } else {
        setError(response.data.error || 'Failed to generate product details');
      }
      
    } catch (err) {
      console.error('Error generating product details:', err);
      if (err.response?.data?.detail) {
        setError(err.response.data.detail);
      } else if (err.code === 'ECONNABORTED') {
        setError('Request timeout. Please try again.');
      } else {
        setError('Failed to connect to AI service. Please try again.');
      }
    } finally {
      setLoading(false);
    }
  };
  
  /**
   * Handle form field changes
   */
  const handleInputChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };
  
  /**
   * Handle tag addition/removal
   */
  const handleTagChange = (index, value) => {
    const newTags = [...formData.tags];
    newTags[index] = value;
    setFormData(prev => ({
      ...prev,
      tags: newTags
    }));
  };
  
  const addTag = () => {
    setFormData(prev => ({
      ...prev,
      tags: [...prev.tags, '']
    }));
  };
  
  const removeTag = (index) => {
    setFormData(prev => ({
      ...prev,
      tags: prev.tags.filter((_, i) => i !== index)
    }));
  };
  
  /**
   * Submit the final product data
   */
  const handleSubmit = (e) => {
    e.preventDefault();
    console.log('Submitting product data:', formData);
    // Here you would typically send the data to your main product creation API
    alert('Product details saved successfully!');
  };
  
  return (
    <div className="catalog-form-container">
      <h2>AI-Powered Product Cataloging</h2>
      
      {/* File Upload Section */}
      <div className="upload-section">
        <h3>Step 1: Upload Product Image</h3>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileSelect}
          className="file-input"
        />
        
        {selectedFile && (
          <div className="file-info">
            <p>Selected: {selectedFile.name}</p>
            <button 
              onClick={generateProductDetails}
              disabled={loading}
              className="generate-btn"
            >
              {loading ? 'Generating Details...' : 'Generate Product Details'}
            </button>
          </div>
        )}
      </div>
      
      {/* Error Display */}
      {error && (
        <div className="error-message">
          <p>Error: {error}</p>
        </div>
      )}
      
      {/* AI Generated Data Display */}
      {aiGeneratedData && (
        <div className="ai-results">
          <h3>Step 2: Review AI-Generated Details</h3>
          <div className="confidence-indicator">
            <span>AI Confidence: {(aiGeneratedData.confidence * 100).toFixed(1)}%</span>
          </div>
        </div>
      )}
      
      {/* Editable Form Section */}
      {(aiGeneratedData || formData.category) && (
        <form onSubmit={handleSubmit} className="product-form">
          <h3>Step 3: Review and Edit Details</h3>
          
          {/* Category Field */}
          <div className="form-group">
            <label htmlFor="category">Category:</label>
            <input
              type="text"
              id="category"
              value={formData.category}
              onChange={(e) => handleInputChange('category', e.target.value)}
              className="form-input"
              placeholder="Enter product category"
            />
          </div>
          
          {/* Title Field */}
          <div className="form-group">
            <label htmlFor="title">Product Title:</label>
            <input
              type="text"
              id="title"
              value={formData.title}
              onChange={(e) => handleInputChange('title', e.target.value)}
              className="form-input"
              placeholder="Enter product title"
            />
          </div>
          
          {/* Description Field */}
          <div className="form-group">
            <label htmlFor="description">Description:</label>
            <textarea
              id="description"
              value={formData.description}
              onChange={(e) => handleInputChange('description', e.target.value)}
              className="form-textarea"
              rows="4"
              placeholder="Enter product description"
            />
          </div>
          
          {/* Tags Section */}
          <div className="form-group">
            <label>Tags:</label>
            {formData.tags.map((tag, index) => (
              <div key={index} className="tag-input-group">
                <input
                  type="text"
                  value={tag}
                  onChange={(e) => handleTagChange(index, e.target.value)}
                  className="tag-input"
                  placeholder="Enter tag"
                />
                <button
                  type="button"
                  onClick={() => removeTag(index)}
                  className="remove-tag-btn"
                >
                  Remove
                </button>
              </div>
            ))}
            <button
              type="button"
              onClick={addTag}
              className="add-tag-btn"
            >
              Add Tag
            </button>
          </div>
          
          {/* Submit Button */}
          <button type="submit" className="submit-btn">
            Save Product Details
          </button>
        </form>
      )}
    </div>
  );
};

export default CatalogForm;
