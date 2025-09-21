/**
 * React component for product listing with direct link to live product
 * Displays product information and provides one-click access to live listing
 */

import React, { useState } from 'react';

const ProductListing = ({ 
  product_id, 
  status, 
  title, 
  price, 
  image_url, 
  category,
  views,
  sales,
  lastUpdated 
}) => {
  // State for tracking link clicks
  const [linkClicked, setLinkClicked] = useState(false);
  
  // Base URL for Meesho product pages (configure for production)
  const MEESHO_BASE_URL = 'https://meesho.com/product';
  
  /**
   * Generate direct link to live product on Meesho
   */
  const generateProductLink = (productId) => {
    return `${MEESHO_BASE_URL}/${productId}`;
  };
  
  /**
   * Handle click on "View on Meesho" button
   */
  const handleViewOnMeesho = () => {
    if (status === 'Live') {
      const productLink = generateProductLink(product_id);
      
      // Track the click for analytics
      setLinkClicked(true);
      
      // Open link in new tab
      window.open(productLink, '_blank', 'noopener,noreferrer');
      
      // Log analytics event (in production, send to analytics service)
      console.log('Product link clicked:', {
        product_id,
        link: productLink,
        timestamp: new Date().toISOString()
      });
      
      // Reset click state after 2 seconds
      setTimeout(() => setLinkClicked(false), 2000);
    }
  };
  
  /**
   * Get status badge styling
   */
  const getStatusBadge = (status) => {
    const statusConfig = {
      'Live': { className: 'status-badge live', text: 'Live' },
      'Pending': { className: 'status-badge pending', text: 'Pending Review' },
      'Rejected': { className: 'status-badge rejected', text: 'Rejected' },
      'Draft': { className: 'status-badge draft', text: 'Draft' },
      'Paused': { className: 'status-badge paused', text: 'Paused' }
    };
    
    const config = statusConfig[status] || statusConfig['Draft'];
    return (
      <span className={config.className}>
        {config.text}
      </span>
    );
  };
  
  /**
   * Format price for display
   */
  const formatPrice = (price) => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      minimumFractionDigits: 0
    }).format(price);
  };
  
  /**
   * Format date for display
   */
  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-IN', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  };
  
  return (
    <div className="product-listing-card">
      {/* Product Image */}
      <div className="product-image-container">
        <img 
          src={image_url || '/placeholder-product.jpg'} 
          alt={title}
          className="product-image"
          onError={(e) => {
            e.target.src = '/placeholder-product.jpg';
          }}
        />
      </div>
      
      {/* Product Information */}
      <div className="product-info">
        <div className="product-header">
          <h3 className="product-title">{title}</h3>
          <div className="product-meta">
            {getStatusBadge(status)}
            <span className="product-id">ID: {product_id}</span>
          </div>
        </div>
        
        <div className="product-details">
          <div className="detail-row">
            <span className="detail-label">Category:</span>
            <span className="detail-value">{category}</span>
          </div>
          
          <div className="detail-row">
            <span className="detail-label">Price:</span>
            <span className="detail-value price">{formatPrice(price)}</span>
          </div>
          
          {status === 'Live' && (
            <>
              <div className="detail-row">
                <span className="detail-label">Views:</span>
                <span className="detail-value">{views?.toLocaleString() || 0}</span>
              </div>
              
              <div className="detail-row">
                <span className="detail-label">Sales:</span>
                <span className="detail-value">{sales || 0}</span>
              </div>
            </>
          )}
          
          <div className="detail-row">
            <span className="detail-label">Last Updated:</span>
            <span className="detail-value">{formatDate(lastUpdated)}</span>
          </div>
        </div>
      </div>
      
      {/* Action Buttons */}
      <div className="product-actions">
        {/* View on Meesho Button - Only show for Live products */}
        {status === 'Live' && (
          <button
            onClick={handleViewOnMeesho}
            className={`view-on-meesho-btn ${linkClicked ? 'clicked' : ''}`}
            title="View this product on Meesho"
          >
            {linkClicked ? 'Opening...' : 'View on Meesho'}
            <svg 
              className="external-link-icon" 
              width="16" 
              height="16" 
              viewBox="0 0 24 24" 
              fill="none" 
              stroke="currentColor" 
              strokeWidth="2"
            >
              <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path>
              <polyline points="15,3 21,3 21,9"></polyline>
              <line x1="10" y1="14" x2="21" y2="3"></line>
            </svg>
          </button>
        )}
        
        {/* Edit Button - Available for all statuses */}
        <button
          className="edit-product-btn"
          onClick={() => console.log('Edit product:', product_id)}
          title="Edit product details"
        >
          Edit
        </button>
        
        {/* Status-specific actions */}
        {status === 'Pending' && (
          <button
            className="check-status-btn"
            onClick={() => console.log('Check status for:', product_id)}
            title="Check review status"
          >
            Check Status
          </button>
        )}
        
        {status === 'Rejected' && (
          <button
            className="view-feedback-btn"
            onClick={() => console.log('View feedback for:', product_id)}
            title="View rejection feedback"
          >
            View Feedback
          </button>
        )}
      </div>
      
      {/* Link Preview (for Live products) */}
      {status === 'Live' && (
        <div className="link-preview">
          <small className="link-text">
            Direct link: {generateProductLink(product_id)}
          </small>
        </div>
      )}
    </div>
  );
};

// Default props for demonstration
ProductListing.defaultProps = {
  title: 'Sample Product',
  price: 999,
  image_url: null,
  category: 'Electronics',
  views: 0,
  sales: 0,
  lastUpdated: new Date().toISOString()
};

export default ProductListing;
