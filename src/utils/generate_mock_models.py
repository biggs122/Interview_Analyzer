#!/usr/bin/env python3
"""
Generate mock models for the Interview Analyzer

This script creates simple mock models for use in testing the Interview Analyzer application.
In a real implementation, these would be actual trained ML models.
"""

import os
import pickle
import logging
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define model directory
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")

def create_directory_if_not_exists(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def create_mock_cv_classifier():
    """Create a mock CV classifier model"""
    # Create a simple logistic regression model
    X = np.random.rand(100, 10)  # 100 samples, 10 features
    y = np.random.randint(0, 5, 100)  # 5 classes
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    
    # Save the model
    model_path = os.path.join(MODEL_DIR, "cv_classifier.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"Created mock CV classifier model: {model_path}")

def create_mock_interview_analysis_model():
    """Create a mock interview analysis model"""
    # Create a simple random forest model
    X = np.random.rand(100, 20)  # 100 samples, 20 features
    y = np.random.rand(100)  # Regression target
    
    model = RandomForestClassifier(n_estimators=10)
    model.fit(X, np.random.randint(0, 3, 100))  # 3 classes
    
    # Save the model
    model_path = os.path.join(MODEL_DIR, "interview_analysis.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"Created mock interview analysis model: {model_path}")

def create_mock_emotion_detection_model():
    """Create a mock emotion detection model"""
    # Create a simple logistic regression model
    X = np.random.rand(100, 50)  # 100 samples, 50 features (image features)
    y = np.random.randint(0, 7, 100)  # 7 emotions
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    
    # Save the model
    model_path = os.path.join(MODEL_DIR, "emotion_detection.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"Created mock emotion detection model: {model_path}")

def create_mock_speech_recognition_model():
    """Create a mock speech recognition model"""
    # Create a simple random forest model
    X = np.random.rand(100, 30)  # 100 samples, 30 features (audio features)
    y = np.random.randint(0, 10, 100)  # 10 different speech classes
    
    model = RandomForestClassifier(n_estimators=10)
    model.fit(X, y)
    
    # Save the model
    model_path = os.path.join(MODEL_DIR, "speech_recognition.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"Created mock speech recognition model: {model_path}")

def main():
    """Generate all mock models"""
    logger.info("Generating mock models for Interview Analyzer")
    
    # Create models directory if it doesn't exist
    create_directory_if_not_exists(MODEL_DIR)
    
    # Create mock models
    create_mock_cv_classifier()
    create_mock_interview_analysis_model()
    create_mock_emotion_detection_model()
    create_mock_speech_recognition_model()
    
    logger.info("All mock models generated successfully")

if __name__ == "__main__":
    main() 