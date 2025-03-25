#!/usr/bin/env python3
"""
Advanced Model Testing Script

This script tests all the advanced AI models to verify they're working properly
with the new deep learning capabilities.
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path

# Add the src directory to the path
src_dir = Path(__file__).resolve().parent
sys.path.append(str(src_dir))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the advanced model loader
from utils.advanced_model_loader import AdvancedModelLoader

def main():
    """Run tests for all advanced models"""
    logger.info("Starting advanced model testing")
    
    # Initialize the model loader
    model_loader = AdvancedModelLoader(use_gpu=False)
    
    # Get version information
    version_info = model_loader.get_version_info()
    current_version = version_info.get("current_version", "unknown")
    logger.info(f"Current model version: {current_version}")
    
    # TEST 1: CV Classification
    logger.info("Testing CV Classification model")
    sample_cvs = [
        """
        John Smith
        Software Engineer

        SKILLS:
        Programming: Python, Java, C++, JavaScript
        Frameworks: TensorFlow, PyTorch, React, Django
        
        EXPERIENCE:
        Amazon - Senior Machine Learning Engineer (2018-2023)
        - Developed deep learning models for product recommendations
        - Improved conversion rates by 15% with personalized algorithms
        """,
        
        """
        Sarah Johnson
        Marketing Manager

        SKILLS:
        - Brand strategy development
        - Campaign management
        - Team leadership
        - Budget planning
        
        EXPERIENCE:
        Apple - Marketing Director (2019-2023)
        - Led rebranding initiatives for product lines
        - Managed team of 15 marketing professionals
        - Responsible for $2M marketing budget
        """
    ]
    
    for i, cv in enumerate(sample_cvs):
        category = model_loader.predict_cv_category(cv)
        logger.info(f"CV {i+1} classified as: {category}")
    
    # TEST 2: Interview Content Analysis
    logger.info("Testing Interview Analysis model")
    sample_responses = [
        """
        I've been working with Python for over 5 years, specializing in machine learning applications.
        At my previous job, I implemented a recommendation system that increased user engagement by 25%.
        I'm particularly skilled in TensorFlow and PyTorch for building deep learning models.
        """,
        
        """
        Um, I have some experience with programming. I did a few projects in college and
        worked on some small things. I think I'm pretty good at it but haven't done much
        professionally yet.
        """
    ]
    
    for i, response in enumerate(sample_responses):
        analysis = model_loader.analyze_interview_content(response)
        logger.info(f"Response {i+1} analysis:")
        for metric, score in analysis.items():
            logger.info(f"  - {metric}: {score:.2f}")
    
    # TEST 3: Emotion Detection (using random image simulation)
    logger.info("Testing Emotion Detection model")
    # Create random 224x224x3 RGB images to simulate facial images
    sample_images = [
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    ]
    
    for i, image in enumerate(sample_images):
        emotion = model_loader.detect_emotion(image)
        logger.info(f"Image {i+1} emotion detected: {emotion}")
    
    # TEST 4: Speech Recognition
    logger.info("Testing Speech Recognition model")
    # We'll simulate audio file paths here
    sample_audio_files = [
        str(Path(src_dir) / ".." / "data" / "sample_audio.wav")
    ]
    
    for i, audio_file in enumerate(sample_audio_files):
        if os.path.exists(audio_file):
            transcription = model_loader.transcribe_speech(audio_file)
            logger.info(f"Audio {i+1} transcription: {transcription[:100]}...")
        else:
            # Simulate audio features for quality assessment
            audio_features = np.random.random(20)  # 20 MFCC features
            quality = model_loader.assess_speech_quality(audio_features)
            logger.info(f"Audio {i+1} quality assessment:")
            for metric, score in quality.items():
                logger.info(f"  - {metric}: {score:.2f}")
    
    # Print performance metrics
    metrics = model_loader.get_performance_metrics()
    logger.info("Model performance metrics:")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"  - {key}: {value:.4f}")
    
    logger.info("All model tests completed")

if __name__ == "__main__":
    main() 