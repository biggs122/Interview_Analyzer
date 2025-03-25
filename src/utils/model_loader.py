import os
import logging
import numpy as np
import json
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define model locations - can be adjusted based on actual paths
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")

# Model file paths
CV_CLASSIFIER_MODEL = os.path.join(MODEL_DIR, "cv_classifier.pkl")
INTERVIEW_ANALYSIS_MODEL = os.path.join(MODEL_DIR, "interview_analysis.pkl")
EMOTION_DETECTION_MODEL = os.path.join(MODEL_DIR, "emotion_detection.pkl")
SPEECH_RECOGNITION_MODEL = os.path.join(MODEL_DIR, "speech_recognition.pkl")

class ModelLoader:
    """Utility for loading pre-trained models from the models directory"""
    
    def __init__(self):
        """Initialize the model loader"""
        self.models = {}
        self.model_paths = {
            "cv_classifier": CV_CLASSIFIER_MODEL,
            "interview_analysis": INTERVIEW_ANALYSIS_MODEL,
            "emotion_detection": EMOTION_DETECTION_MODEL,
            "speech_recognition": SPEECH_RECOGNITION_MODEL
        }
        
        # Check if models directory exists
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR, exist_ok=True)
            logger.warning(f"Models directory not found. Created directory at {MODEL_DIR}")
    
    def load_model(self, model_name):
        """
        Load a model from file
        
        Args:
            model_name (str): Name of the model to load
            
        Returns:
            The loaded model or None if not found
        """
        if model_name in self.models:
            return self.models[model_name]
            
        model_path = self.model_paths.get(model_name)
        if not model_path:
            logger.error(f"Unknown model name: {model_name}")
            return None
            
        if not os.path.exists(model_path):
            logger.warning(f"Model not found at path: {model_path}")
            return None
            
        try:
            logger.info(f"Loading model: {model_name}")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            self.models[model_name] = model
            logger.info(f"Successfully loaded model: {model_name}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            return None
    
    def get_model_path(self, model_name):
        """Get the path to a model file"""
        return self.model_paths.get(model_name)
    
    def list_available_models(self):
        """List all available models in the models directory"""
        available = []
        for name, path in self.model_paths.items():
            if os.path.exists(path):
                available.append(name)
        
        return available

# Singleton instance
model_loader = ModelLoader() 