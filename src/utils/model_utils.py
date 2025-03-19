import os
import pickle
import sys
import torch
from pathlib import Path
import numpy as np
from transformers import pipeline
from deepface import DeepFace
import mediapipe as mp

def ensure_dir(dir_path):
    """Ensure directory exists"""
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

def save_model_to_pkl(model, output_path, verbose=True):
    """Save a model to pickle format"""
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)
        if verbose:
            print(f"Model successfully saved to {output_path}")
        return True
    except Exception as e:
        if verbose:
            print(f"Error saving model to {output_path}: {e}")
        return False

def load_model_from_pkl(model_path, verbose=True):
    """Load a model from pickle format"""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        if verbose:
            print(f"Model successfully loaded from {model_path}")
        return model
    except Exception as e:
        if verbose:
            print(f"Error loading model from {model_path}: {e}")
        return None

def export_nlp_model(output_dir=None, model_name="sentiment_model.pkl"):
    """Extract and save NLP sentiment model in pkl format"""
    if output_dir is None:
        # Get the base directory of the project
        base_dir = Path(__file__).resolve().parents[2]
        output_dir = os.path.join(base_dir, "models", "nlp")
    
    ensure_dir(output_dir)
    output_path = os.path.join(output_dir, model_name)
    
    print(f"Exporting NLP sentiment model to {output_path}...")
    try:
        # Initialize the sentiment analysis pipeline
        sentiment_model = pipeline("sentiment-analysis")
        
        # Save the model
        success = save_model_to_pkl(sentiment_model, output_path)
        if success:
            print("NLP model export successful")
        return success
    except Exception as e:
        print(f"Error exporting NLP model: {e}")
        return False

def export_facial_model(output_dir=None, model_name="facial_model.pkl"):
    """Extract and save facial emotion model configurations in pkl format"""
    if output_dir is None:
        # Get the base directory of the project
        base_dir = Path(__file__).resolve().parents[2]
        output_dir = os.path.join(base_dir, "models", "facial")
    
    ensure_dir(output_dir)
    output_path = os.path.join(output_dir, model_name)
    
    print(f"Exporting facial model configurations to {output_path}...")
    try:
        # Instead of getting DeepFace model paths directly, just save the configuration
        # since the actual model will be loaded by DeepFace when needed
        
        # Get the mediapipe face mesh
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Create a dictionary with model configurations
        facial_models = {
            "deepface_config": {
                "model_name": "Emotion",  # The model name to use with DeepFace
                "actions": ["emotion"],
                "enforce_detection": False,
                "silent": True
            },
            "mediapipe_config": {
                "max_num_faces": 1,
                "refine_landmarks": True,
                "min_detection_confidence": 0.5,
                "min_tracking_confidence": 0.5
            }
        }
        
        # Save the configurations
        success = save_model_to_pkl(facial_models, output_path)
        if success:
            print("Facial model configurations export successful")
        return success
    except Exception as e:
        print(f"Error exporting facial model configurations: {e}")
        return False

def export_audio_analysis_params(output_dir=None, model_name="audio_params.pkl"):
    """Save audio analysis parameters in pkl format"""
    if output_dir is None:
        # Get the base directory of the project
        base_dir = Path(__file__).resolve().parents[2]
        output_dir = os.path.join(base_dir, "models", "audio")
    
    ensure_dir(output_dir)
    output_path = os.path.join(output_dir, model_name)
    
    print(f"Exporting audio analysis parameters to {output_path}...")
    try:
        # Define audio analysis parameters
        audio_params = {
            "zcr_thresholds": {
                "angry": 0.05,
                "happy": 0.04,
                "sad": 0.03,
                "calm": 0.04
            },
            "energy_thresholds": {
                "angry": 0.005,
                "happy": 0.003,
                "sad": 0.002,
                "calm": 0.003
            }
        }
        
        # Save the parameters
        success = save_model_to_pkl(audio_params, output_path)
        if success:
            print("Audio analysis parameters export successful")
        return success
    except Exception as e:
        print(f"Error exporting audio analysis parameters: {e}")
        return False

def export_all_models(output_base_dir=None):
    """Export all available models to pkl format"""
    if output_base_dir is None:
        # Get the base directory of the project
        output_base_dir = Path(__file__).resolve().parents[2]
    
    nlp_dir = os.path.join(output_base_dir, "models", "nlp")
    facial_dir = os.path.join(output_base_dir, "models", "facial")
    audio_dir = os.path.join(output_base_dir, "models", "audio")
    
    print("Starting export of all models to pkl format...")
    
    nlp_result = export_nlp_model(nlp_dir)
    facial_result = export_facial_model(facial_dir)
    audio_result = export_audio_analysis_params(audio_dir)
    
    success_count = sum([nlp_result, facial_result, audio_result])
    print(f"Export completed. {success_count}/3 models successfully exported.")
    
    return {
        "nlp": nlp_result,
        "facial": facial_result,
        "audio": audio_result
    }

if __name__ == "__main__":
    # If the script is run directly, export all models
    export_all_models() 