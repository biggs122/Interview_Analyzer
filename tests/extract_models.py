#!/usr/bin/env python3
"""
Extract and save interview analyzer models in .pkl format
This script extracts all the models used in the interview analyzer app
and saves them in pickle format for faster loading and portability.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import from src
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

# Explicitly add the src directory path
src_dir = os.path.join(parent_dir, "src")
sys.path.append(str(src_dir))

# Import the utility functions
# First try the direct import
try:
    # Try importing from the module path
    from src.utils.model_utils import (
        export_nlp_model,
        export_facial_model,
        export_audio_analysis_params,
        export_all_models
    )
except ImportError:
    # If that fails, try direct import from file
    print("Package import failed, trying direct file import...")
    sys.path.append(os.path.join(parent_dir, "src", "utils"))
    from model_utils import (
        export_nlp_model,
        export_facial_model,
        export_audio_analysis_params,
        export_all_models
    )

def main():
    print("=== Interview Analyzer Model Extractor ===")
    print("This script will extract models and save them in .pkl format")
    print("Models will be saved in the 'models' directory")
    
    # Get the base directory
    base_dir = parent_dir
    models_dir = os.path.join(base_dir, "models")
    
    # Ensure the model directories exist
    os.makedirs(os.path.join(models_dir, "nlp"), exist_ok=True)
    os.makedirs(os.path.join(models_dir, "facial"), exist_ok=True)
    os.makedirs(os.path.join(models_dir, "audio"), exist_ok=True)
    
    # Prompt for extraction
    print("\nWhich models would you like to extract?")
    print("1. NLP models (sentiment analysis)")
    print("2. Facial models (emotion recognition)")
    print("3. Audio models (voice analysis parameters)")
    print("4. All models")
    
    try:
        choice = int(input("Enter your choice (1-4): "))
        
        if choice == 1:
            print("\nExtracting NLP models...")
            export_nlp_model(os.path.join(models_dir, "nlp"))
        elif choice == 2:
            print("\nExtracting facial models...")
            export_facial_model(os.path.join(models_dir, "facial"))
        elif choice == 3:
            print("\nExtracting audio models...")
            export_audio_analysis_params(os.path.join(models_dir, "audio"))
        elif choice == 4:
            print("\nExtracting all models...")
            export_all_models(base_dir)
        else:
            print("\nInvalid choice. Please select a number between 1 and 4.")
            return
        
        print("\nExtraction process completed.")
        print(f"Model files have been saved to: {models_dir}")
        print("These models can now be loaded by the application for faster startup.")
    
    except ValueError:
        print("\nInvalid input. Please enter a number.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main() 