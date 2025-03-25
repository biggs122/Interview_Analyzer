#!/usr/bin/env python3
"""
Advanced Model Loader for Interview Analyzer Pro

This module handles loading and management of the advanced AI models:
1. BERT/RoBERTa-based CV analysis
2. RoBERTa-based interview content analysis 
3. ResNet-based emotion detection
4. Whisper-integrated speech recognition

It provides intelligent fallbacks and lazy loading to optimize performance.
"""

import os
import sys
import json
import pickle
import logging
from pathlib import Path
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
VERSION_FILE = MODELS_DIR / "version_info.json"


class AdvancedModelLoader:
    """Manages loading and access to advanced AI models"""
    
    def __init__(self, use_gpu=False):
        """Initialize the model loader"""
        self.model_paths = {
            "cv_classifier": MODELS_DIR / "cv_classifier.pkl",
            "interview_analysis": MODELS_DIR / "interview_analysis.pkl",
            "emotion_detection": MODELS_DIR / "emotion_detection.pkl",
            "speech_recognition": MODELS_DIR / "speech_recognition.pkl"
        }
        
        # Keep track of loaded models
        self.loaded_models = {}
        self.use_gpu = use_gpu
        
        # Version management
        self.version_info = self._load_version_info()
        self.current_version = self.version_info.get("current_version", "v1.0")
        
        # Initialize models dictionary
        self.models = {}
        
        # Check available libraries for advanced models
        self.available_libraries = self._check_available_libraries()
        logger.info(f"Available libraries: {', '.join(self.available_libraries)}")
        
        # Log current version
        self._log_version_info()
    
    def _check_available_libraries(self):
        """Check which advanced libraries are available"""
        libraries = []
        
        try:
            import torch
            libraries.append("torch")
            if torch.cuda.is_available() and self.use_gpu:
                libraries.append("cuda")
                logger.info("GPU acceleration is available and enabled")
            else:
                logger.info("Using CPU for inference")
        except ImportError:
            logger.warning("PyTorch not available. Some models will use fallbacks.")
        
        try:
            import transformers
            libraries.append("transformers")
        except ImportError:
            logger.warning("Transformers not available. BERT/RoBERTa models will use fallbacks.")
        
        try:
            import librosa
            libraries.append("librosa")
        except ImportError:
            logger.warning("Librosa not available. Advanced audio processing will be limited.")
        
        try:
            import whisper
            libraries.append("whisper")
        except ImportError:
            logger.warning("OpenAI Whisper not available. Will use basic speech recognition.")
        
        return libraries
    
    def _load_version_info(self):
        """Load the version info from the version file"""
        if os.path.exists(VERSION_FILE):
            try:
                with open(VERSION_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading version info: {e}")
        
        logger.warning("Version info file not found. Using default version.")
        return {"current_version": "v1.0"}
    
    def _log_version_info(self):
        """Log information about the current model version"""
        if self.version_info and "available_versions" in self.version_info:
            version_data = self.version_info["available_versions"].get(self.current_version, {})
            metrics = self.version_info.get("performance_metrics", {}).get(self.current_version, {})
            
            logger.info(f"Using model version: {self.current_version}")
            if version_data:
                logger.info(f"Version description: {version_data.get('description', 'N/A')}")
            
            if metrics:
                logger.info(f"Model performance metrics:")
                for key, value in metrics.items():
                    logger.info(f"  - {key}: {value:.4f}")
    
    def _load_model(self, model_name):
        """Load a model from disk with error handling and fallbacks"""
        if model_name not in self.model_paths:
            logger.error(f"Unknown model: {model_name}")
            return None
        
        model_path = self.model_paths[model_name]
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return self._create_mock_model(model_name)
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Check if this model requires special libraries
            if model_data.get("needs_transformers", False) and "transformers" not in self.available_libraries:
                logger.warning(f"{model_name} requires transformers library. Using fallback.")
                return self._create_mock_model(model_name)
            
            if model_data.get("needs_pytorch", False) and "torch" not in self.available_libraries:
                logger.warning(f"{model_name} requires PyTorch. Using fallback.")
                return self._create_mock_model(model_name)
            
            if model_data.get("needs_whisper", False) and "whisper" not in self.available_libraries:
                logger.warning(f"{model_name} requires Whisper. Using limited functionality.")
            
            # Special handling for PyTorch models (emotion detection)
            if model_name == "emotion_detection" and model_data.get("torch_model_path"):
                model_data = self._load_pytorch_model(model_data)
            
            logger.info(f"Successfully loaded {model_name} ({model_data.get('type', 'unknown type')})")
            return model_data
            
        except Exception as e:
            logger.error(f"Error loading {model_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self._create_mock_model(model_name)
    
    def _load_pytorch_model(self, model_data):
        """Load a PyTorch model"""
        try:
            import torch
            import torch.nn as nn
            from torchvision.models import resnet18, ResNet18_Weights
            
            # Path to the saved model weights
            torch_model_path = model_data.get("torch_model_path")
            if not os.path.exists(torch_model_path):
                logger.error(f"PyTorch model weights not found: {torch_model_path}")
                return model_data  # Return without the actual model
            
            # Define the model architecture
            if model_data.get("base_model") == "resnet18":
                # Create a ResNet model with the right number of classes
                num_emotions = len(model_data.get("emotions", []))
                model = resnet18(weights=None)  # Don't load pretrained weights
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs, num_emotions)
                
                # Load the trained weights
                device = torch.device('cuda' if self.use_gpu and torch.cuda.is_available() else 'cpu')
                model.load_state_dict(torch.load(torch_model_path, map_location=device))
                model.to(device)
                model.eval()  # Set to evaluation mode
                
                # Add the loaded PyTorch model to the model data
                model_data["torch_model"] = model
                model_data["device"] = device
                
                logger.info(f"Successfully loaded PyTorch model from {torch_model_path}")
                
            return model_data
            
        except Exception as e:
            logger.error(f"Error loading PyTorch model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return model_data  # Return without the actual model
    
    def _create_mock_model(self, model_name):
        """Create a mock model when the real one can't be loaded"""
        logger.warning(f"Creating mock implementation for {model_name}")
        
        if model_name == "cv_classifier":
            return {
                "type": "MockCVClassifier",
                "version": "mock",
                "classes": ["technical", "management", "creative", "customer_service", "research"],
                "predict": lambda x: np.random.choice(["technical", "management", "creative", "customer_service", "research"])
            }
            
        elif model_name == "interview_analysis":
            return {
                "type": "MockInterviewAnalyzer",
                "version": "mock",
                "metrics": ["quality", "relevance", "depth", "clarity", "structure"],
                "predict": lambda x: {
                    "quality": np.random.uniform(0.5, 1.0),
                    "relevance": np.random.uniform(0.5, 1.0),
                    "depth": np.random.uniform(0.5, 1.0),
                    "clarity": np.random.uniform(0.5, 1.0),
                    "structure": np.random.uniform(0.5, 1.0)
                }
            }
            
        elif model_name == "emotion_detection":
            emotions = ['neutral', 'happy', 'sad', 'angry', 'surprised', 'scared', 'disgusted']
            return {
                "type": "MockEmotionDetector",
                "version": "mock",
                "emotions": emotions,
                "predict": lambda x: emotions[np.random.randint(0, len(emotions))]
            }
            
        elif model_name == "speech_recognition":
            return {
                "type": "MockSpeechRecognizer",
                "version": "mock",
                "quality_classes": [
                    "poor_articulation",
                    "average_clarity",
                    "good_clarity",
                    "excellent_clarity",
                    "professional_quality"
                ],
                "transcribe": lambda audio: "This is a mock transcription. The actual model couldn't be loaded.",
                "predict_quality": lambda audio: np.random.randint(0, 5)
            }
            
        return None
    
    def get_model(self, model_name):
        """Get a model, loading it if necessary"""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        model = self._load_model(model_name)
        self.loaded_models[model_name] = model
        return model
    
    def predict_cv_category(self, cv_text):
        """Predict the category of a CV"""
        model_data = self.get_model("cv_classifier")
        if not model_data:
            return "unknown"
        
        if model_data.get("type") == "MockCVClassifier":
            return model_data["predict"](cv_text)
        
        try:
            # BERT-based model requires transformers
            if "transformers" in self.available_libraries and model_data.get("vectorizer_type") == "bert-base-uncased":
                from transformers import AutoTokenizer, AutoModel
                import torch
                
                # Initialize BERT components
                tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
                bert_model = AutoModel.from_pretrained('bert-base-uncased')
                
                # Move to device
                device = torch.device('cuda' if self.use_gpu and torch.cuda.is_available() else 'cpu')
                bert_model.to(device)
                bert_model.eval()
                
                # Process text
                encoded_input = tokenizer(
                    cv_text, 
                    padding=True, 
                    truncation=True, 
                    max_length=512, 
                    return_tensors='pt'
                ).to(device)
                
                # Get embeddings
                with torch.no_grad():
                    output = bert_model(**encoded_input)
                
                # Get sentence embedding from CLS token
                features = output.last_hidden_state[:, 0, :].cpu().numpy()
                
                # Use the classifier on the BERT features
                prediction = model_data["model"].predict(features)[0]
                return prediction
                
            elif "vectorizer" in model_data:
                # TF-IDF based model
                features = model_data["vectorizer"].transform([cv_text])
                prediction = model_data["model"].predict(features)[0]
                return prediction
            
            else:
                logger.error(f"Unsupported CV classifier type: {model_data.get('type')}")
                return "unknown"
                
        except Exception as e:
            logger.error(f"Error in CV classification: {e}")
            return "unknown"
    
    def analyze_interview_content(self, response_text):
        """Analyze the content of an interview response"""
        model_data = self.get_model("interview_analysis")
        if not model_data:
            return {
                "quality": 0.5,
                "relevance": 0.5,
                "depth": 0.5,
                "clarity": 0.5,
                "structure": 0.5
            }
        
        if model_data.get("type") == "MockInterviewAnalyzer":
            return model_data["predict"](response_text)
        
        try:
            # RoBERTa-based model
            if "transformers" in self.available_libraries and model_data.get("base_model") == "roberta-base":
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                import torch
                
                # Initialize RoBERTa
                tokenizer = AutoTokenizer.from_pretrained("roberta-base")
                roberta_model = AutoModelForSequenceClassification.from_pretrained(
                    "roberta-base", 
                    num_labels=2
                )
                
                # Move to device
                device = torch.device('cuda' if self.use_gpu and torch.cuda.is_available() else 'cpu')
                roberta_model.to(device)
                roberta_model.eval()
                
                # Process text
                encoded_data = tokenizer(
                    response_text,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                )
                
                # Get features
                with torch.no_grad():
                    input_ids = encoded_data['input_ids'].to(device)
                    attention_mask = encoded_data['attention_mask'].to(device)
                    outputs = roberta_model(input_ids, attention_mask=attention_mask)
                    features = outputs.logits.cpu().numpy()
                
                # Use the gradient boosting classifier
                quality_score = model_data["model"].predict_proba(features)[0][1]  # Probability of class 1 (good)
                
                # Generate synthetic scores for other metrics based on quality
                # In a real implementation, you'd have separate models for each metric
                noise = np.random.normal(0, 0.1, 4)  # Small random variations
                
                return {
                    "quality": float(quality_score),
                    "relevance": float(min(1.0, max(0.0, quality_score + noise[0]))),
                    "depth": float(min(1.0, max(0.0, quality_score * 0.9 + noise[1]))),
                    "clarity": float(min(1.0, max(0.0, quality_score * 1.1 + noise[2]))),
                    "structure": float(min(1.0, max(0.0, quality_score + noise[3])))
                }
                
            elif "vectorizer" in model_data:
                # TF-IDF based model
                features = model_data["vectorizer"].transform([response_text])
                # This might return a single value instead of the detailed metrics
                # We'll convert it to a more detailed format
                score = float(model_data["model"].predict(features)[0]) / 4.0  # Assuming scores are 0-4
                
                return {
                    "quality": score,
                    "relevance": score * 1.1,
                    "depth": score * 0.9,
                    "clarity": score * 1.0,
                    "structure": score * 0.95
                }
            
            else:
                logger.error(f"Unsupported interview analyzer type: {model_data.get('type')}")
                return {
                    "quality": 0.5,
                    "relevance": 0.5,
                    "depth": 0.5,
                    "clarity": 0.5,
                    "structure": 0.5
                }
                
        except Exception as e:
            logger.error(f"Error in interview content analysis: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "quality": 0.5,
                "relevance": 0.5,
                "depth": 0.5,
                "clarity": 0.5,
                "structure": 0.5
            }
    
    def detect_emotion(self, image):
        """Detect emotion in a facial image"""
        model_data = self.get_model("emotion_detection")
        if not model_data:
            return "neutral"
        
        if model_data.get("type") == "MockEmotionDetector":
            return model_data["predict"](image)
        
        try:
            # ResNet-based model
            if "torch" in self.available_libraries and "torch_model" in model_data:
                import torch
                import cv2
                import torchvision.transforms as transforms
                
                # Get model and device
                model = model_data["torch_model"]
                device = model_data.get("device", torch.device("cpu"))
                
                # Preprocess the image
                transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=model_data.get("transform_mean", [0.485, 0.456, 0.406]),
                        std=model_data.get("transform_std", [0.229, 0.224, 0.225])
                    ),
                ])
                
                # Convert to RGB if grayscale
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif image.shape[2] == 4:  # With alpha channel
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
                elif image.shape[2] == 3:  # BGR format
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Apply transformations
                img_tensor = transform(image).unsqueeze(0).to(device)
                
                # Make prediction
                with torch.no_grad():
                    outputs = model(img_tensor)
                    _, predicted = torch.max(outputs, 1)
                    emotion_idx = predicted.item()
                
                # Get emotion label
                emotions = model_data.get("emotions", 
                    ['neutral', 'happy', 'sad', 'angry', 'surprised', 'scared', 'disgusted']
                )
                
                if 0 <= emotion_idx < len(emotions):
                    return emotions[emotion_idx]
                else:
                    return "neutral"
                    
            elif model_data.get("type") == "LogisticRegression" and "scaler" in model_data:
                # Traditional ML model with scikit-learn
                # Assuming image is already grayscale and proper size
                features = image.flatten()
                features = model_data["scaler"].transform([features])
                emotion_idx = model_data["model"].predict(features)[0]
                
                emotions = model_data.get("emotions", 
                    ['neutral', 'happy', 'sad', 'angry', 'surprised', 'scared', 'disgusted']
                )
                
                if isinstance(emotion_idx, str):
                    return emotion_idx
                elif 0 <= emotion_idx < len(emotions):
                    return emotions[emotion_idx]
                else:
                    return "neutral"
            
            else:
                logger.error(f"Unsupported emotion detector type: {model_data.get('type')}")
                return "neutral"
                
        except Exception as e:
            logger.error(f"Error in emotion detection: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return "neutral"
    
    def transcribe_speech(self, audio_file):
        """Transcribe speech from an audio file"""
        model_data = self.get_model("speech_recognition")
        if not model_data:
            return "Could not transcribe speech. Model not available."
        
        if model_data.get("type") == "MockSpeechRecognizer":
            return model_data["transcribe"](audio_file)
        
        try:
            # Whisper-based transcription
            if "whisper" in self.available_libraries and model_data.get("whisper_integration", False):
                import whisper
                
                # Load the Whisper model
                whisper_model_size = model_data.get("whisper_model", "base")
                whisper_model = whisper.load_model(whisper_model_size)
                
                # Transcribe
                result = whisper_model.transcribe(audio_file)
                return result["text"]
            
            # Fallback to basic model
            logger.warning("Using basic speech recognition (simulated)")
            return "This is a simulated transcription. Install Whisper for better results."
                
        except Exception as e:
            logger.error(f"Error in speech transcription: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return "Error transcribing speech. See logs for details."
    
    def assess_speech_quality(self, audio_features):
        """Assess the quality of speech from audio features"""
        model_data = self.get_model("speech_recognition")
        if not model_data:
            return {
                "clarity": 0.5,
                "confidence": 0.5,
                "pace": 0.5,
                "filler_ratio": 0.5
            }
        
        if model_data.get("type") == "MockSpeechRecognizer":
            quality_idx = model_data["predict_quality"](audio_features)
            quality = quality_idx / 4.0  # Normalize to 0-1
            
            return {
                "clarity": quality,
                "confidence": quality + np.random.normal(0, 0.1),
                "pace": 0.5,  # Middle pace is often best
                "filler_ratio": 1.0 - quality + np.random.normal(0, 0.1)
            }
        
        try:
            # For the advanced model, we'd extract MFCC features
            # Here we assume audio_features are already MFCC features
            
            # Predict quality class
            quality_idx = model_data["model"].predict([audio_features])[0]
            quality = quality_idx / 4.0  # Normalize to 0-1
            
            # In a real implementation, we'd have separate models for each aspect
            # Here we're simulating based on the overall quality
            return {
                "clarity": quality,
                "confidence": quality + np.random.normal(0, 0.1),
                "pace": 0.5,  # Middle pace is often best
                "filler_ratio": 1.0 - quality + np.random.normal(0, 0.1)
            }
                
        except Exception as e:
            logger.error(f"Error in speech quality assessment: {e}")
            return {
                "clarity": 0.5,
                "confidence": 0.5,
                "pace": 0.5,
                "filler_ratio": 0.5
            }
            
    def get_performance_metrics(self):
        """Get the performance metrics for the current version"""
        if self.version_info and "performance_metrics" in self.version_info:
            return self.version_info["performance_metrics"].get(self.current_version, {})
        return {}
    
    def get_version_info(self):
        """Get information about available versions"""
        return self.version_info
    
    def switch_version(self, version):
        """Switch to a different model version"""
        if version not in self.version_info.get("available_versions", {}):
            logger.error(f"Version {version} not available")
            return False
        
        # Update version info
        self.current_version = version
        self.version_info["current_version"] = version
        
        # Save changes
        try:
            with open(VERSION_FILE, 'w') as f:
                json.dump(self.version_info, f, indent=4)
            logger.info(f"Switched to version {version}")
            
            # Clear loaded models to force reload
            self.loaded_models = {}
            
            # Log the new version info
            self._log_version_info()
            
            return True
        except Exception as e:
            logger.error(f"Error saving version info: {e}")
            return False


# Usage example
if __name__ == "__main__":
    model_loader = AdvancedModelLoader(use_gpu=False)
    
    # Print available versions
    version_info = model_loader.get_version_info()
    print("Available versions:")
    for version, details in version_info.get("available_versions", {}).items():
        print(f"- {version}: {details.get('description', 'No description')}")
    
    # Print performance metrics
    metrics = model_loader.get_performance_metrics()
    print("\nPerformance metrics:")
    for key, value in metrics.items():
        print(f"- {key}: {value:.4f}")
    
    # Test CV classifier
    sample_cv = """
    John Doe
    Software Engineer
    
    SKILLS:
    Programming: Python, JavaScript, C++
    Frameworks: Django, React, TensorFlow
    
    EXPERIENCE:
    Google - Software Engineer (2018-2022)
    Developed machine learning models for improving search results
    
    EDUCATION:
    Stanford University - Computer Science (2014-2018)
    """
    
    print(f"\nCV Classification: {model_loader.predict_cv_category(sample_cv)}")
    
    # Test interview analyzer
    sample_answer = """
    I have 5 years of experience as a software engineer, focusing on machine learning applications.
    At Google, I worked on improving search algorithms, which increased relevance by 15%.
    I'm particularly skilled in Python and TensorFlow for building ML models.
    """
    
    analysis = model_loader.analyze_interview_content(sample_answer)
    print("\nInterview Content Analysis:")
    for metric, score in analysis.items():
        print(f"- {metric}: {score:.2f}") 