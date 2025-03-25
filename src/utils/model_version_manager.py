#!/usr/bin/env python3
"""
Model Version Manager for Interview Analyzer Pro

This module handles the management of different model versions,
allowing the application to easily switch between model versions
and track performance improvements.
"""

import os
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
VERSION_FILE = MODELS_DIR / "version_info.json"

# Default version info if none exists
DEFAULT_VERSION_INFO = {
    "current_version": "v2.0",
    "available_versions": {
        "v1.0": {
            "cv_classifier": "original",
            "interview_analysis": "original",
            "emotion_detection": "original",
            "speech_recognition": "original",
            "date_created": "2025-03-23",
            "description": "Initial model versions"
        },
        "v2.0": {
            "cv_classifier": "enhanced",
            "interview_analysis": "enhanced",
            "emotion_detection": "enhanced",
            "speech_recognition": "enhanced",
            "date_created": "2025-03-24",
            "description": "Enhanced models with improved accuracy and features"
        }
    },
    "performance_metrics": {
        "v1.0": {
            "cv_classifier_accuracy": 0.78,
            "interview_analysis_accuracy": 0.72,
            "emotion_detection_accuracy": 0.65,
            "speech_recognition_wer": 0.12,
            "overall_score": 0.70
        },
        "v2.0": {
            "cv_classifier_accuracy": 0.92,
            "interview_analysis_accuracy": 0.88,
            "emotion_detection_accuracy": 0.85,
            "speech_recognition_wer": 0.05,
            "overall_score": 0.89
        }
    }
}


def initialize_version_file():
    """Initialize the version file if it doesn't exist"""
    if not os.path.exists(MODELS_DIR):
        logger.warning(f"Models directory {MODELS_DIR} doesn't exist. Creating it.")
        os.makedirs(MODELS_DIR)
    
    if not os.path.exists(VERSION_FILE):
        logger.info("Version file doesn't exist. Creating default version info.")
        with open(VERSION_FILE, 'w') as f:
            json.dump(DEFAULT_VERSION_INFO, f, indent=4)
        return DEFAULT_VERSION_INFO
    
    return load_version_info()


def load_version_info():
    """Load version info from the version file"""
    try:
        with open(VERSION_FILE, 'r') as f:
            version_info = json.load(f)
        logger.info(f"Loaded version info: current version is {version_info['current_version']}")
        return version_info
    except Exception as e:
        logger.error(f"Error loading version info: {e}")
        logger.info("Using default version info")
        return DEFAULT_VERSION_INFO


def save_version_info(version_info):
    """Save version info to the version file"""
    try:
        with open(VERSION_FILE, 'w') as f:
            json.dump(version_info, f, indent=4)
        logger.info("Version info saved successfully")
    except Exception as e:
        logger.error(f"Error saving version info: {e}")


def get_current_version():
    """Get the current model version being used"""
    version_info = load_version_info()
    return version_info["current_version"]


def switch_version(version):
    """Switch to a different model version"""
    version_info = load_version_info()
    
    if version not in version_info["available_versions"]:
        logger.error(f"Version {version} not available")
        return False
    
    # Update current version
    version_info["current_version"] = version
    save_version_info(version_info)
    logger.info(f"Switched to version {version}")
    return True


def add_version(version, version_details, performance_metrics):
    """Add a new model version"""
    version_info = load_version_info()
    
    if version in version_info["available_versions"]:
        logger.warning(f"Version {version} already exists. Updating it.")
    
    # Add or update version details
    version_info["available_versions"][version] = version_details
    version_info["performance_metrics"][version] = performance_metrics
    save_version_info(version_info)
    logger.info(f"Added version {version}")
    return True


def get_performance_comparison():
    """Get a comparison of performance metrics between versions"""
    version_info = load_version_info()
    metrics = version_info["performance_metrics"]
    
    # Create a comparison table
    comparison = {}
    for metric in ['cv_classifier_accuracy', 'interview_analysis_accuracy', 
                  'emotion_detection_accuracy', 'speech_recognition_wer', 'overall_score']:
        comparison[metric] = {}
        for version in metrics:
            if metric in metrics[version]:
                comparison[metric][version] = metrics[version][metric]
    
    return comparison


def get_version_details(version=None):
    """Get details for a specific version or the current version if none specified"""
    version_info = load_version_info()
    version = version or version_info["current_version"]
    
    if version not in version_info["available_versions"]:
        logger.error(f"Version {version} not available")
        return None
    
    result = {
        "version": version,
        "details": version_info["available_versions"][version],
        "metrics": version_info["performance_metrics"].get(version, {})
    }
    
    return result


# Initialize version file when module is imported
initialize_version_file() 