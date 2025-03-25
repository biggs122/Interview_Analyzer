#!/bin/bash
# Setup script for Interview Analyzer Pro using Conda

echo "===================================================="
echo "Interview Analyzer Pro - Conda Setup"
echo "===================================================="

# Check if conda is installed
if ! command -v conda &> /dev/null
then
    echo "Conda is not installed. Please install Anaconda or Miniconda first."
    exit 1
fi

# Create or update the environment
if conda info --envs | grep -q "interview_analyzer"
then
    echo "Updating existing 'interview_analyzer' environment..."
    conda env update -n interview_analyzer --file environment.yml
else
    echo "Creating new 'interview_analyzer' environment..."
    conda env create -f environment.yml
fi

# Activate the environment
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate interview_analyzer

# Install spaCy models
echo "Installing spaCy models..."
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
python -m spacy download fr_core_news_md

# Install NLTK data
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('maxent_ne_chunker'); nltk.download('words')"

echo "Installation complete!"
echo ""
echo "To run the application:"
echo "1. Activate the environment: conda activate interview_analyzer"
echo "2. Run the application: python tests/kivy_app_with_real_models.py" 