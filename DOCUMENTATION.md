# Interview Analyzer Pro - Complete Documentation

## Overview

Interview Analyzer Pro is a comprehensive application that provides CV analysis and interview assessment capabilities with real-time emotion, voice, and text sentiment analysis. This document provides a complete overview of the application, its architecture, and usage guidelines.

## Documentation Index

1. [User Guide](#user-guide)
2. [Installation](#installation)
3. [Technical Architecture](#technical-architecture)
4. [Development](#development)
5. [API Reference](#api-reference)
6. [Troubleshooting](#troubleshooting)

## User Guide

### Getting Started

Interview Analyzer Pro combines CV analysis and interview assessment in a single application:

1. **Start the application**: Launch using the methods described in the [Installation](#installation) section
2. **Choose a functionality**: The application has two main tabs - CV Analysis and Interview Analysis

### CV Analysis

The CV Analysis functionality allows you to extract information from CVs in various formats:

1. **Select a CV**: Use the file browser to select a CV file (PDF, DOCX, or TXT)
2. **Analyze**: Click "ANALYZE CV" to process the document
3. **Review Results**: Examine the extracted information including:
   - Personal details (name, contact information)
   - Education and work experience
   - Skills and proficiencies
   - Career match recommendations
4. **Save or Continue**: Save the analysis report or proceed directly to an interview

### Interview Analysis

The Interview Analysis functionality provides real-time assessment during interviews:

1. **Start Interview**: Click the "START INTERVIEW" button to begin recording
2. **Conduct Interview**: The system will analyze in real-time:
   - Facial expressions and emotions
   - Voice tone and emotions
   - Speech content and sentiment
3. **Problem-Solving Assessment**: Use the built-in coding challenge to assess technical skills
4. **Review Results**: After completing the interview, review the comprehensive analysis
5. **Save Report**: Save the full interview assessment report

## Installation

For detailed installation instructions, see [INSTALL.md](INSTALL.md).

### Quick Installation

```bash
# Option 1: Using Conda (recommended)
bash setup_conda.sh
conda activate interview_analyzer
python tests/kivy_app_with_real_models.py

# Option 2: Using Pip
python run_app.py
```

## Technical Architecture

The application is built with a modular architecture that separates the UI, analysis engine, and data processing components.

### Key Components

1. **UI Layer**: Built with Kivy framework
   - Main application: `tests/kivy_app_with_real_models.py`
   - Handles user interaction and result visualization

2. **Analysis Engine**: Core functionality
   - Main class: `RealInterviewAnalyzer` in `tests/real_interview_analyzer.py`
   - Manages all analysis operations across CV and interview processing

3. **Data Processing**: Specialized modules
   - CV parsing and extraction: `src/preprocessing/`
   - Model inference: `src/inference/`
   - Utility functions: `src/utils/`

### Machine Learning Models

The application uses several ML models for different analysis tasks:

1. **CV Analysis**:
   - spaCy for named entity recognition and information extraction
   - Custom classifiers for skills identification and career matching

2. **Facial Analysis**:
   - MediaPipe for facial landmark detection
   - Custom emotion classification model

3. **Voice Analysis**:
   - SpeechRecognition for transcription
   - Librosa for audio feature extraction
   - Custom emotion classification model

4. **Text Analysis**:
   - VADER Sentiment for sentiment scoring
   - Transformers for advanced text understanding

### Data Flow

1. **CV Analysis Pipeline**:
   - Document parsing → Text extraction → Entity recognition → Skills extraction → Career matching

2. **Interview Analysis Pipeline**:
   - Video capture → Facial emotion detection
   - Audio capture → Speech recognition → Voice emotion analysis
   - Text analysis → Sentiment scoring
   - Combined metrics → Comprehensive assessment

## Development

For detailed development documentation, see [DEVELOPER.md](DEVELOPER.md).

### Project Structure

```
interview_analyzer/
├── app/                      # Application-specific code
├── data/                     # Sample and test data
├── models/                   # Machine learning model files
├── results/                  # Generated analysis results
├── src/                      # Core source code
│   ├── preprocessing/        # Data preprocessing modules
│   ├── inference/            # Model inference logic
│   ├── fine_tuning/          # Model fine-tuning scripts
│   └── utils/                # Utility functions
├── tests/                    # Tests and application entry points
│   ├── kivy_app_with_real_models.py  # Main application
│   ├── real_interview_analyzer.py    # Interview analysis backend
│   └── ...                   # Various test modules
├── environment.yml           # Conda environment specification
├── requirements.txt          # Pip requirements
├── run_app.py                # Application entry script
└── setup_conda.sh            # Conda setup script
```

## API Reference

### RealInterviewAnalyzer Class

The main class that handles all analysis operations:

```python
from tests.real_interview_analyzer import RealInterviewAnalyzer

# Initialize the analyzer
analyzer = RealInterviewAnalyzer()

# CV analysis
cv_results = analyzer.analyze_cv("/path/to/cv.pdf")

# Start interview analysis
analyzer.start_interview()

# Get facial emotions
facial_emotions = analyzer.get_facial_emotions()

# Get voice emotions
voice_emotions = analyzer.get_voice_emotions()

# Get transcription
transcript = analyzer.get_transcript()

# Stop interview analysis
analyzer.stop_interview()

# Generate report
full_report = analyzer.generate_full_report()
```

## Troubleshooting

### Common Issues

1. **Application crashes during startup**:
   - Ensure all dependencies are correctly installed
   - Check if your Python version is 3.8 or higher
   - On Windows, you may need to install Visual C++ Build Tools

2. **Camera or microphone not working**:
   - Ensure your device has permission to access camera/microphone
   - Check if another application is using these resources
   - Try disconnecting and reconnecting the devices

3. **CV analysis not accurate**:
   - Try converting your CV to a different format (PDF often works best)
   - Make sure your CV follows standard formatting

### Getting Help

If you encounter issues not covered in this documentation:
- Check the issue tracker on GitHub
- Open a new issue with detailed information about your problem 