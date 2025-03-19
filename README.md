# Interview Analyzer

A comprehensive tool for analyzing interviews using facial expression recognition, speech emotion detection, and text sentiment analysis.

## Overview

The Interview Analyzer is an advanced application that processes multi-modal inputs (video, audio, and transcription) to provide real-time feedback on a candidate's emotional state during interviews. It combines computer vision, audio processing, and natural language processing techniques to deliver a holistic view of interview performance.

## Features

- **Facial Expression Analysis**: Recognizes emotions from facial expressions using DeepFace and MediaPipe
- **Voice Emotion Detection**: Analyzes audio features to detect emotional states in speech
- **Text Sentiment Analysis**: Processes transcribed speech to determine sentiment using DistilBERT
- **Real-time Processing**: Provides immediate feedback as the interview progresses
- **Problem-Solving Assessment**: Includes a coding challenge module to evaluate technical abilities
- **Report Generation**: Creates comprehensive reports of the interview analysis
- **Model Serialization**: Saves models in .pkl format for faster loading and portability

## Project Structure

```
interview_analyzer/
├── data/                  # Data directory (not included in repository)
│   ├── facial/            # Facial expression data
│   ├── audio/             # Audio samples
│   ├── cv/                # CV/resume data
│   └── processed/         # Processed and combined data
├── models/                # Trained models and configurations
│   ├── facial/            # Facial emotion models
│   ├── audio/             # Audio analysis parameters
│   └── nlp/               # NLP/sentiment models
├── src/                   # Source code
│   ├── preprocessing/     # Data preprocessing modules
│   ├── inference/         # Model inference code
│   ├── fine_tuning/       # Code for fine-tuning models
│   └── utils/             # Utility functions
├── tests/                 # Test scripts
│   ├── advanced_interview_analyzer.py       # Main analyzer class
│   ├── enhanced_interview_analyzer.py       # Enhanced UI version
│   ├── extract_models.py                    # Model extraction utility
│   └── test_*.py                            # Various test scripts
├── results/               # Analysis results (not included in repository)
└── temp/                  # Temporary files (not included in repository)
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/interview_analyzer.git
cd interview_analyzer
```

2. Create a virtual environment and install dependencies:
```bash
conda create -n interview_analyzer python=3.8
conda activate interview_analyzer
pip install -r requirements.txt
```

3. Download necessary models:
```bash
python tests/extract_models.py
```

## Usage

Run the enhanced interview analyzer:

```bash
python tests/enhanced_interview_analyzer.py
```

## Data Directory (not included)

To use your own data, create the following structure in the `data` directory:
- `facial/`: Contains facial expression datasets
- `audio/`: Contains speech samples
- `cv/`: Contains CV documents
- `processed/`: For outputs from preprocessing scripts

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
