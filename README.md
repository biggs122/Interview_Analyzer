# Interview Analyzer Pro

A professional application for CV analysis and automated interview assessment with real-time feedback using artificial intelligence.

## Overview

Interview Analyzer Pro is a comprehensive tool designed to help job seekers prepare for interviews by leveraging AI and machine learning to:

1. Analyze CVs/resumes to identify strengths, weaknesses, and career matches
2. Conduct simulated interview sessions with real-time performance feedback
3. Analyze facial expressions, voice tone, and answer content during interviews
4. Generate actionable recommendations for improvement

By practicing with Interview Analyzer Pro, users can gain confidence, identify areas for improvement, and develop better interview skills before facing real interviewers.

## Features

- **CV Analysis**: 
  - Parse CVs in various formats (PDF, DOCX, TXT)
  - Extract and categorize skills (technical vs. soft)
  - Match candidate profiles to potential career paths
  - Generate tailored recommendations

- **Interview Simulation**: 
  - Automated question generation based on CV content
  - Multiple interview modes (Automatic, Manual, Practice)
  - Real-time video and audio capture
  - Countdown timers and response timing

- **Performance Analysis**:
  - Facial expression and emotion analysis
  - Voice tone and confidence detection
  - Answer relevance and clarity assessment
  - Comprehensive scoring system

- **Feedback System**:
  - Real-time feedback during interviews
  - Question-by-question breakdown
  - Overall performance assessment
  - Personalized improvement tips

## Project Structure

```
interview_analyzer/
├── run_interview_analyzer.py      # Main application entry point
├── models/                        # Machine learning models directory
│   ├── cv_classifier.pkl          # Model for CV classification and analysis
│   ├── interview_analysis.pkl     # Model for content analysis of interview responses
│   ├── emotion_detection.pkl      # Model for facial emotion detection
│   ├── speech_recognition.pkl     # Model for speech-to-text and voice analysis
│   └── pretrained/                # Pre-trained external models
│       └── models--facebook--wav2vec2-base/  # Facebook's wav2vec2 model for speech recognition
├── results/                       # Directory for saving analysis results
├── temp/                          # Directory for temporary files during processing
└── src/                           # Source code
    ├── pyqt_interview_analyzer.py # Main application window and controller
    ├── pyqt_tabs/                 # UI tab components
    │   ├── __init__.py
    │   ├── cv_analysis_tab.py     # CV Upload and analysis UI
    │   ├── interview_tab.py       # Interview UI with video/audio components
    │   └── mock_backend.py        # Mock implementation for testing without models
    └── utils/                     # Utility modules
        ├── __init__.py
        ├── model_loader.py        # Model loading and management
        ├── analyzer_backend.py    # Core analysis engine using models
        └── generate_mock_models.py # Creates mock models for testing
```

## Models Used

The application uses four primary machine learning models:

### 1. CV Classifier Model (`cv_classifier.pkl`)
- **Purpose**: Analyzes CV/resume content to extract information, classify skills, and identify career matches
- **Type**: Multi-class classifier (Logistic Regression)
- **Input**: Text features extracted from CV documents
- **Output**: 
  - Candidate information (name, email, phone)
  - Skills classification (technical vs. soft)
  - Career match percentages
  - Recommendations

### 2. Interview Analysis Model (`interview_analysis.pkl`)
- **Purpose**: Evaluates interview answer content for relevance, clarity, and quality
- **Type**: Random Forest Classifier
- **Input**: Text features from transcribed interview responses
- **Output**:
  - Relevance score (how well the answer addresses the question)
  - Clarity score (how clearly the response is structured)
  - Content quality score
  - Improvement recommendations

### 3. Emotion Detection Model (`emotion_detection.pkl`)
- **Purpose**: Analyzes facial expressions during interviews to detect emotions and confidence
- **Type**: Logistic Regression classifier trained on facial landmarks
- **Input**: Video frames from webcam during interview
- **Output**:
  - Emotion classification (confidence, nervousness, etc.)
  - Eye contact assessment
  - Posture evaluation
  - Non-verbal communication feedback

### 4. Speech Recognition Model (`speech_recognition.pkl`)
- **Purpose**: Transcribes and analyzes speech during interviews
- **Type**: Random Forest Classifier with integration to Facebook's wav2vec2
- **Input**: Audio recordings during interview responses
- **Output**:
  - Transcribed text of responses
  - Speech confidence metrics
  - Voice tone analysis
  - Filler word detection

### External Pre-trained Models

The system also leverages Facebook's wav2vec2 model for improved speech recognition:

- **Facebook wav2vec2**: A powerful pre-trained model for converting speech to text with high accuracy
- **Location**: `models/pretrained/models--facebook--wav2vec2-base/`
- **Integration**: Used to enhance the speech recognition capabilities of the application

## Model Management

The `ModelLoader` class in `src/utils/model_loader.py` handles model loading and management:

- Maintains paths to all model files
- Provides lazy loading to optimize memory usage
- Handles fallback to mock implementations when models aren't available
- Provides utilities to check which models are available

## Requirements

- **Python**: 3.9 or higher
- **Core Libraries**:
  - PyQt5: UI framework
  - OpenCV: Video capture and processing
  - NumPy: Numerical operations
  - scikit-learn: ML model support
- **Optional Libraries**:
  - PyPDF2: PDF parsing capabilities
  - python-docx: DOCX document processing
  - transformers: For advanced NLP capabilities

## Installation

1. **Clone the repository**:
   ```
   git clone https://github.com/yourusername/interview_analyzer.git
   cd interview_analyzer
   ```

2. **Create and activate virtual environment**:
   ```
   python -m venv interview_analyzer_env
   source interview_analyzer_env/bin/activate  # On Windows: interview_analyzer_env\Scripts\activate
   ```

3. **Install dependencies**:
   ```
   pip install PyQt5 opencv-python numpy scikit-learn
   ```

4. **Optional: Install document parsing libraries**:
   ```
   pip install PyPDF2 python-docx
   ```

5. **Generate mock models** (if you don't have trained models):
   ```
   python src/utils/generate_mock_models.py
   ```

## Usage

### Running the Application

```
python run_interview_analyzer.py
```

### CV Analysis Workflow

1. Launch the application
2. Upload your CV using the "Upload CV" button on the CV Analysis tab
3. Click "ANALYZE CV" to process your document
4. Review the analysis results:
   - Personal information extraction
   - Technical and soft skills identified
   - Career matches with match percentages
   - Recommended skills to develop
5. Click "PROCEED TO INTERVIEW" when ready

### Interview Practice Workflow

1. Select your preferred interview mode:
   - **Automatic**: AI-guided interview with automated questions
   - **Manual**: You control the questions
   - **Practice**: For basic interview practice
2. Click the green "START INTERVIEW" button
3. Prepare for the first question (3-second countdown)
4. Answer each question naturally using your webcam and microphone
5. Review real-time feedback during your responses
6. After answering all questions, review your detailed performance results
7. Save your results for future reference and improvement

## Customizing Models

You can replace the mock models with your own trained models:

1. Train your custom models using relevant datasets
2. Save your models in pickle format
3. Place them in the `models/` directory with the same names as the default models
4. The ModelLoader will automatically use your custom models

## Troubleshooting

- **Video not displaying**: Ensure your webcam is properly connected and not being used by another application
- **Audio issues**: Check your microphone settings and permissions
- **Model loading failures**: Ensure models exist in the correct locations or regenerate mock models
- **PDF/DOCX parsing errors**: Install optional dependencies (PyPDF2, python-docx)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PyQt5 for the GUI framework
- OpenCV for video processing
- scikit-learn for machine learning functionality
- Facebook's wav2vec2 model for speech recognition
