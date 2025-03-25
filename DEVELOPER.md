# Developer Guide for Interview Analyzer Pro

This guide provides information for developers who want to contribute to or extend the Interview Analyzer Pro application.

## Project Architecture

The Interview Analyzer Pro project is organized into several components:

### Core Components

1. **UI Layer**: Implemented with Kivy in `tests/kivy_app_with_real_models.py`
2. **Analysis Engine**: Main functionality in `tests/real_interview_analyzer.py`
3. **Data Processing**: Components in the `src/` directory

### Directory Structure

```
interview_analyzer/
├── app/                      # Application-specific code (future mobile/web deployments)
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
```

## Development Environment Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/interview_analyzer.git
   cd interview_analyzer
   ```

2. **Setup development environment**:
   ```bash
   # Using conda (recommended for development)
   bash setup_conda.sh
   conda activate interview_analyzer
   
   # Install additional development tools
   pip install pytest black isort flake8
   ```

## Main Components

### 1. Interview Analyzer Backend

The core functionality is implemented in `tests/real_interview_analyzer.py`, which contains the `RealInterviewAnalyzer` class. This class is responsible for:

- Loading and initializing ML models
- Processing CV documents
- Analyzing facial expressions
- Processing audio for voice emotion analysis
- Generating reports

### 2. User Interface (Kivy)

The UI is implemented in `tests/kivy_app_with_real_models.py` and contains several key classes:

- `KivyInterviewAnalyzerApp`: Main application class
- `InterviewAnalyzerApp`: Main layout container
- `CVAnalysisTab`: UI for CV analysis
- `InterviewAnalysisTab`: UI for interview analysis

### 3. CV Analysis Pipeline

The CV analysis pipeline processes documents through several stages:
1. Document parsing (PDF, DOCX, TXT)
2. Entity extraction (names, contact info, etc.)
3. Skills identification
4. Career matching

### 4. Interview Analysis Pipeline

The interview analysis component works in real-time to:
1. Capture and analyze facial expressions with MediaPipe
2. Record and process audio for voice emotion analysis
3. Transcribe speech and analyze sentiment
4. Evaluate coding solutions

## Adding New Features

### Adding a New ML Model

1. Place model files in the appropriate directory under `models/`
2. Add model loading code to `RealInterviewAnalyzer.__init__()`
3. Implement inference function in the analyzer class
4. Update UI to expose the new functionality

### Extending CV Analysis

To add support for new CV formats or extraction features:
1. Add new parser in `src/preprocessing/`
2. Update the `analyze_cv()` method in `RealInterviewAnalyzer`

### Adding New Emotions or Metrics

1. Update the emotion detection models in the appropriate module
2. Modify the result data structures in `RealInterviewAnalyzer`
3. Update the UI to display new metrics

## Testing

The project uses pytest for testing. To run tests:

```bash
# Activate the conda environment
conda activate interview_analyzer

# Run all tests
pytest

# Run specific test
pytest tests/test_models.py
```

## Code Style

This project follows these style guidelines:
- PEP 8 for Python code
- Docstrings for all functions and classes
- Type hints where appropriate

Use the following tools to maintain code quality:
```bash
# Format code
black .

# Sort imports
isort .

# Check style
flake8
```

## Building for Distribution

To prepare the application for distribution:

1. **For standalone executables**:
   ```bash
   pip install pyinstaller
   pyinstaller --name InterviewAnalyzerPro tests/kivy_app_with_real_models.py
   ```

2. **For pip package**:
   Update setup.py and use:
   ```bash
   python setup.py sdist bdist_wheel
   ```

## Troubleshooting Development Issues

### Common Issues

1. **Model loading errors**:
   - Ensure all model files are present in the correct locations
   - Check the model versions match what's expected in the code

2. **UI rendering issues**:
   - Kivy has different behaviors across platforms
   - Test UI changes on all target platforms

3. **Audio/Video capture issues**:
   - Different platforms handle device access differently
   - Use the abstraction in `src/utils/` for cross-platform compatibility 