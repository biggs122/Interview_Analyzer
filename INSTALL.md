# Installation Guide for Interview Analyzer Pro

This guide provides detailed instructions for installing and running the Interview Analyzer Pro application on different operating systems.

## Prerequisites

Before installing, ensure your system meets these requirements:
- Python 3.8 or higher
- Webcam and microphone access
- At least 4GB RAM
- 1GB free disk space
- Git (for cloning the repository)

## Installation Methods

### Method 1: Using Conda (Recommended)

Conda provides the most reliable environment for running Interview Analyzer Pro, especially for handling complex dependencies like OpenCV, dlib, and PyAudio.

#### Step 1: Install Anaconda or Miniconda
If you don't have Anaconda or Miniconda installed:
- Download and install from [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Follow the installation instructions for your operating system

#### Step 2: Clone the repository
```bash
git clone https://github.com/yourusername/interview_analyzer.git
cd interview_analyzer
```

#### Step 3: Run the setup script
```bash
bash setup_conda.sh
```

This script will:
- Create a new conda environment called `interview_analyzer`
- Install all required dependencies
- Download necessary NLP models and data

#### Step 4: Activate the environment and run
```bash
conda activate interview_analyzer
python tests/kivy_app_with_real_models.py
```

### Method 2: Using Pip

#### Step 1: Clone the repository
```bash
git clone https://github.com/yourusername/interview_analyzer.git
cd interview_analyzer
```

#### Step 2: Run the application script
```bash
python run_app.py
```

This script will:
- Install all required dependencies
- Download necessary NLP models
- Launch the application

#### Alternative: Manual installation
```bash
pip install -r requirements.txt
python -m nltk.downloader punkt wordnet averaged_perceptron_tagger maxent_ne_chunker words
python -m spacy download en_core_web_sm
python tests/kivy_app_with_real_models.py
```

## Operating System Specific Instructions

### Windows

#### Additional requirements:
- Visual C++ Build Tools might be required for some packages
- You can install them from [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

#### Potential audio issues:
If you encounter issues with PyAudio installation:
```bash
pip install pipwin
pipwin install pyaudio
```

### macOS

#### Additional requirements:
- Command Line Tools for Xcode: `xcode-select --install`
- For M1/M2 Macs, use the Conda installation method for best compatibility

#### Potential issues:
If you encounter issues with dlib installation:
```bash
brew install cmake
pip install dlib
```

### Linux (Ubuntu/Debian)

#### Additional requirements:
```bash
sudo apt-get update
sudo apt-get install -y python3-dev portaudio19-dev python3-pyaudio libopencv-dev
sudo apt-get install -y cmake libopenblas-dev liblapack-dev libx11-dev
```

## Troubleshooting

### Common Installation Issues

1. **PyAudio installation fails**:
   - On Windows: Use `pipwin install pyaudio`
   - On macOS: `brew install portaudio` then `pip install pyaudio`
   - On Linux: `sudo apt-get install portaudio19-dev python3-pyaudio`

2. **dlib installation fails**:
   - Ensure you have CMake installed
   - On Windows: Install Visual C++ Build Tools
   - On macOS: `brew install cmake`
   - On Linux: `sudo apt-get install cmake libopenblas-dev liblapack-dev libx11-dev`

3. **OpenCV issues**:
   - Using Conda resolves most OpenCV issues automatically
   - If using pip, ensure you have development libraries installed

4. **Kivy installation issues**:
   - Refer to [Kivy's installation instructions](https://kivy.org/doc/stable/gettingstarted/installation.html) for your specific OS

### Application Startup Issues

If the application fails to start:
- Check for missing dependencies in the error messages
- Ensure all required data and model files are downloaded
- Verify webcam and microphone permissions 