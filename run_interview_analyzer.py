#!/usr/bin/env python3
"""
Interview Analyzer Pro - PyQt5 Application

This script runs the PyQt5 version of the Interview Analyzer Pro application.
"""

import sys
import os
import logging

from PyQt5.QtWidgets import QApplication
from src.pyqt_interview_analyzer import InterviewAnalyzerApp

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Run the Interview Analyzer application"""
    logger.info("Starting Interview Analyzer Pro application")
    
    # Create required directories
    for directory in ['results', 'temp', 'models']:
        os.makedirs(directory, exist_ok=True)
    
    # Create QApplication instance
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Use Fusion style for a cleaner look
    
    # Create and show the main window
    logger.info("Creating main application window")
    window = InterviewAnalyzerApp()
    
    # Run the application
    logger.info("Running application main loop")
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 