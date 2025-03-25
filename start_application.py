#!/usr/bin/env python3
"""
Interview Analyzer Pro - Startup Script

This script ensures proper display settings before running the application.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ensure_display_settings():
    """Run the display fix script and ensure proper settings"""
    logger.info("Checking display settings...")
    
    fix_script = Path('fix_display.py')
    if fix_script.exists():
        try:
            # Run the display fix script
            subprocess.check_call([sys.executable, str(fix_script)])
            logger.info("Display settings updated successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error updating display settings: {e}")
    else:
        logger.warning(f"Display fix script not found: {fix_script}")

def run_application():
    """Run the main application with proper settings"""
    logger.info("Starting Interview Analyzer Pro...")
    
    # Find the main application script
    main_script = Path('run_application.py')
    if main_script.exists():
        try:
            # Run the main application
            subprocess.check_call([sys.executable, str(main_script)])
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running application: {e}")
            return 1
    else:
        logger.error(f"Main application script not found: {main_script}")
        return 1
    
    return 0

if __name__ == "__main__":
    # Ensure proper display settings
    ensure_display_settings()
    
    # Run the application
    sys.exit(run_application())
