import sys
import os
import time
import cv2
import numpy as np
import logging
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, 
    QWidget, QTabWidget, QLabel, QSplitter, QFileDialog, QMessageBox,
    QTextEdit, QScrollArea, QGridLayout, QGroupBox, QFrame, QStatusBar, QDesktopWidget
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QFont, QColor, QPalette, QPixmap, QImage

# Import tab classes
from src.pyqt_tabs.cv_analysis_tab import CVAnalysisTab
from src.pyqt_tabs.interview_tab import InterviewTab

# Try to import our backend modules - use mock versions if not available
try:
    from src.utils.analyzer_backend import AnalyzerBackend
    backend_available = True
except ImportError:
    backend_available = False
    from src.pyqt_tabs.mock_backend import MockAnalyzerBackend
    
class ButtonFlasher(QTimer):
    """Timer to handle button flashing animation"""
    def __init__(self, button, parent=None):
        super().__init__(parent)
        self.button = button
        self.original_style = button.styleSheet()
        self.alternate_style = """
            QPushButton {
                background-color: #FFC107;  /* Yellow color */
                color: black;
                border-radius: 5px;
                padding: 20px;
                min-height: 60px;
                font-weight: bold;
            }
        """
        self.is_original = True
        self.timeout.connect(self.toggle_style)
        self.setInterval(500)  # Toggle every 500ms
        
    def toggle_style(self):
        if self.is_original:
            self.button.setStyleSheet(self.alternate_style)
        else:
            self.button.setStyleSheet(self.original_style)
        self.is_original = not self.is_original
        
    def stop_flashing(self):
        self.stop()
        self.button.setStyleSheet(self.original_style)

class InterviewAnalyzerApp(QMainWindow):
    """Main window for PyQt Interview Analyzer application"""
    
    def __init__(self):
        super().__init__()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('InterviewAnalyzerApp')
        
        # Initialize the backend
        if backend_available:
            self.backend = AnalyzerBackend()
            self.logger.info("Using real analyzer backend")
        else:
            self.logger.warning("Using mock interview analyzer. Real analysis not available.")
            self.backend = MockAnalyzerBackend()
        
        # UI settings
        self.setWindowTitle("Interview Analyzer Pro")
        self.setMinimumSize(1200, 800)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #F8FAFC;
            }
            QTabWidget::pane {
                border: none;
                background-color: #F8FAFC;
            }
            QTabBar::tab {
                background-color: #E2E8F0;
                color: #475569;
                padding: 12px 20px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                margin-right: 4px;
                font-size: 14px;
                font-weight: 500;
            }
            QTabBar::tab:selected {
                background-color: #2563EB;
                color: white;
            }
            QTabBar::tab:hover:!selected {
                background-color: #CBD5E1;
            }
        """)
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        # Set up central widget with vertical layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Use system fonts
        font_family = "-apple-system, BlinkMacSystemFont, 'Helvetica Neue', Arial, sans-serif"
        
        # Create header with professional styling
        header = QFrame()
        header.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1E40AF, stop:1 #3B82F6);
                min-height: 80px;
            }}
        """)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(30, 15, 30, 15)
        
        # App title with modern typography
        app_title = QLabel("INTERVIEW ANALYZER PRO")
        app_title.setStyleSheet(f"""
            color: white;
            font-size: 28px;
            font-weight: 700;
            font-family: {font_family};
            letter-spacing: 1px;
        """)
        
        # Logo (if available)
        logo_label = QLabel()
        # If you have a logo file:
        # logo_pixmap = QPixmap("path/to/logo.png").scaled(60, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        # logo_label.setPixmap(logo_pixmap)
        
        header_layout.addWidget(logo_label)
        header_layout.addWidget(app_title)
        header_layout.addStretch()
        
        # Add a version label
        version_label = QLabel("v1.0")
        version_label.setStyleSheet(f"color: rgba(255, 255, 255, 0.8); font-size: 14px; font-family: {font_family};")
        header_layout.addWidget(version_label)
        
        # Add header to main layout
        main_layout.addWidget(header)
        
        # Create a welcome banner
        welcome_banner = QFrame()
        welcome_banner.setStyleSheet(f"""
            QFrame {{
                background-color: #FCD34D;
                min-height: 60px;
            }}
            QPushButton {{
                background-color: #0D6EFD;
                color: white;
                border: none;
                padding: 12px 30px;
                border-radius: 8px;
                font-size: 16px;
                font-weight: 600;
                font-family: {font_family};
            }}
            QPushButton:hover {{
                background-color: #0B5ED7;
            }}
        """)
        welcome_layout = QHBoxLayout(welcome_banner)
        welcome_layout.setContentsMargins(40, 20, 40, 20)
        
        # Add start button to welcome banner
        start_button = QPushButton("START INTERVIEW")
        start_button.setMinimumWidth(200)
        start_button.setMaximumWidth(300)
        start_button.setCursor(Qt.PointingHandCursor)
        start_button.clicked.connect(self.start_interview)
        
        welcome_layout.addStretch()
        welcome_layout.addWidget(start_button)
        welcome_layout.addStretch()
        
        # Add welcome banner to main layout
        main_layout.addWidget(welcome_banner)
        
        # Create tab widget for different sections with system fonts
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                padding: 20px;
            }}
            QTabBar::tab {{
                background-color: #E2E8F0;
                color: #475569;
                padding: 12px 20px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                margin-right: 4px;
                font-size: 14px;
                font-weight: 500;
                font-family: {font_family};
            }}
            QTabBar::tab:selected {{
                background-color: #2563EB;
                color: white;
            }}
            QTabBar::tab:hover:!selected {{
                background-color: #CBD5E1;
            }}
        """)
        
        # Create tabs
        self.cv_tab = CVAnalysisTab(self.backend)
        self.interview_tab = InterviewTab(self.backend, self)
        
        # Add tabs to tab widget
        self.tabs.addTab(self.cv_tab, "CV Analysis")
        self.tabs.addTab(self.interview_tab, "Interview")
        
        # Connect tab changes
        self.tabs.currentChanged.connect(self.on_tab_changed)
        
        # Add tab widget to main layout
        main_layout.addWidget(self.tabs)
        
        # Add a status bar with professional styling and system fonts
        self.statusBar().setStyleSheet(f"""
            QStatusBar {{
                background-color: #F1F5F9;
                color: #64748B;
                padding: 5px;
                font-size: 13px;
                font-family: {font_family};
            }}
        """)
        
        # Show a welcome message
        self.statusBar().showMessage("Welcome to Interview Analyzer Pro. Ready to help you improve your interview skills.")
        
        # Center the window on the screen
        self.center_on_screen()
        
    def center_on_screen(self):
        """Center the window on the screen"""
        try:
            # Get available screen geometry
            screen = QDesktopWidget().availableGeometry()
            self.logger.info(f"Screen size: {screen.width()}x{screen.height()}")
            
            # Set default size to 80% of screen size
            default_width = int(screen.width() * 0.8)
            default_height = int(screen.height() * 0.8)
            self.resize(default_width, default_height)
            
            # Center window
            frame_geometry = self.frameGeometry()
            screen_center = screen.center()
            frame_geometry.moveCenter(screen_center)
            self.move(frame_geometry.topLeft())
            
            self.logger.info(f"Window positioned at: {self.pos().x()},{self.pos().y()} with size: {self.width()}x{self.height()}")
        except Exception as e:
            self.logger.error(f"Error centering window: {e}")
            # Fallback to fixed position and size
            self.resize(1200, 800)
            self.move(100, 100)
        
    def on_tab_changed(self, index):
        """Handle tab change events"""
        tab_name = self.tabs.tabText(index)
        self.statusBar().showMessage(f"Switched to {tab_name} tab")
        
    def start_interview(self):
        """Start an interview session"""
        # Switch to the interview tab
        self.tabs.setCurrentIndex(1)
        
        # Start the interview
        self.interview_tab.prepare_for_interview()
        self.interview_tab.start_interview()
        
    def interview_completed(self):
        """Handle interview completion"""
        # Display notification or any post-interview steps
        self.statusBar().showMessage("Interview completed! Review your results in the Interview tab.")

def create_directory_structure():
    """Create the required directory structure for the application"""
    directories = ['results', 'models', 'data', 'temp']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def main():
    # Create required directories
    create_directory_structure()
    
    try:
        app = QApplication(sys.argv)
        app.setStyle('Fusion')  # Use Fusion style for a cleaner look
        
        # Set application-wide palette for a clean look
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(248, 249, 250))
        palette.setColor(QPalette.WindowText, QColor(33, 37, 41))
        app.setPalette(palette)
        
        print("Creating main window...")
        window = InterviewAnalyzerApp()
        
        # Ensure window is shown
        window.show()
        print("Window should be visible now")
        
        return app.exec_()
    except Exception as e:
        print(f"ERROR: Application crashed during startup: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())