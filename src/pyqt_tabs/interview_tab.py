import os
import sys
import time
import threading
import cv2
import numpy as np
import pyttsx3
from datetime import datetime, timedelta
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit, 
    QGroupBox, QGridLayout, QSplitter, QFrame, QMessageBox, QScrollArea,
    QComboBox, QProgressBar, QSlider, QCheckBox, QListWidget, QTabWidget,
    QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot, QThread, QSize, QObject
from PyQt5.QtGui import QFont, QPixmap, QImage, QIcon

class VideoThread(QThread):
    """Thread for capturing video frames"""
    update_frame = pyqtSignal(QImage)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self.cap = None
        
    def run(self):
        self.running = True
        self.cap = cv2.VideoCapture(0)  # Default camera
        
        if not self.cap.isOpened():
            return
            
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Convert frame to format suitable for Qt
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            self.update_frame.emit(qt_image)
            time.sleep(0.03)  # ~30 FPS
            
    def stop(self):
        self.running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()

class AudioRecordThread(QThread):
    """Thread for recording audio"""
    update_audio_level = pyqtSignal(int)
    recording_complete = pyqtSignal(str)  # Signal emitted with path to audio file
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self.output_file = None
        
    def run(self):
        self.running = True
        
        # Mock recording for demo purposes
        # In a real implementation, use pyaudio, sounddevice, or other audio libraries
        for i in range(100):
            if not self.running:
                break
                
            # Simulate audio level (0-100)
            audio_level = np.random.randint(0, 100)
            self.update_audio_level.emit(audio_level)
            
            time.sleep(0.1)
            
        # Signal recording complete with mock file path
        if self.running and self.output_file:
            # In a real implementation, actually save the audio file here
            self.recording_complete.emit(self.output_file)
            
    def stop(self):
        self.running = False

class TTSPlayer(QObject):
    """Text-to-speech player that works safely with PyQt"""
    finished = pyqtSignal()
    status_update = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.text = ""
        self.engine = None
        self.is_speaking = False
        
    def set_text(self, text):
        """Set the text to be spoken"""
        self.text = text
        
    def play(self):
        """Play the text using a safer approach"""
        self.is_speaking = True
        self.status_update.emit("Starting AI voice...")
        
        # Create the engine in the same thread that will use it
        try:
            self.engine = pyttsx3.init()
            self.setup_engine()
            self.speak_text()
        except Exception as e:
            print(f"TTS Error: {e}")
            self.is_speaking = False
            self.finished.emit()
    
    def setup_engine(self):
        """Set up the TTS engine with optimal settings"""
        if not self.engine:
            return
            
        # Configure voice properties for better clarity
        self.engine.setProperty('rate', 150)  # Speech rate
        self.engine.setProperty('volume', 1.0)  # Maximum volume
        
        # Select the best English voice available
        voices = self.engine.getProperty('voices')
        
        # Look for a good English voice
        selected_voice = None
        
        # Priority list: 1. Samantha, 2. Any English US, 3. Any English voice, 4. First available
        for voice in voices:
            voice_id = voice.id.lower()
            voice_name = voice.name.lower()
            
            # Print voice details for debugging
            print(f"Checking voice: {voice.name} ({voice.id})")
            
            # Best option: Samantha or similar high-quality voices
            if "samantha" in voice_name:
                selected_voice = voice
                print(f"Selected preferred voice: {voice.name}")
                break
                
            # Second option: Any en-US voice
            if "en-us" in voice_id or "en_us" in voice_id:
                selected_voice = voice
                print(f"Selected en-US voice: {voice.name}")
                # Don't break yet, keep looking for Samantha
                
            # Third option: Any English voice
            elif not selected_voice and ("en-" in voice_id or "en_" in voice_id):
                selected_voice = voice
                print(f"Selected English voice: {voice.name}")
                
        # If no English voice found, use the first voice
        if not selected_voice and voices:
            selected_voice = voices[0]
            print(f"No English voice found, using: {selected_voice.name}")
            
        # Set the selected voice
        if selected_voice:
            self.engine.setProperty('voice', selected_voice.id)
            print(f"Using voice: {selected_voice.name}")
    
    def speak_text(self):
        """Speak the text and handle completion"""
        if not self.engine or not self.text or not self.is_speaking:
            self.finished.emit()
            return
            
        self.status_update.emit("AI speaking question...")
        
        # Define callback for speaking completed
        def on_complete(name, completed):
            if self.is_speaking:
                self.status_update.emit("AI voice finished")
                self.is_speaking = False
                self.finished.emit()
        
        # Connect the callback
        self.engine.connect('finished-utterance', on_complete)
        
        # Say the text
        print(f"Speaking: {self.text}")
        self.engine.say(self.text)
        
        # Run in blocking mode (this is safe when called from a dedicated thread)
        self.engine.runAndWait()
        
        # Make sure we emit finished signal in case callback didn't work
        if self.is_speaking:
            self.is_speaking = False
            self.finished.emit()
    
    def stop(self):
        """Stop the TTS engine"""
        self.is_speaking = False
        if self.engine:
            try:
                self.engine.stop()
            except:
                pass

class VoiceQuestionPlayer(QThread):
    """Thread for playing voice questions"""
    question_completed = pyqtSignal()
    status_update = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.question_text = ""
        self.running = False
        self.tts_player = None
        
    def set_question(self, question_text):
        self.question_text = question_text
        
    def run(self):
        self.running = True
        self.status_update.emit("AI voice speaking...")
        
        # Use the new TTS player but run it in this thread
        self.tts_player = TTSPlayer(self)
        self.tts_player.status_update.connect(self.status_update)
        self.tts_player.finished.connect(self._on_tts_finished)
        self.tts_player.set_text(self.question_text)
        
        # Run the TTS player
        self.tts_player.play()
        
    def _on_tts_finished(self):
        """Handle TTS completion"""
        if self.running:
            self.status_update.emit("AI voice finished, now listening for your response")
            self.question_completed.emit()
    
    def stop(self):
        self.running = False
        if self.tts_player:
            self.tts_player.stop()

class InterviewTab(QWidget):
    """Interview tab for PyQt Interview Analyzer application"""
    
    def __init__(self, backend, parent=None):
        super().__init__(parent)
        self.backend = backend
        self.main_app = parent
        
        # Interview state
        self.interview_in_progress = False
        self.current_question_idx = 0
        self.interview_results = []
        self.questions = []
        self.mock_questions = [
            "Tell me about yourself.",
            "What are your greatest strengths?",
            "What are your greatest weaknesses?",
            "Why are you interested in this position?",
            "Where do you see yourself in 5 years?",
            "Describe a challenging situation at work and how you handled it."
        ]
        
        # Interview duration settings
        self.interview_duration_minutes = 10
        self.interview_start_time = None
        self.interview_end_time = None
        
        # Question state
        self.is_listening_for_response = False
        self.voice_only_questions = True
        
        # Video capture thread
        self.video_thread = None
        
        # Audio recording thread
        self.audio_thread = None
        
        # Voice question player
        self.voice_player = None
        
        # Timers
        self.countdown_timer = QTimer(self)
        self.countdown_timer.timeout.connect(self.update_countdown)
        self.countdown_seconds = 0
        
        self.response_timer = QTimer(self)
        self.response_timer.timeout.connect(self.update_response_time)
        self.response_seconds = 0
        
        # Interview timer - checks overall interview duration
        self.interview_timer = QTimer(self)
        self.interview_timer.timeout.connect(self.check_interview_duration)
        
        # Set default window size and properties
        self.setMinimumSize(800, 600)  # Set minimum window size
        self.setSizePolicy(
            QSizePolicy.Expanding, 
            QSizePolicy.Expanding
        )
        
        # Window resize event handler
        self.resized = False
        
        # Screen metrics for responsive design
        self.screen_metrics = {
            "width": 1920,  # Default screen width
            "height": 1080,  # Default screen height
            "scale_factor": 1.0,  # Default scale factor
            "font_scale": 1.0     # Default font scale
        }
        
        # Initialize UI metrics with the screen metrics
        self.init_screen_metrics()
        
        # Initialize UI
        self.init_ui()
        
        # Resize after UI is initialized
        if parent is None:
            # If used standalone, set a default size
            self.resize(1200, 800)
            
        # Connect resize event
        self.resizeEvent = self.on_resize
        
    def init_screen_metrics(self):
        """Initialize screen metrics for responsive design"""
        # Get the screen geometry from QApplication
        try:
            from PyQt5.QtWidgets import QApplication
            screen = QApplication.primaryScreen()
            if screen:
                screen_geometry = screen.availableGeometry()
                screen_dpi = screen.logicalDotsPerInch()
                
                # Update screen metrics based on actual screen
                self.screen_metrics["width"] = screen_geometry.width()
                self.screen_metrics["height"] = screen_geometry.height()
                
                # Calculate scale based on reference resolution of 1920x1080
                width_scale = screen_geometry.width() / 1920
                height_scale = screen_geometry.height() / 1080
                
                # Use the smaller scale to ensure everything fits
                self.screen_metrics["scale_factor"] = min(width_scale, height_scale)
                
                # Calculate font scaling based on DPI
                self.screen_metrics["font_scale"] = screen_dpi / 96.0  # 96 is the reference DPI
                
                print(f"Screen Metrics: {self.screen_metrics}")
            else:
                print("Could not detect primary screen, using default metrics")
        except Exception as e:
            # If there's any error, use default metrics
            print(f"Error detecting screen metrics: {e}. Using defaults.")
            self.screen_metrics = {
                "width": 1920,
                "height": 1080,
                "scale_factor": 1.0,
                "font_scale": 1.0
            }
            
    def scaled_size(self, width, height):
        """Return a size scaled by the screen's scale factor"""
        try:
            return QSize(
                int(width * self.screen_metrics["scale_factor"]),
                int(height * self.screen_metrics["scale_factor"])
            )
        except (KeyError, TypeError):
            # Fallback in case of issues
            return QSize(width, height)
        
    def scaled_margins(self, left, top, right, bottom):
        """Return a tuple of margins scaled by the screen's scale factor"""
        try:
            return (
                int(left * self.screen_metrics["scale_factor"]),
                int(top * self.screen_metrics["scale_factor"]), 
                int(right * self.screen_metrics["scale_factor"]),
                int(bottom * self.screen_metrics["scale_factor"])
            )
        except (KeyError, TypeError):
            # Fallback in case of issues
            return (left, top, right, bottom)
        
    def scaled_font_size(self, size):
        """Return a font size scaled for the current display"""
        try:
            return int(size * self.screen_metrics["font_scale"])
        except (KeyError, TypeError):
            # Fallback in case of issues
            return size
        
    def on_resize(self, event):
        """Handle window resize events"""
        # Call the parent class's resize event
        super().resizeEvent(event)
        
        # Store the new size
        self.resized = True
        
        # If we have a splitter, maintain proper proportions
        if hasattr(self, 'main_splitter') and self.main_splitter:
            # Maintain a 40/60 split or the user's last split
            current_total = sum(self.main_splitter.sizes())
            if current_total > 0:
                left_ratio = 0.4  # Default left panel ratio
                right_ratio = 0.6  # Default right panel ratio
                
                # Set new sizes based on ratios
                self.main_splitter.setSizes([
                    int(current_total * left_ratio),
                    int(current_total * right_ratio)
                ])
        
        # Adjust panels based on new size
        self.adjust_for_screen_size()
        
    def adjust_for_screen_size(self):
        """Adjust UI elements based on current screen size"""
        # Get the current width and height
        width = self.width()
        height = self.height()
        
        # Calculate a scaling factor based on current size
        width_scale = width / 1200  # Reference width
        height_scale = height / 800  # Reference height
        scale = min(width_scale, height_scale)
        
        # Only make adjustments if significant changes
        if not self.resized or scale < 0.5 or scale > 2.0:
            return
            
        # Reset the flag
        self.resized = False
        
        # Adjust font sizes based on window size
        base_font_size = self.scaled_font_size(13)
        small_font_size = self.scaled_font_size(12)
        medium_font_size = self.scaled_font_size(15)
        large_font_size = self.scaled_font_size(17)
        xlarge_font_size = self.scaled_font_size(20)
        
        # Check if video label exists
        if hasattr(self, 'video_label'):
            # Scale the video preview proportionally
            video_width = int(450 * scale)
            video_height = int(338 * scale)
            self.video_label.setMinimumSize(video_width, video_height)
        
        # Adjust text elements if they exist
        if hasattr(self, 'question_label'):
            font = self.question_label.font()
            font.setPointSize(int(xlarge_font_size * scale))
            self.question_label.setFont(font)
        
        if hasattr(self, 'response_text'):
            font = self.response_text.font()
            font.setPointSize(int(base_font_size * scale))
            self.response_text.setFont(font)
            
        if hasattr(self, 'feedback_text'):
            font = self.feedback_text.font()
            font.setPointSize(int(base_font_size * scale))
            self.feedback_text.setFont(font)
            
        if hasattr(self, 'results_text'):
            font = self.results_text.font()
            font.setPointSize(int(base_font_size * scale))
            self.results_text.setFont(font)

    def init_ui(self):
        """Initialize the Interview tab UI"""
        # Set application-wide stylesheet variables
        self.colors = {
            "primary": "#2563EB",         # Vibrant blue for primary actions
            "primary_hover": "#1D4ED8",   # Darker blue for hover states
            "secondary": "#4B5563",       # Dark gray for secondary elements
            "accent": "#3B82F6",          # Light blue for accents
            "success": "#10B981",         # Green for success indicators
            "danger": "#EF4444",          # Red for warnings/dangers
            "danger_hover": "#DC2626",    # Darker red for hover states
            "warning": "#F59E0B",         # Amber for warnings
            "text_primary": "#1F2937",    # Dark gray for primary text
            "text_secondary": "#6B7280",  # Medium gray for secondary text
            "text_light": "#F9FAFB",      # Off-white for text on dark backgrounds
            "background": "#FFFFFF",      # White for main backgrounds
            "background_alt": "#F3F4F6",  # Light gray for alternate backgrounds
            "border": "#E5E7EB",          # Light gray for borders
        }
        
        # Calculate font sizes based on screen metrics
        base_font_size = self.scaled_font_size(13)
        small_font_size = self.scaled_font_size(12)
        medium_font_size = self.scaled_font_size(15)
        large_font_size = self.scaled_font_size(17)
        xlarge_font_size = self.scaled_font_size(20)
        
        # Apply a base stylesheet with dynamic font sizes - using system fonts
        self.setStyleSheet(f"""
            QWidget {{
                font-family: -apple-system, BlinkMacSystemFont, 'Helvetica Neue', Arial, sans-serif;
                color: {self.colors["text_primary"]};
                background-color: {self.colors["background"]};
                font-size: {base_font_size}px;
            }}
            QGroupBox {{
                font-weight: bold;
                border: 1px solid {self.colors["border"]};
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 16px;
                font-size: {medium_font_size}px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: {self.colors["primary"]};
            }}
            QLabel {{
                color: {self.colors["text_primary"]};
                font-size: {base_font_size}px;
            }}
            QTextEdit {{
                border: 1px solid {self.colors["border"]};
                border-radius: 6px;
                padding: 10px;
                background-color: white;
                font-size: {base_font_size}px;
            }}
            QPushButton {{
                font-size: {medium_font_size}px;
            }}
            QTabWidget {{
                font-size: {base_font_size}px;
            }}
            QComboBox {{
                font-size: {base_font_size}px;
                padding: 6px 12px;
                border-radius: 6px;
                min-height: 32px;
            }}
            QProgressBar {{
                text-align: center;
            }}
        """)
        
        # Main layout with dynamic margins
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(
            self.scaled_size(25, 25).width(),
            self.scaled_size(25, 25).height(),
            self.scaled_size(25, 25).width(),
            self.scaled_size(25, 25).height()
        )
        main_layout.setSpacing(self.scaled_size(20, 20).height())
        
        # Create horizontal splitter with proper sizing
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setHandleWidth(2)
        self.main_splitter.setStyleSheet(f"""
            QSplitter::handle {{
                background-color: {self.colors["border"]};
            }}
        """)
        
        #------------------- Left Panel --------------------#
        left_panel = QWidget()
        left_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, self.scaled_size(12, 0).width(), 0)
        left_layout.setSpacing(self.scaled_size(18, 18).height())
        
        # Video preview section
        video_group = QGroupBox("Video Preview")
        video_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        video_layout = QVBoxLayout(video_group)
        video_layout.setContentsMargins(
            *self.scaled_margins(18, 24, 18, 18)
        )
        
        # Create a container for the video and placeholder
        video_container = QWidget()
        video_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        video_container.setStyleSheet(f"""
            background-color: #000000;
            border-radius: 10px;
        """)
        video_container_layout = QVBoxLayout(video_container)
        video_container_layout.setContentsMargins(0, 0, 0, 0)
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(self.scaled_size(450, 338))  # 4:3 aspect ratio
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setStyleSheet("background-color: transparent;")
        
        self.no_video_label = QLabel("Video will appear here when interview starts")
        self.no_video_label.setAlignment(Qt.AlignCenter)
        self.no_video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.no_video_label.setStyleSheet(f"""
            color: white;
            font-size: {large_font_size}px;
            font-weight: 500;
            padding: 20px;
        """)
        
        # Initially show the placeholder text in the container
        video_container_layout.addWidget(self.no_video_label)
        
        # Add the container to the video layout
        video_layout.addWidget(video_container)
        
        # Store the container for later use
        self.video_container = video_container
        
        # Audio level section
        audio_group = QGroupBox("Audio Level")
        audio_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        audio_layout = QVBoxLayout(audio_group)
        audio_layout.setContentsMargins(
            *self.scaled_margins(18, 24, 18, 18)
        )
        
        self.audio_level = QProgressBar()
        self.audio_level.setMinimum(0)
        self.audio_level.setMaximum(100)
        self.audio_level.setValue(0)
        self.audio_level.setTextVisible(False)
        self.audio_level.setFixedHeight(self.scaled_size(0, 10).height())
        self.audio_level.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.audio_level.setStyleSheet(f"""
            QProgressBar {{
                border: none;
                background-color: {self.colors["background_alt"]};
                border-radius: 5px;
            }}
            QProgressBar::chunk {{
                background-color: {self.colors["accent"]};
                border-radius: 5px;
            }}
        """)
        
        level_label = QLabel("Current audio level")
        level_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        level_label.setStyleSheet(f"""
            font-size: {small_font_size}px;
            color: {self.colors["text_secondary"]};
            margin-bottom: 6px;
        """)
        
        audio_layout.addWidget(level_label)
        audio_layout.addWidget(self.audio_level)
        
        # Interview controls section
        controls_group = QGroupBox("Interview Controls")
        controls_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        controls_layout = QVBoxLayout(controls_group)
        controls_layout.setContentsMargins(
            *self.scaled_margins(18, 24, 18, 18)
        )
        controls_layout.setSpacing(self.scaled_size(18, 18).height())
        
        # Interview mode selection
        mode_layout = QHBoxLayout()
        mode_label = QLabel("Interview Mode:")
        mode_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        mode_label.setStyleSheet(f"""
            font-weight: 500;
            color: {self.colors["text_primary"]};
            font-size: {base_font_size}px;
        """)
        
        self.mode_combo = QComboBox()
        self.mode_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.mode_combo.addItems(["Automatic", "Manual", "Practice"])
        self.mode_combo.setToolTip("Automatic: AI-guided interview\nManual: You control the questions\nPractice: For interview practice")
        self.mode_combo.setStyleSheet(f"""
            QComboBox {{
                border: 1px solid {self.colors["border"]};
                border-radius: 6px;
                padding: 8px 12px;
                background-color: white;
                font-size: {base_font_size}px;
                min-height: 36px;
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 24px;
                border-left: 1px solid {self.colors["border"]};
            }}
        """)
        
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.mode_combo)
        
        controls_layout.addLayout(mode_layout)
        
        # Voice-only questions checkbox - Always checked and disabled since we always want voice
        voice_layout = QHBoxLayout()
        self.voice_questions_checkbox = QCheckBox("Voice-only questions")
        self.voice_questions_checkbox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.voice_questions_checkbox.setChecked(True)
        self.voice_questions_checkbox.setEnabled(False)  # Disable to enforce voice questions
        self.voice_questions_checkbox.setToolTip("Questions will be spoken aloud by AI voice")
        self.voice_questions_checkbox.setStyleSheet(f"""
            QCheckBox {{
                font-size: {base_font_size}px;
                min-height: 30px;
            }}
            QCheckBox::indicator {{
                width: {self.scaled_size(20, 20).width()}px;
                height: {self.scaled_size(20, 20).height()}px;
            }}
            QCheckBox::indicator:checked {{
                image: url(icons/checkbox_checked.png);
                background-color: {self.colors["primary"]};
                border-radius: 4px;
            }}
            QCheckBox::indicator:unchecked {{
                image: url(icons/checkbox_unchecked.png);
                border: 1px solid {self.colors["border"]};
                border-radius: 4px;
            }}
        """)
        voice_layout.addWidget(self.voice_questions_checkbox)
        
        controls_layout.addLayout(voice_layout)
        
        # Interview duration
        duration_layout = QHBoxLayout()
        duration_label = QLabel("Duration:")
        duration_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        duration_label.setStyleSheet(f"""
            font-weight: 500;
            color: {self.colors["text_primary"]};
            font-size: {base_font_size}px;
        """)
        
        self.duration_label = QLabel(f"{self.interview_duration_minutes} minutes")
        self.duration_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.duration_label.setStyleSheet(f"""
            color: {self.colors["text_secondary"]};
            font-weight: 400;
            font-size: {base_font_size}px;
        """)
        duration_layout.addWidget(duration_label)
        duration_layout.addWidget(self.duration_label)
        
        controls_layout.addLayout(duration_layout)
        
        # Control buttons
        button_layout = QVBoxLayout()
        button_layout.setSpacing(self.scaled_size(12, 12).height())
        
        self.start_button = QPushButton("START INTERVIEW")
        self.start_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.start_button.setMinimumHeight(self.scaled_size(0, 45).height())
        self.start_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.colors["success"]};
                color: white;
                padding: {self.scaled_size(14, 14).height()}px;
                font-weight: 600;
                border-radius: 8px;
                font-size: {medium_font_size}px;
            }}
            QPushButton:hover {{
                background-color: #0CA36E;
            }}
            QPushButton:disabled {{
                background-color: #A1A1AA;
                color: #F1F1F1;
            }}
        """)
        self.start_button.clicked.connect(self.start_interview)
        
        self.stop_button = QPushButton("END INTERVIEW")
        self.stop_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.stop_button.setMinimumHeight(self.scaled_size(0, 45).height())
        self.stop_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.colors["danger"]};
                color: white;
                padding: {self.scaled_size(14, 14).height()}px;
                font-weight: 600;
                border-radius: 8px;
                font-size: {medium_font_size}px;
            }}
            QPushButton:hover {{
                background-color: {self.colors["danger_hover"]};
            }}
            QPushButton:disabled {{
                background-color: #A1A1AA;
                color: #F1F1F1;
            }}
        """)
        self.stop_button.clicked.connect(self.stop_interview)
        self.stop_button.setEnabled(False)
        
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        
        controls_layout.addLayout(button_layout)
        
        # Add sections to left panel with dynamic stretch factors
        left_layout.addWidget(video_group, 5)
        left_layout.addWidget(audio_group, 1)
        left_layout.addWidget(controls_group, 3)
        
        #------------------- Right Panel --------------------#
        right_panel = QWidget()
        right_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(
            self.scaled_size(12, 0).width(), 
            0, 
            0, 
            0
        )
        right_layout.setSpacing(self.scaled_size(18, 18).height())
        
        # Question section with enhanced styling
        question_group = QGroupBox("Current Question")
        question_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        question_group.setStyleSheet(f"""
            QGroupBox {{
                font-weight: 600;
                border: 1px solid {self.colors["border"]};
                border-radius: 8px;
                background-color: {self.colors["background"]};
                font-size: {medium_font_size}px;
            }}
        """)
        question_layout = QVBoxLayout(question_group)
        question_layout.setContentsMargins(
            *self.scaled_margins(18, 28, 18, 18)
        )
        question_layout.setSpacing(self.scaled_size(14, 14).height())
        
        self.countdown_label = QLabel("Next question in: 10s")
        self.countdown_label.setAlignment(Qt.AlignCenter)
        self.countdown_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.countdown_label.setStyleSheet(f"""
            font-size: {medium_font_size}px;
            color: {self.colors["primary"]};
            font-weight: 600;
            padding: 8px;
            background-color: #EFF6FF;
            border-radius: 6px;
        """)
        self.countdown_label.setVisible(False)
        
        # Question card with shadow effect
        question_card = QFrame()
        question_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        question_card.setStyleSheet(f"""
            QFrame {{
                background-color: white;
                border-radius: 10px;
                border: 1px solid {self.colors["border"]};
                padding: 10px;
            }}
        """)
        question_card_layout = QVBoxLayout(question_card)
        question_card_layout.setContentsMargins(
            *self.scaled_margins(20, 20, 20, 20)
        )
        
        self.question_label = QLabel("Questions will appear here")
        self.question_label.setAlignment(Qt.AlignCenter)
        self.question_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.question_label.setStyleSheet(f"""
            font-size: {xlarge_font_size}px;
            font-weight: 600;
            color: {self.colors["text_primary"]};
            padding: 15px;
            line-height: 1.5;
        """)
        self.question_label.setWordWrap(True)
        question_card_layout.addWidget(self.question_label)
        
        # Add the card to the question layout
        question_layout.addWidget(self.countdown_label)
        question_layout.addWidget(question_card)
        
        # Info bar with time and emotion
        info_bar = QFrame()
        info_bar.setStyleSheet(f"""
            background-color: {self.colors["background_alt"]};
            border-radius: 8px;
            padding: 5px;
        """)
        info_layout = QHBoxLayout(info_bar)
        info_layout.setContentsMargins(12, 12, 12, 12)
        
        # Time remaining indicator
        time_box = QFrame()
        time_box.setStyleSheet(f"""
            background-color: white;
            border-radius: 6px;
            padding: 5px;
        """)
        time_layout = QHBoxLayout(time_box)
        time_layout.setContentsMargins(10, 10, 10, 10)
        
        time_icon = QLabel("⏱")
        time_icon.setStyleSheet("font-size: 18px;")
        
        self.time_label = QLabel("Time remaining: 10:00")
        self.time_label.setStyleSheet(f"""
            font-size: {base_font_size}px;
            color: {self.colors["text_primary"]};
            font-weight: 500;
        """)
        
        time_layout.addWidget(time_icon)
        time_layout.addWidget(self.time_label)
        
        # Emotion indicator
        emotion_box = QFrame()
        emotion_box.setStyleSheet(f"""
            background-color: white;
            border-radius: 6px;
            padding: 5px;
        """)
        emotion_layout = QHBoxLayout(emotion_box)
        emotion_layout.setContentsMargins(10, 10, 10, 10)
        
        emotion_icon = QLabel("😊")
        emotion_icon.setStyleSheet("font-size: 18px;")
        
        self.emotion_label = QLabel("Current emotion: Neutral")
        self.emotion_label.setStyleSheet(f"""
            font-size: {base_font_size}px;
            color: {self.colors["text_primary"]};
            font-weight: 500;
        """)
        
        emotion_layout.addWidget(emotion_icon)
        emotion_layout.addWidget(self.emotion_label)
        
        # Add time and emotion to info bar
        info_layout.addWidget(time_box, 1)
        info_layout.addWidget(emotion_box, 1)
        
        # Add info bar to question layout
        question_layout.addWidget(info_bar)
        
        # Questions list - Hidden as requested
        self.questions_list = QListWidget()
        self.questions_list.setVisible(False)  # Hide the questions list
        
        # Progress section with visual indicators
        progress_group = QGroupBox("Interview Progress")
        progress_layout = QVBoxLayout(progress_group)
        progress_layout.setContentsMargins(*self.scaled_margins(18, 28, 18, 18))
        
        # Progress indicator row
        progress_indicator = QFrame()
        progress_indicator_layout = QHBoxLayout(progress_indicator)
        progress_indicator_layout.setContentsMargins(0, 0, 0, 0)
        
        progress_text = QLabel("Overall Progress")
        progress_text.setStyleSheet(f"""
            font-size: {base_font_size}px;
            font-weight: 500;
            color: {self.colors["text_secondary"]};
        """)
        
        self.progress_percentage = QLabel("0%")
        self.progress_percentage.setStyleSheet(f"""
            font-size: {medium_font_size}px;
            font-weight: 600;
            color: {self.colors["primary"]};
        """)
        
        progress_indicator_layout.addWidget(progress_text)
        progress_indicator_layout.addStretch()
        progress_indicator_layout.addWidget(self.progress_percentage)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("")  # Remove percentage text inside bar
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: none;
                background-color: {self.colors["background_alt"]};
                border-radius: 5px;
                height: {self.scaled_size(0, 10).height()}px;
            }}
            QProgressBar::chunk {{
                background-color: {self.colors["primary"]};
                border-radius: 5px;
            }}
        """)
        
        progress_layout.addWidget(progress_indicator)
        progress_layout.addWidget(self.progress_bar)
        
        # Response section
        response_group = QGroupBox("Your Response")
        response_layout = QVBoxLayout(response_group)
        response_layout.setContentsMargins(*self.scaled_margins(18, 28, 18, 18))
        response_layout.setSpacing(14)
        
        # Description label
        response_desc = QLabel("Type or record your answer to the current question")
        response_desc.setStyleSheet(f"""
            font-size: {base_font_size}px;
            color: {self.colors["text_secondary"]};
            margin-bottom: 6px;
        """)
        response_layout.addWidget(response_desc)
        
        # Response text area
        self.response_text = QTextEdit()
        self.response_text.setPlaceholderText("Type or record your response here...")
        self.response_text.setStyleSheet(f"""
            border: 1px solid {self.colors["border"]};
            border-radius: 8px;
            padding: 15px;
            font-size: {base_font_size}px;
            background-color: white;
            min-height: 120px;
        """)
        self.response_text.setMinimumHeight(140)
        response_layout.addWidget(self.response_text)
        
        # Button row
        button_row = QHBoxLayout()
        button_row.setSpacing(15)
        
        self.record_button = QPushButton("RECORD ANSWER")
        self.record_button.setIcon(QIcon("icons/mic.png"))  # Add an icon if available
        self.record_button.setMinimumHeight(self.scaled_size(0, 45).height())
        self.record_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.colors["success"]};
                color: white;
                padding: {self.scaled_size(14, 14).height()}px;
                font-weight: 600;
                border-radius: 8px;
                font-size: {medium_font_size}px;
            }}
            QPushButton:hover {{
                background-color: #0CA36E;
            }}
        """)
        self.record_button.clicked.connect(self.start_recording)
        
        self.continue_button = QPushButton("NEXT QUESTION")
        self.continue_button.setIcon(QIcon("icons/arrow_right.png"))  # Add an icon if available
        self.continue_button.setMinimumHeight(self.scaled_size(0, 45).height())
        self.continue_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.colors["primary"]};
                color: white;
                padding: {self.scaled_size(14, 14).height()}px;
                font-weight: 600;
                border-radius: 8px;
                font-size: {medium_font_size}px;
            }}
            QPushButton:hover {{
                background-color: {self.colors["primary_hover"]};
            }}
        """)
        self.continue_button.clicked.connect(self.show_next_question)
        self.continue_button.hide()  # Initially hidden
        
        button_row.addWidget(self.record_button)
        button_row.addWidget(self.continue_button)
        
        response_layout.addLayout(button_row)
        
        # Feedback panel with tabs
        self.feedback_panel = QTabWidget()
        self.feedback_panel.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 1px solid {self.colors["border"]};
                border-radius: 8px;
                background-color: white;
            }}
            QTabBar::tab {{
                background-color: {self.colors["background_alt"]};
                border: 1px solid {self.colors["border"]};
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                padding: 12px 18px;
                font-size: {base_font_size}px;
                font-weight: 500;
                margin-right: 3px;
            }}
            QTabBar::tab:selected {{
                background-color: white;
                border-bottom-color: white;
                color: {self.colors["primary"]};
            }}
        """)
        
        # Real-time feedback tab
        feedback_tab = QWidget()
        feedback_tab_layout = QVBoxLayout(feedback_tab)
        feedback_tab_layout.setContentsMargins(18, 18, 18, 18)
        
        self.feedback_text = QTextEdit()
        self.feedback_text.setReadOnly(True)
        self.feedback_text.setStyleSheet(f"""
            border: none;
            background-color: white;
            font-size: {base_font_size}px;
        """)
        self.feedback_text.setPlaceholderText("Feedback will appear here after each response")
        
        feedback_tab_layout.addWidget(self.feedback_text)
        
        # Results tab
        results_tab = QWidget()
        results_tab_layout = QVBoxLayout(results_tab)
        results_tab_layout.setContentsMargins(18, 18, 18, 18)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setStyleSheet(f"""
            border: none;
            background-color: white;
            font-size: {base_font_size}px;
        """)
        self.results_text.setPlaceholderText("Final results will appear here when the interview is complete")
        
        results_tab_layout.addWidget(self.results_text)
        
        # Add tabs to the panel
        self.feedback_panel.addTab(feedback_tab, "Response Feedback")
        self.feedback_panel.addTab(results_tab, "Final Results")
        
        # Status bar with modern styling
        status_bar = QFrame()
        status_bar.setFrameShape(QFrame.StyledPanel)
        status_bar.setStyleSheet(f"""
            background-color: {self.colors["background_alt"]};
            border-radius: 8px;
            padding: 5px;
        """)
        status_layout = QHBoxLayout(status_bar)
        status_layout.setContentsMargins(15, 10, 15, 10)
        
        status_icon = QLabel("📌")
        status_icon.setStyleSheet("font-size: 18px;")
        
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet(f"""
            color: {self.colors["text_secondary"]};
            font-size: {base_font_size}px;
            font-weight: 400;
            padding-left: 8px;
        """)
        
        status_layout.addWidget(status_icon)
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        
        # Add sections to right panel with proportional sizing
        right_layout.addWidget(question_group, 3)
        right_layout.addWidget(progress_group, 1)
        right_layout.addWidget(response_group, 3)
        right_layout.addWidget(self.feedback_panel, 4)
        right_layout.addWidget(status_bar, 1)
        
        # Add panels to splitter
        self.main_splitter.addWidget(left_panel)
        self.main_splitter.addWidget(right_panel)
        
        # Calculate initial split proportions
        left_width = int(self.width() * 0.4)
        right_width = int(self.width() * 0.6)
        self.main_splitter.setSizes([left_width, right_width])
        
        # Add splitter to main layout
        main_layout.addWidget(self.main_splitter)
        
        # Configure the layout to work with window resizing
        main_layout.setStretch(0, 1)  # Make the main layout element expand
        
        # Ensure proper resizing behavior for all widgets
        self.adjustSizePolicy(left_panel)
        self.adjustSizePolicy(right_panel)
        
        # Force initial adjustment based on current size
        QTimer.singleShot(100, self.adjust_for_screen_size)
    
    def adjustSizePolicy(self, widget):
        """Helper method to set appropriate size policies for responsive layout"""
        # Set expanding policies for container widgets
        if isinstance(widget, QWidget) and widget.layout() is not None:
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            
            # Apply to all children that are containers
            for i in range(widget.layout().count()):
                item = widget.layout().itemAt(i)
                if item.widget() is not None:
                    child = item.widget()
                    if isinstance(child, (QGroupBox, QFrame, QWidget)) and child.layout() is not None:
                        child.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                        self.adjustSizePolicy(child)
                        
    def showEvent(self, event):
        """Called when the widget is shown"""
        super().showEvent(event)
        # Adjust layout when first shown
        QTimer.singleShot(100, self.adjust_for_screen_size)

    def prepare_for_interview(self):
        """Prepare the interface for an interview"""
        # Reset interview state
        self.current_question_idx = 0
        self.interview_results = []
        self.questions = []
        
        # Generate questions based on CV analysis
        if hasattr(self.main_app, 'cv_tab') and hasattr(self.main_app.cv_tab, 'analysis_results') and self.main_app.cv_tab.analysis_results:
            self.questions = self.backend.generate_interview_questions(self.main_app.cv_tab.analysis_results)
        else:
            # Use mock questions if no CV analysis available
            self.questions = self.mock_questions.copy()
            
        # Ensure we have at least some questions
        if not self.questions:
            self.questions = self.mock_questions.copy()
            
        # Update the questions panel
        if hasattr(self, 'questions_list'):
            self.questions_list.clear()
            for i, question in enumerate(self.questions):
                self.questions_list.addItem(f"Q{i+1}: {question}")
            
        # Update status
        if hasattr(self, 'status_label'):
            self.update_status(f"Ready to start interview with {len(self.questions)} questions")
        
        # Print questions for debugging
        print(f"Prepared {len(self.questions)} interview questions")

    def start_interview(self):
        """Start the interview process"""
        if self.interview_in_progress:
            return
            
        # Ensure we have questions
        if not self.questions:
            self.prepare_for_interview()
            
        # Update UI state
        self.interview_in_progress = True
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.mode_combo.setEnabled(False)
        self.voice_questions_checkbox.setEnabled(False)
        
        # Always use voice questions
        self.voice_only_questions = True
        
        # Store start time and calculate end time
        self.interview_start_time = datetime.now()
        self.interview_end_time = self.interview_start_time + timedelta(minutes=self.interview_duration_minutes)
        
        # Start the interview timer (checks every 10 seconds)
        self.interview_timer.start(10000)
        
        # Initialize results array
        self.interview_results = []
        
        # Reset progress
        self.current_question_idx = 0
        self.progress_bar.setValue(0)
        
        # Start video and audio capture
        self.start_video()
        self.start_audio()
        
        # Initialize voice question player
        self.voice_player = VoiceQuestionPlayer(self)
        self.voice_player.question_completed.connect(self.on_question_spoken)
        
        # Update status
        self.update_status("Interview started")
        
        # Show the first question
        self.show_next_question()
        
        # If the parent app has an interview_completed method, connect to it
        if hasattr(self.main_app, 'interview_completed'):
            self.interview_completed_callback = self.main_app.interview_completed

    def stop_interview(self):
        """Stop the interview process"""
        if not self.interview_in_progress:
            return
            
        # Ask for confirmation
        reply = QMessageBox.question(
            self, 
            'End Interview',
            'Do you want to end this interview session?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.No:
            return
            
        # Stop video and audio threads
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            
        if self.audio_thread and self.audio_thread.isRunning():
            self.audio_thread.stop()
            
        # Stop voice player if running
        if self.voice_player and self.voice_player.isRunning():
            self.voice_player.stop()
            
        # Stop timers
        self.countdown_timer.stop()
        self.response_timer.stop()
        self.interview_timer.stop()
        
        # Update UI state
        self.interview_in_progress = False
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.mode_combo.setEnabled(True)
        self.voice_questions_checkbox.setEnabled(True)
        
        # Reset video
        if hasattr(self, 'video_label'):
            self.video_label.clear()
            
        # Update status
        self.update_status("Interview ended")
        
        # Display results if we have any
        if self.interview_results:
            self.display_results()
            
        # Call the parent app's interview_completed method if available
        if hasattr(self, 'interview_completed_callback'):
            self.interview_completed_callback()

    def check_interview_duration(self):
        """Check if the interview has reached its time limit"""
        if not self.interview_in_progress:
            return
            
        now = datetime.now()
        
        # Calculate and update remaining time
        remaining = self.interview_end_time - now if now < self.interview_end_time else timedelta(0)
        remaining_minutes = int(remaining.total_seconds() / 60)
        remaining_seconds = int(remaining.total_seconds() % 60)
        
        # Update the status with remaining time
        self.time_label.setText(f"Time remaining: {remaining_minutes:02d}:{remaining_seconds:02d}")
        
        # If time is up, stop the interview
        if now >= self.interview_end_time:
            QMessageBox.information(
                self,
                "Interview Time Limit",
                f"The {self.interview_duration_minutes}-minute interview time limit has been reached.",
                QMessageBox.Ok
            )
            self.stop_interview()

    def start_video(self):
        """Start the video capture thread"""
        self.video_thread = VideoThread(self)
        self.video_thread.update_frame.connect(self.update_video_frame)
        self.video_thread.start()

    def start_audio(self):
        """Start the audio recording thread"""
        self.audio_thread = AudioRecordThread(self)
        self.audio_thread.update_audio_level.connect(self.update_audio_level)
        self.audio_thread.recording_complete.connect(self.process_audio)
        self.audio_thread.start()

    @pyqtSlot(QImage)
    def update_video_frame(self, image):
        """Update the video preview with the current frame"""
        if not self.interview_in_progress:
            return
            
        # Ensure the video label is visible and added to the layout
        if self.no_video_label.isVisible():
            video_container_layout = self.video_container.layout()
            video_container_layout.removeWidget(self.no_video_label)
            self.no_video_label.hide()
            video_container_layout.addWidget(self.video_label)
            self.video_label.show()
            
        # Scale image to fit the label
        scaled_image = image.scaled(self.video_label.size(), Qt.KeepAspectRatio)
        self.video_label.setPixmap(QPixmap.fromImage(scaled_image))

    @pyqtSlot(int)
    def update_audio_level(self, level):
        """Update the audio level indicator"""
        self.audio_level.setValue(level)

    def update_countdown(self):
        """Update the countdown timer"""
        self.countdown_seconds -= 1
        if self.countdown_seconds <= 0:
            self.countdown_timer.stop()
            self.countdown_label.setText("Recording starting...")
            self.start_recording()
        else:
            self.countdown_label.setText(f"Recording starts in {self.countdown_seconds} seconds...")

    def update_response_time(self):
        """Update the response timer"""
        self.response_seconds += 1
        elapsed_time = time.strftime('%M:%S', time.gmtime(self.response_seconds))
        self.time_label.setText(f"Time elapsed: {elapsed_time}")
        
        # Give real-time feedback every 5 seconds
        if self.response_seconds % 5 == 0:
            self.update_realtime_feedback()

    def show_next_question(self):
        """Show the next interview question"""
        # Safety check - ensure we're in a valid interview state
        if not self.interview_in_progress:
            self.update_status("Cannot show question - no active interview")
            return
            
        if self.current_question_idx >= len(self.questions):
            # All questions have been asked
            self.update_status("All questions completed")
            self.stop_interview()
            return
            
        # Get the current question
        question = self.questions[self.current_question_idx]
        
        # Update question display
        self.question_label.setText(question)
        
        # Highlight current question in the list
        if self.questions_list.isVisible():
            self.questions_list.setCurrentRow(self.current_question_idx)
        
        # Update status
        self.update_status(f"Question {self.current_question_idx + 1} of {len(self.questions)}")
        
        # Reset countdown just for visual indication but don't actually show it
        self.countdown_seconds = 3
        # Don't start the countdown timer as we're using voice instead
        
        # Reset response data
        self.response_text.clear()
        self.response_seconds = 0
        
        # Reset listening state
        self.is_listening_for_response = False
        
        # Try to speak the question safely
        self.speak_question_safely(question)

    def speak_question_safely(self, question):
        """Speak the question using either TTS or simulation"""
        # Create a TTS player and run it on a dedicated thread
        self.tts_player = TTSPlayer(self)
        self.tts_player.set_text(question)
        
        # Connect signals
        self.tts_player.status_update.connect(self.update_status)
        self.tts_player.finished.connect(self.on_question_spoken)
        
        # Create a worker thread to run the TTS player
        self.tts_thread = QThread()
        self.tts_player.moveToThread(self.tts_thread)
        
        # Connect thread events
        self.tts_thread.started.connect(self.tts_player.play)
        self.tts_player.finished.connect(self.tts_thread.quit)
        self.tts_player.finished.connect(self.tts_player.deleteLater)
        self.tts_thread.finished.connect(self.tts_thread.deleteLater)
        
        # Start the thread
        self.update_status("Starting AI voice...")
        self.tts_thread.start()
        
        return True

    def on_question_spoken(self):
        """Called when the voice question has been completely spoken"""
        # This is where we would start listening for the response
        if not self.is_listening_for_response:
            self.is_listening_for_response = True
            self.start_recording()

    def start_recording(self):
        """Start recording the response"""
        # Safety check - ensure we're in a valid interview state
        if not self.interview_in_progress or not self.questions or self.current_question_idx >= len(self.questions):
            self.update_status("Cannot start recording - no active question")
            return
            
        # Update UI
        self.record_button.setText("STOP RECORDING")
        self.record_button.setStyleSheet("""
            QPushButton {
                background-color: #DC3545;
                color: white;
                padding: 8px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #C82333;
            }
        """)
        self.record_button.clicked.disconnect()
        self.record_button.clicked.connect(self.stop_recording)
        
        # Start the response timer
        self.response_timer.start(1000)
        
        # Start recording audio
        if self.audio_thread:
            temp_audio_file = os.path.join("temp", f"interview_audio_{int(time.time())}.wav")
            self.audio_thread.output_file = temp_audio_file

    def stop_recording(self):
        """Stop recording the response"""
        # Safety check - ensure we're in a valid interview state
        if not self.interview_in_progress:
            self.update_status("Cannot stop recording - no active interview")
            return
            
        # Stop the response timer
        self.response_timer.stop()
        
        # Update UI
        self.record_button.setText("RECORD ANSWER")
        self.record_button.setStyleSheet("""
            QPushButton {
                background-color: #28A745;
                color: white;
                padding: 8px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        self.record_button.clicked.disconnect()
        self.record_button.clicked.connect(self.start_recording)
        
        # Process the current response
        self.process_response()

    def process_response(self):
        """Process the current interview response"""
        # Ensure we have a valid question index
        if not self.questions or self.current_question_idx >= len(self.questions):
            self.update_status("No valid question to process response for")
            return
            
        # Get the current question
        question = self.questions[self.current_question_idx]
        
        # Get response text (in a real app, this might come from speech-to-text)
        response_text = self.response_text.toPlainText()
        
        # Get a reference frame for emotion analysis
        frame = None
        if self.video_thread and self.video_thread.isRunning():
            # In a real app, we'd capture a frame from the video thread
            # For the mock, we'll just create a placeholder
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Analyze the response
        analysis = self.backend.analyze_interview_response(
            question=question,
            response_text=response_text,
            video_frame=frame
        )
        
        # Save this question and response
        self.interview_results.append({
            "question": question,
            "answer": response_text,
            "analysis": analysis,
            "response_time": self.response_seconds
        })
        
        # Display feedback
        self.display_feedback(analysis)
        
        # Update progress
        self.current_question_idx += 1
        self.update_progress()
        
        # Show continue button
        self.continue_button.show()
        self.continue_button.setFocus()

    def process_audio(self, audio_file):
        """Process the recorded audio file"""
        # In a real app, this would use speech-to-text
        # For now, we just print the file path
        print(f"Audio recording saved to: {audio_file}")
        
        # In a real app, we would transcribe the audio to text
        # and update the response_text field
        # For the mock, we'll just leave it to manual entry

    def update_realtime_feedback(self):
        """Update real-time feedback during response"""
        if not self.interview_in_progress or not self.is_listening_for_response:
            return
            
        # In a real app, we might analyze facial expressions, tone, etc.
        # For now, we'll just simulate some feedback
        
        # Create mock emotions
        emotions = {
            "neutral": np.random.uniform(0.3, 0.7),
            "happy": np.random.uniform(0, 0.3),
            "surprised": np.random.uniform(0, 0.2),
            "angry": np.random.uniform(0, 0.1)
        }
        
        # Determine primary emotion
        primary = max(emotions.items(), key=lambda x: x[1])
        
        # Update emotion indicators
        self.update_emotion_indicators(primary[0], emotions)

    def update_emotion_indicators(self, primary_emotion, emotion_scores):
        """Update the emotion indicators in the UI"""
        # Update UI with emotion data
        self.emotion_label.setText(f"Current emotion: {primary_emotion.capitalize()}")
        
        # In a full implementation, we might have visual indicators for each emotion

    def display_feedback(self, feedback):
        """Display feedback for the current response"""
        # Create feedback text
        feedback_text = "<h3>Response Analysis</h3>"
        
        if "text_analysis" in feedback:
            scores = feedback["text_analysis"]
            feedback_text += "<h4>Content Analysis</h4>"
            feedback_text += f"<p>Relevance: {int(scores.get('relevance_score', 0) * 100)}%</p>"
            feedback_text += f"<p>Clarity: {int(scores.get('clarity_score', 0) * 100)}%</p>"
            feedback_text += f"<p>Depth: {int(scores.get('depth_score', 0) * 100)}%</p>"
            feedback_text += f"<p>Structure: {int(scores.get('structure_score', 0) * 100)}%</p>"
            
        if "video_analysis" in feedback:
            emotion = feedback["video_analysis"].get("primary_emotion", "neutral")
            feedback_text += "<h4>Visual Analysis</h4>"
            feedback_text += f"<p>Primary Emotion: {emotion.capitalize()}</p>"
            feedback_text += f"<p>Confidence: {int(feedback['video_analysis'].get('confidence_score', 0) * 100)}%</p>"
            
        if "improvement_tips" in feedback:
            feedback_text += "<h4>Improvement Tips</h4><ul>"
            for tip in feedback["improvement_tips"]:
                feedback_text += f"<li>{tip}</li>"
            feedback_text += "</ul>"
            
        # Display in the feedback panel
        self.feedback_text.setHtml(feedback_text)

    def update_progress(self):
        """Update the interview progress bar"""
        if len(self.questions) > 0:
            progress = int((self.current_question_idx / len(self.questions)) * 100)
            self.progress_bar.setValue(progress)
            self.progress_percentage.setText(f"{progress}%")

    def display_results(self):
        """Display final interview results"""
        # Convert the results to the format expected by the backend
        formatted_results = {
            "questions": self.interview_results,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration_minutes": self.interview_duration_minutes
        }
        
        # Generate final report
        report = self.backend.generate_final_report(formatted_results)
        
        # Create HTML content for report
        html_content = """
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #2C3E50; }
                h2 { color: #3498DB; }
                .section { margin-bottom: 20px; }
                .score { font-weight: bold; color: #27AE60; }
                .low-score { font-weight: bold; color: #E74C3C; }
                .medium-score { font-weight: bold; color: #F39C12; }
                .recommendation { margin: 5px 0; padding: 5px; background-color: #F8F9F9; }
                .question { margin-top: 15px; font-weight: bold; color: #2C3E50; }
                .answer { margin: 5px 0; padding: 5px; background-color: #EBF5FB; }
            </style>
        </head>
        <body>
            <h1>Interview Analysis Report</h1>
            <div class="section">
                <h2>Overall Performance</h2>
                <p>Overall Score: <span class="score">{overall_score}%</span></p>
                <p>Relevance: <span class="{relevance_class}">{relevance_score}%</span></p>
                <p>Clarity: <span class="{clarity_class}">{clarity_score}%</span></p>
                <p>Depth: <span class="{depth_class}">{depth_score}%</span></p>
                <p>Confidence: <span class="{confidence_class}">{confidence_score}%</span></p>
            </div>
            
            <div class="section">
                <h2>Summary</h2>
                <p>{summary}</p>
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                {recommendations}
            </div>
            
            <div class="section">
                <h2>Detailed Question Analysis</h2>
                {detailed_analysis}
            </div>
            
            <p>Report generated on: {timestamp}</p>
        </body>
        </html>
        """
        
        # Format scores and determine classes
        scores = report.get("scores", {})
        overall_score = int(scores.get("overall_score", 0) * 100)
        relevance_score = int(scores.get("relevance_score", 0) * 100)
        clarity_score = int(scores.get("clarity_score", 0) * 100)
        depth_score = int(scores.get("depth_score", 0) * 100)
        confidence_score = int(scores.get("confidence_score", 0) * 100)
        
        def get_score_class(score):
            if score >= 75:
                return "score"
            elif score >= 50:
                return "medium-score"
            else:
                return "low-score"
                
        # Format recommendations
        recommendations_html = "<ul>"
        for rec in report.get("recommendations", []):
            recommendations_html += f"<li class='recommendation'>{rec}</li>"
        recommendations_html += "</ul>"
        
        # Format detailed analysis
        detailed_html = ""
        for i, q_data in enumerate(report.get("questions_analysis", [])):
            question = q_data.get("question", "")
            answer = q_data.get("answer", "")
            
            # Extract scores
            q_scores = {}
            if "analysis" in q_data and "text_analysis" in q_data["analysis"]:
                q_scores = q_data["analysis"]["text_analysis"]
            
            detailed_html += f"<div class='question'>Question {i+1}: {question}</div>"
            detailed_html += f"<div class='answer'>{answer}</div>"
            
            if q_scores:
                detailed_html += "<ul>"
                for key, value in q_scores.items():
                    if key.endswith("_score"):
                        score_name = key.replace("_score", "").capitalize()
                        score_value = int(value * 100)
                        detailed_html += f"<li>{score_name}: {score_value}%</li>"
                detailed_html += "</ul>"
        
        # Fill the template
        formatted_html = html_content.format(
            overall_score=overall_score,
            relevance_score=relevance_score,
            clarity_score=clarity_score,
            depth_score=depth_score,
            confidence_score=confidence_score,
            relevance_class=get_score_class(relevance_score),
            clarity_class=get_score_class(clarity_score),
            depth_class=get_score_class(depth_score),
            confidence_class=get_score_class(confidence_score),
            summary=report.get("summary", "No summary available."),
            recommendations=recommendations_html,
            detailed_analysis=detailed_html,
            timestamp=report.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        
        # Display the report
        self.results_text.setHtml(formatted_html)
        
        # Show the results panel
        self.feedback_panel.setCurrentIndex(1)
        
        # Save results to file
        self.save_results()

    def save_results(self):
        """Save interview results to a file"""
        # Generate a filename based on the date/time
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"interview_results_{timestamp}.json"
        filepath = os.path.join("results", filename)
        
        # Ensure the directory exists
        os.makedirs("results", exist_ok=True)
        
        # Save the results
        with open(filepath, "w") as f:
            import json
            json.dump(self.interview_results, f, indent=4)
            
        # Show confirmation
        self.update_status(f"Results saved to {filepath}")

    def update_status(self, message):
        """Update the status label"""
        self.status_label.seText(message)