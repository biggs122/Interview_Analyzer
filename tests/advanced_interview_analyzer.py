import cv2
import numpy as np
import threading
import queue
import time
import os
import pyaudio
import wave
import mediapipe as mp
from deepface import DeepFace
import soundfile as sf
from pathlib import Path
import platform
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from PIL import Image, ImageTk
import speech_recognition as sr
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, pipeline
import torch

class AdvancedInterviewAnalyzer:
    def __init__(self):
        # Paths
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.temp_dir = os.path.join(self.base_dir, "temp")
        os.makedirs(self.temp_dir, exist_ok=True)

        # Check if running on macOS
        self.is_macos = platform.system() == 'Darwin'

        # Models paths
        self.models_dir = os.path.join(self.base_dir, "models")
        self.facial_model_dir = os.path.join(self.models_dir, "facial")
        self.audio_model_dir = os.path.join(self.models_dir, "audio")
        self.nlp_model_dir = os.path.join(self.models_dir, "nlp")

        # Video analysis
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Audio analysis
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk = 1024
        self.record_seconds = 5  # Process audio in 5-second chunks for speech recognition

        # Speech recognition
        self.recognizer = sr.Recognizer()

        # Results
        self.facial_emotion = "neutral"
        self.voice_emotion = "neutral"
        self.text_sentiment = "neutral"
        self.confidence_score = 0.0
        self.facial_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        self.transcript_queue = queue.Queue()

        # Text transcript
        self.transcript = ""
        self.full_interview_text = ""

        # Analysis control
        self.running = False
        self.threads = []

        # For saving results
        self.results_dir = os.path.join(self.base_dir, "results")
        os.makedirs(self.results_dir, exist_ok=True)
        self.save_frequency = 5  # Save every 5th frame
        self.frame_count = 0

        # Current frame for display
        self.current_frame = None
        self.frame_for_display = None

        # Load NLP model
        try:
            print("Loading pre-trained sentiment analysis model...")
            self.nlp_model = pipeline("sentiment-analysis")
            self.using_pipeline = True
            print("Pre-trained sentiment analysis model loaded successfully")
        except Exception as e:
            print(f"NLP model loading error: {e}")
            self.nlp_model = None
            self.using_pipeline = False

        # Problem Solving Attributes
        self.problem_duration = 300  # Duration in seconds (e.g., 5 minutes)
        self.problem_remaining = self.problem_duration
        self.problem_running = False
        self.problem_statement = ("Problem: Write a Python function that reverses a given string.\n"
                                  "Example: reverse_string('hello') should return 'olleh'.")
        self.candidate_solution = ""

        # Setup UI
        self.setup_ui()

    def setup_ui(self):
        """Set up the Tkinter UI with a professional design"""
        self.root = tk.Tk()
        self.root.title("Interview Analyzer Pro")
        self.root.configure(background="#F0F2F5")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.geometry("1280x800")
        self.root.minsize(1024, 768)

        # Set custom fonts
        title_font = ("Helvetica", 14, "bold")
        subtitle_font = ("Helvetica", 12, "bold")
        normal_font = ("Helvetica", 11)
        data_font = ("Helvetica", 11, "bold")

        # Style configuration
        style = ttk.Style()
        style.theme_use('clam')  # Use a more modern theme as base

        # Configure colors
        primary_color = "#2C3E50"     # Dark blue for headers
        secondary_color = "#3498DB"   # Light blue for highlights
        bg_color = "#F0F2F5"          # Light grey for background
        accent_color = "#1ABC9C"      # Teal for accents
        text_color = "#34495E"        # Darker grey for text

        # Configure styles
        style.configure("TFrame", background=bg_color)
        style.configure("TLabel", background=bg_color, foreground=text_color, font=normal_font)
        style.configure("TLabelframe", background=bg_color, foreground=text_color)
        style.configure("TLabelframe.Label", background=bg_color, foreground=primary_color, font=subtitle_font)

        # Button styles
        style.configure("TButton", background=primary_color, foreground="white", font=normal_font)
        style.map("TButton",
                  background=[('active', secondary_color), ('pressed', primary_color)],
                  foreground=[('active', 'white'), ('pressed', 'white')])

        # Action button style
        style.configure("Action.TButton", background=accent_color, foreground="white", font=subtitle_font)
        style.map("Action.TButton",
                  background=[('active', "#16A085"), ('pressed', accent_color)],
                  foreground=[('active', 'white'), ('pressed', 'white')])

        # Header style
        style.configure("Header.TLabel", background=primary_color, foreground="white", font=title_font, padding=10)

        # Data label style
        style.configure("Data.TLabel", background=bg_color, foreground=secondary_color, font=data_font)

        # Panel styles
        style.configure("Panel.TFrame", background="white", relief="raised")
        style.configure("PanelHeader.TLabel", background=primary_color, foreground="white", font=subtitle_font, padding=5)

        # Create header
        header_frame = ttk.Frame(self.root, style="TFrame")
        header_frame.pack(fill=tk.X, padx=0, pady=0)

        header_label = ttk.Label(header_frame, text="INTERVIEW ANALYZER PRO", style="Header.TLabel")
        header_label.pack(fill=tk.X, padx=0, pady=0)

        # Main container with left and right panels
        main_container = ttk.Frame(self.root, style="TFrame")
        main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Left panel - Video feed and emotion analysis
        left_panel = ttk.Frame(main_container, style="Panel.TFrame")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Video display
        video_frame = ttk.LabelFrame(left_panel, text="Video Analysis", style="TLabelframe")
        video_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        video_container = ttk.Frame(video_frame, style="TFrame")
        video_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.video_label = ttk.Label(video_container, background="black")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # Emotion analysis panel
        emotion_frame = ttk.LabelFrame(left_panel, text="Real-time Emotion Analysis", style="TLabelframe")
        emotion_frame.pack(fill=tk.X, padx=10, pady=10)

        # Create a grid for emotion indicators
        emotion_grid = ttk.Frame(emotion_frame, style="TFrame")
        emotion_grid.pack(fill=tk.X, padx=10, pady=10)

        # Facial emotion
        ttk.Label(emotion_grid, text="Facial Expression:", style="TLabel").grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        self.facial_emotion_label = ttk.Label(emotion_grid, text="Neutral", style="Data.TLabel")
        self.facial_emotion_label.grid(row=0, column=1, sticky=tk.W, padx=10, pady=5)

        # Voice emotion
        ttk.Label(emotion_grid, text="Voice Emotion:", style="TLabel").grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
        self.voice_emotion_label = ttk.Label(emotion_grid, text="Neutral", style="Data.TLabel")
        self.voice_emotion_label.grid(row=1, column=1, sticky=tk.W, padx=10, pady=5)

        # Text sentiment
        ttk.Label(emotion_grid, text="Text Sentiment:", style="TLabel").grid(row=2, column=0, sticky=tk.W, padx=10, pady=5)
        self.sentiment_label = ttk.Label(emotion_grid, text="Neutral (0.0)", style="Data.TLabel")
        self.sentiment_label.grid(row=2, column=1, sticky=tk.W, padx=10, pady=5)

        # Overall assessment
        ttk.Label(emotion_grid, text="OVERALL ASSESSMENT:", style="TLabel").grid(row=3, column=0, sticky=tk.W, padx=10, pady=5)
        self.assessment_label = ttk.Label(emotion_grid, text="Neutral", style="Data.TLabel")
        self.assessment_label.grid(row=3, column=1, sticky=tk.W, padx=10, pady=5)

        # Right panel - Transcript, controls, and problem solving
        right_panel = ttk.Frame(main_container, style="Panel.TFrame")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Transcript display
        transcript_frame = ttk.LabelFrame(right_panel, text="Interview Transcript", style="TLabelframe")
        transcript_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.transcript_text = scrolledtext.ScrolledText(
            transcript_frame,
            wrap=tk.WORD,
            font=("Courier New", 11),
            background="white",
            foreground=text_color,
            insertbackground=text_color,
            borderwidth=1,
            relief="solid"
        )
        self.transcript_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Problem Solving Frame
        problem_frame = ttk.LabelFrame(right_panel, text="Problem Solving Challenge", style="TLabelframe")
        problem_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Timer label for problem solving
        self.problem_timer_label = ttk.Label(problem_frame, text="Time Remaining: 05:00", style="Data.TLabel")
        self.problem_timer_label.pack(anchor=tk.W, padx=10, pady=5)

        # Problem statement display
        self.problem_statement_label = ttk.Label(problem_frame, text=self.problem_statement, style="TLabel", wraplength=400, justify=tk.LEFT)
        self.problem_statement_label.pack(fill=tk.X, padx=10, pady=5)

        # Candidate solution text widget
        self.solution_text = scrolledtext.ScrolledText(problem_frame, wrap=tk.WORD, height=10, font=("Courier New", 11))
        self.solution_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Buttons for problem solving
        ps_button_frame = ttk.Frame(problem_frame, style="TFrame")
        ps_button_frame.pack(fill=tk.X, padx=10, pady=5)

        self.start_problem_button = ttk.Button(ps_button_frame, text="Start Challenge", style="Action.TButton", command=self.start_problem_solving)
        self.start_problem_button.pack(side=tk.LEFT, padx=5, pady=5, ipadx=10, ipady=5)

        self.submit_solution_button = ttk.Button(ps_button_frame, text="Submit Solution", style="TButton", command=self.evaluate_solution, state=tk.DISABLED)
        self.submit_solution_button.pack(side=tk.LEFT, padx=5, pady=5, ipadx=10, ipady=5)

        # Controls frame (for transcript and timer)
        control_frame = ttk.Frame(right_panel, style="TFrame")
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        self.time_label = ttk.Label(control_frame, text="00:00:00", style="Data.TLabel")
        self.time_label.pack(side=tk.LEFT, padx=10)

        button_frame = ttk.Frame(control_frame, style="TFrame")
        button_frame.pack(side=tk.RIGHT, padx=10)

        self.start_button = ttk.Button(button_frame, text="Start Analysis", style="Action.TButton", command=self.start)
        self.start_button.pack(side=tk.LEFT, padx=5, pady=5, ipadx=10, ipady=5)

        self.stop_button = ttk.Button(button_frame, text="Stop Analysis", style="TButton", command=self.stop, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5, pady=5, ipadx=10, ipady=5)

        self.save_button = ttk.Button(button_frame, text="Save Report", style="TButton", command=self.save_full_report)
        self.save_button.pack(side=tk.LEFT, padx=5, pady=5, ipadx=10, ipady=5)

        # Status bar
        status_frame = ttk.Frame(self.root, style="TFrame", relief="sunken", borderwidth=1)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_label = ttk.Label(status_frame, text="Ready to start analysis", style="TLabel", anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=5)

        version_label = ttk.Label(status_frame, text="v1.0.0", style="TLabel")
        version_label.pack(side=tk.RIGHT, padx=10, pady=5)

        # Initialize the timer
        self.recording_time = 0
        self.update_timer()

        # Update UI loop
        self.update_ui()

    def on_close(self):
        """Handle window close event"""
        self.running = False
        self.problem_running = False
        time.sleep(0.5)  # Give threads time to clean up
        self.root.destroy()

    def update_ui(self):
        """Update UI elements"""
        if self.frame_for_display is not None:
            try:
                display_width = 640
                display_height = 480
                resized = cv2.resize(self.frame_for_display, (display_width, display_height))
                img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
            except Exception as e:
                print(f"Error updating video: {e}")
            self.frame_for_display = None

        emotion_colors = {
            "happy": "#27AE60",
            "sad": "#3498DB",
            "angry": "#E74C3C",
            "fearful": "#9B59B6",
            "surprised": "#F39C12",
            "neutral": "#7F8C8D",
            "calm": "#2980B9"
        }

        facial_color = emotion_colors.get(self.facial_emotion.lower(), "#7F8C8D")
        voice_color = emotion_colors.get(self.voice_emotion.lower(), "#7F8C8D")
        text_color = emotion_colors.get(self.text_sentiment.lower(), "#7F8C8D")

        self.facial_emotion_label.configure(
            text=self.facial_emotion.capitalize(),
            foreground=facial_color
        )
        self.voice_emotion_label.configure(
            text=self.voice_emotion.capitalize(),
            foreground=voice_color
        )
        self.sentiment_label.configure(
            text=f"{self.text_sentiment.capitalize()} ({self.confidence_score:.2f})",
            foreground=text_color
        )
        emotions = [self.facial_emotion, self.voice_emotion, self.text_sentiment]
        most_common = max(set(emotions), key=emotions.count)
        assessment_color = emotion_colors.get(most_common.lower(), "#7F8C8D")
        self.assessment_label.configure(
            text=most_common.capitalize(),
            foreground=assessment_color
        )

        self.root.after(33, self.update_ui)  # ~30 FPS

    def start(self):
        """Start the real-time analysis"""
        self.running = True
        self.transcript = ""
        self.full_interview_text = ""
        self.transcript_text.delete(1.0, tk.END)
        self.recording_time = 0

        video_thread = threading.Thread(target=self.video_capture_loop)
        audio_thread = threading.Thread(target=self.audio_capture_loop)
        facial_analysis_thread = threading.Thread(target=self.facial_analysis_loop)
        audio_analysis_thread = threading.Thread(target=self.audio_analysis_loop)
        transcript_analysis_thread = threading.Thread(target=self.transcript_analysis_loop)

        self.threads = [
            video_thread,
            audio_thread,
            facial_analysis_thread,
            audio_analysis_thread,
            transcript_analysis_thread
        ]

        for thread in self.threads:
            thread.daemon = True
            thread.start()

        self.status_label.configure(text="Analysis in progress...")
        self.start_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)

    def stop(self):
        """Stop the real-time analysis"""
        self.running = False
        self.problem_running = False
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=1.0)

        self.status_label.configure(text="Analysis complete. Results saved to: " + self.results_dir)
        self.start_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)

    def video_capture_loop(self):
        """Continuously capture video frames from webcam"""
        cap = cv2.VideoCapture(0)
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                    )

            if self.facial_queue.qsize() < 1:
                self.facial_queue.put(frame)

            cv2.putText(frame, f"Facial: {self.facial_emotion}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Voice: {self.voice_emotion}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Text: {self.text_sentiment}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            self.frame_count += 1
            if self.frame_count % self.save_frequency == 0:
                timestamp = int(time.time())
                output_path = os.path.join(self.results_dir, f"frame_{timestamp}.jpg")
                cv2.imwrite(output_path, frame)

            self.frame_for_display = frame.copy()
            time.sleep(0.01)
        cap.release()

    def facial_analysis_loop(self):
        """Process frames for facial emotion analysis"""
        counter = 0
        while self.running:
            if not self.facial_queue.empty():
                frame = self.facial_queue.get()
                temp_img_path = os.path.join(self.temp_dir, f"temp_frame_{counter}.jpg")
                cv2.imwrite(temp_img_path, frame)
                try:
                    result = DeepFace.analyze(temp_img_path,
                                              actions=['emotion'],
                                              enforce_detection=False,
                                              silent=True)
                    self.facial_emotion = result[0]['dominant_emotion']
                    print(f"Facial: {self.facial_emotion}")
                    timestamp = int(time.time())
                    with open(os.path.join(self.results_dir, f"facial_emotion_{timestamp}.txt"), "w") as f:
                        f.write(f"Timestamp: {timestamp}\n")
                        f.write(f"Dominant emotion: {self.facial_emotion}\n")
                        f.write("All emotions:\n")
                        for emotion, score in result[0]['emotion'].items():
                            f.write(f"  {emotion}: {score:.2f}%\n")
                except Exception as e:
                    print(f"Facial analysis error: {e}")
                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)
                counter += 1
            time.sleep(0.1)

    def audio_capture_loop(self):
        """Continuously capture audio from microphone"""
        audio = pyaudio.PyAudio()
        counter = 0
        while self.running:
            stream = audio.open(format=self.audio_format, channels=self.channels,
                                rate=self.rate, input=True,
                                frames_per_buffer=self.chunk)
            print("Recording audio...")
            frames = []
            for i in range(0, int(self.rate / self.chunk * self.record_seconds)):
                if not self.running:
                    break
                data = stream.read(self.chunk, exception_on_overflow=False)
                frames.append(data)
            stream.stop_stream()
            stream.close()
            if not self.running:
                break
            temp_audio_path = os.path.join(self.temp_dir, f"temp_audio_{counter}.wav")
            wf = wave.open(temp_audio_path, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(audio.get_sample_size(self.audio_format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(frames))
            wf.close()
            self.audio_queue.put(temp_audio_path)
            self.transcribe_audio(temp_audio_path)
            counter += 1
        audio.terminate()

    def transcribe_audio(self, audio_path):
        """Convert speech to text"""
        try:
            with sr.AudioFile(audio_path) as source:
                audio_data = self.recognizer.record(source)
                try:
                    text = self.recognizer.recognize_google(audio_data)
                    if text:
                        print(f"Transcribed: {text}")
                        self.transcript += text + " "
                        self.full_interview_text += text + " "
                        self.transcript_text.delete(1.0, tk.END)
                        self.transcript_text.insert(tk.END, self.full_interview_text)
                        self.transcript_text.see(tk.END)
                        self.transcript_queue.put(text)
                except sr.UnknownValueError:
                    print("Speech Recognition could not understand audio")
                except sr.RequestError as e:
                    print(f"Could not request results from Speech Recognition service; {e}")
        except Exception as e:
            print(f"Transcription error: {e}")

    def audio_analysis_loop(self):
        """Process audio for emotion analysis"""
        while self.running:
            if not self.audio_queue.empty():
                audio_path = self.audio_queue.get()
                try:
                    data, sample_rate = sf.read(audio_path)
                    if len(data) > 0:
                        zcr = np.sum(np.abs(np.diff(np.sign(data)))) / (2 * len(data))
                        energy = np.sum(data**2) / len(data)
                        if zcr > 0.05 and energy > 0.005:
                            self.voice_emotion = "angry"
                        elif zcr > 0.04 and energy > 0.003:
                            self.voice_emotion = "happy"
                        elif zcr < 0.03 and energy < 0.002:
                            self.voice_emotion = "sad"
                        elif zcr < 0.04 and energy < 0.003:
                            self.voice_emotion = "calm"
                        else:
                            self.voice_emotion = "neutral"
                        print(f"Voice: {self.voice_emotion} (ZCR={zcr:.4f}, Energy={energy:.6f})")
                        timestamp = int(time.time())
                        with open(os.path.join(self.results_dir, f"voice_emotion_{timestamp}.txt"), "w") as f:
                            f.write(f"Timestamp: {timestamp}\n")
                            f.write(f"Emotion: {self.voice_emotion}\n")
                            f.write(f"ZCR: {zcr:.6f}\n")
                            f.write(f"Energy: {energy:.6f}\n")
                except Exception as e:
                    print(f"Audio analysis error: {e}")
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            time.sleep(0.1)

    def transcript_analysis_loop(self):
        """Analyze transcript for sentiment"""
        while self.running:
            if not self.transcript_queue.empty():
                text = self.transcript_queue.get()
                try:
                    if self.nlp_model is not None:
                        result = self.nlp_model(text)
                        sentiment = result[0]['label']
                        score = result[0]['score']
                        if sentiment == "POSITIVE":
                            self.text_sentiment = "happy"
                        elif sentiment == "NEGATIVE":
                            self.text_sentiment = "sad"
                        else:
                            self.text_sentiment = "neutral"
                        self.confidence_score = score
                        print(f"Text sentiment: {self.text_sentiment} (confidence: {score:.2f})")
                        timestamp = int(time.time())
                        with open(os.path.join(self.results_dir, f"text_sentiment_{timestamp}.txt"), "w") as f:
                            f.write(f"Timestamp: {timestamp}\n")
                            f.write(f"Text: {text}\n")
                            f.write(f"Sentiment: {self.text_sentiment}\n")
                            f.write(f"Confidence: {score:.4f}\n")
                    else:
                        print("No NLP model available for sentiment analysis")
                except Exception as e:
                    print(f"Text analysis error: {e}")
            time.sleep(0.1)

    def update_timer(self):
        """Update the timer display for overall analysis"""
        if self.running:
            self.recording_time += 1
            hours, remainder = divmod(self.recording_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            self.time_label.configure(text=f"{hours:02}:{minutes:02}:{seconds:02}")
        self.root.after(1000, self.update_timer)

    def start_problem_solving(self):
        """Start the problem solving challenge and timer"""
        # Disable the start button and enable submit button
        self.start_problem_button.configure(state=tk.DISABLED)
        self.submit_solution_button.configure(state=tk.NORMAL)
        self.problem_remaining = self.problem_duration
        self.problem_running = True
        self.update_problem_timer()

    def update_problem_timer(self):
        """Update the problem solving countdown timer"""
        if self.problem_running and self.problem_remaining >= 0:
            minutes, seconds = divmod(self.problem_remaining, 60)
            self.problem_timer_label.configure(text=f"Time Remaining: {minutes:02}:{seconds:02}")
            self.problem_remaining -= 1
            self.root.after(1000, self.update_problem_timer)
        elif self.problem_remaining < 0:
            self.problem_running = False
            self.submit_solution_button.configure(state=tk.DISABLED)
            messagebox.showinfo("Time's up", "The problem solving time has ended.")

    def evaluate_solution(self):
        """Evaluate the candidate's solution (placeholder analysis)"""
        self.candidate_solution = self.solution_text.get(1.0, tk.END).strip()
        if not self.candidate_solution:
            messagebox.showwarning("No Submission", "Please enter your solution before submitting.")
            return
        # A placeholder evaluation: Check if the solution contains a function definition
        if "def" in self.candidate_solution and "reverse" in self.candidate_solution:
            result_text = "Solution appears to include a function definition for reversing a string."
        else:
            result_text = "Solution does not seem to contain the required function definition."
        timestamp = int(time.time())
        report_path = os.path.join(self.results_dir, f"problem_solution_{timestamp}.txt")
        with open(report_path, "w") as f:
            f.write("=== Problem Solving Report ===\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write("Problem Statement:\n")
            f.write(self.problem_statement + "\n\n")
            f.write("Candidate's Solution:\n")
            f.write(self.candidate_solution + "\n\n")
            f.write("Evaluation Result:\n")
            f.write(result_text + "\n")
        messagebox.showinfo("Solution Evaluation", result_text)
        self.status_label.configure(text=f"Problem solution evaluated and saved to: {report_path}")
        # Disable submit after evaluation
        self.submit_solution_button.configure(state=tk.DISABLED)

    def save_full_report(self):
        """Save a complete analysis report"""
        timestamp = int(time.time())
        report_path = os.path.join(self.results_dir, f"interview_report_{timestamp}.txt")
        try:
            with open(report_path, "w") as f:
                f.write("===============================================\n")
                f.write("             INTERVIEW ANALYSIS REPORT          \n")
                f.write("===============================================\n\n")
                f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))}\n")
                f.write(f"Duration: {self.time_label.cget('text')}\n\n")
                f.write("-----------------------------------------------\n")
                f.write("               EMOTIONAL ANALYSIS              \n")
                f.write("-----------------------------------------------\n")
                f.write(f"Facial Expression: {self.facial_emotion}\n")
                f.write(f"Voice Emotion: {self.voice_emotion}\n")
                f.write(f"Text Sentiment: {self.text_sentiment} (confidence: {self.confidence_score:.2f})\n\n")
                emotions = [self.facial_emotion, self.voice_emotion, self.text_sentiment]
                most_common = max(set(emotions), key=emotions.count)
                f.write(f"Overall Assessment: {most_common}\n\n")
                f.write("-----------------------------------------------\n")
                f.write("                INTERVIEW TRANSCRIPT           \n")
                f.write("-----------------------------------------------\n")
                f.write(self.full_interview_text)
            self.status_label.configure(text=f"Report saved to: {report_path}")
            messagebox.showinfo("Report Saved", f"Interview analysis report has been saved to:\n{report_path}")
            print(f"Report saved to: {report_path}")
        except Exception as e:
            print(f"Error saving report: {e}")
            self.status_label.configure(text=f"Error saving report: {e}")
            messagebox.showerror("Error", f"Could not save report: {e}")

def main():
    analyzer = AdvancedInterviewAnalyzer()
    analyzer.root.mainloop()

if __name__ == "__main__":
    main()
