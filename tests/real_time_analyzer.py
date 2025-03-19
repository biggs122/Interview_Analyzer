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
from tkinter import ttk
from PIL import Image, ImageTk

class RealTimeInterviewAnalyzer:
    def __init__(self):
        # Paths
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.temp_dir = os.path.join(self.base_dir, "temp")
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Check if running on macOS
        self.is_macos = platform.system() == 'Darwin'
        
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
        self.record_seconds = 3  # Process audio in 3-second chunks
        
        # Results
        self.facial_emotion = "neutral"
        self.voice_emotion = "neutral"
        self.facial_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        
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
        
        # Setup UI
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the Tkinter UI"""
        self.root = tk.Tk()
        self.root.title("Interview Analyzer")
        self.root.configure(background="black")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(padx=10, pady=10)
        
        # Video display
        self.video_label = ttk.Label(main_frame)
        self.video_label.grid(row=0, column=0, rowspan=6, padx=10, pady=10)
        
        # Analysis results
        results_frame = ttk.LabelFrame(main_frame, text="Analysis Results")
        results_frame.grid(row=0, column=1, padx=10, pady=10, sticky="n")
        
        # Facial emotion
        ttk.Label(results_frame, text="Facial Emotion:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.facial_emotion_label = ttk.Label(results_frame, text="Neutral", font=("Arial", 12, "bold"))
        self.facial_emotion_label.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        # Voice emotion
        ttk.Label(results_frame, text="Voice Emotion:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.voice_emotion_label = ttk.Label(results_frame, text="Neutral", font=("Arial", 12, "bold"))
        self.voice_emotion_label.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
        # ZCR & Energy
        ttk.Label(results_frame, text="ZCR:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.zcr_label = ttk.Label(results_frame, text="0.0000")
        self.zcr_label.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        
        ttk.Label(results_frame, text="Energy:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.energy_label = ttk.Label(results_frame, text="0.000000")
        self.energy_label.grid(row=3, column=1, sticky="w", padx=5, pady=5)
        
        # Status
        self.status_label = ttk.Label(self.root, text="Ready", font=("Arial", 10))
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        
        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=1, column=1, padx=10, pady=10)
        
        self.start_button = ttk.Button(btn_frame, text="Start Analysis", command=self.start)
        self.start_button.grid(row=0, column=0, padx=5, pady=5)
        
        self.stop_button = ttk.Button(btn_frame, text="Stop Analysis", command=self.stop, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, padx=5, pady=5)
        
        # Update UI loop
        self.update_ui()
    
    def on_close(self):
        """Handle window close event"""
        self.running = False
        time.sleep(0.5)  # Give threads time to clean up
        self.root.destroy()
    
    def update_ui(self):
        """Update UI elements"""
        # Update video frame if available
        if self.frame_for_display is not None:
            # Convert to PhotoImage
            img = cv2.cvtColor(self.frame_for_display, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            
            # Update label
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            
            # Clear the reference
            self.frame_for_display = None
        
        # Update emotion labels
        self.facial_emotion_label.configure(text=self.facial_emotion.capitalize())
        self.voice_emotion_label.configure(text=self.voice_emotion.capitalize())
        
        # Schedule the next update
        self.root.after(33, self.update_ui)  # ~30 FPS
    
    def start(self):
        """Start the real-time analysis"""
        self.running = True
        
        # Start video and audio capture threads
        video_thread = threading.Thread(target=self.video_capture_loop)
        audio_thread = threading.Thread(target=self.audio_capture_loop)
        facial_analysis_thread = threading.Thread(target=self.facial_analysis_loop)
        audio_analysis_thread = threading.Thread(target=self.audio_analysis_loop)
        
        self.threads = [video_thread, audio_thread, facial_analysis_thread, audio_analysis_thread]
        
        for thread in self.threads:
            thread.daemon = True
            thread.start()
            
        self.status_label.configure(text="Analysis running. Results are being saved to: " + self.results_dir)
        self.start_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
    
    def stop(self):
        """Stop the real-time analysis"""
        self.running = False
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=1.0)
        
        self.status_label.configure(text="Analysis stopped")
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
                
            # Convert to RGB for mediapipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process landmarks
            results = self.face_mesh.process(rgb_frame)
            
            # Draw landmarks
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                    )
            
            # Put frame in queue for emotion analysis (at lower frequency)
            if self.facial_queue.qsize() < 1:  # Limit queue size
                self.facial_queue.put(frame)
            
            # Display results on the frame
            cv2.putText(frame, f"Facial: {self.facial_emotion}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Voice: {self.voice_emotion}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Save frame periodically
            self.frame_count += 1
            if self.frame_count % self.save_frequency == 0:
                timestamp = int(time.time())
                output_path = os.path.join(self.results_dir, f"frame_{timestamp}.jpg")
                cv2.imwrite(output_path, frame)
            
            # Update frame for UI display
            self.frame_for_display = frame.copy()
            
            # Brief pause to reduce CPU usage
            time.sleep(0.01)
        
        cap.release()
    
    def facial_analysis_loop(self):
        """Process frames for facial emotion analysis"""
        counter = 0
        while self.running:
            if not self.facial_queue.empty():
                frame = self.facial_queue.get()
                
                # Save frame temporarily for DeepFace
                temp_img_path = os.path.join(self.temp_dir, f"temp_frame_{counter}.jpg")
                cv2.imwrite(temp_img_path, frame)
                
                try:
                    # Analyze with DeepFace
                    result = DeepFace.analyze(temp_img_path, 
                                             actions=['emotion'],
                                             enforce_detection=False,
                                             silent=True)
                    self.facial_emotion = result[0]['dominant_emotion']
                    print(f"Facial: {self.facial_emotion}")
                    
                    # Save detailed results
                    timestamp = int(time.time())
                    with open(os.path.join(self.results_dir, f"facial_emotion_{timestamp}.txt"), "w") as f:
                        f.write(f"Timestamp: {timestamp}\n")
                        f.write(f"Dominant emotion: {self.facial_emotion}\n")
                        f.write("All emotions:\n")
                        for emotion, score in result[0]['emotion'].items():
                            f.write(f"  {emotion}: {score:.2f}%\n")
                        
                except Exception as e:
                    print(f"Facial analysis error: {e}")
                
                # Clean up
                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)
                
                counter += 1
            
            # Sleep to reduce CPU usage
            time.sleep(0.1)
    
    def audio_capture_loop(self):
        """Continuously capture audio from microphone"""
        audio = pyaudio.PyAudio()
        counter = 0
        
        while self.running:
            # Start recording
            stream = audio.open(format=self.audio_format, channels=self.channels,
                            rate=self.rate, input=True,
                            frames_per_buffer=self.chunk)
            
            print("Recording audio...")
            frames = []
            
            # Record for a few seconds
            for i in range(0, int(self.rate / self.chunk * self.record_seconds)):
                if not self.running:
                    break
                data = stream.read(self.chunk, exception_on_overflow=False)
                frames.append(data)
            
            # Stop recording
            stream.stop_stream()
            stream.close()
            
            # Skip if stopped during recording
            if not self.running:
                break
                
            # Save audio temporarily
            temp_audio_path = os.path.join(self.temp_dir, f"temp_audio_{counter}.wav")
            wf = wave.open(temp_audio_path, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(audio.get_sample_size(self.audio_format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(frames))
            wf.close()
            
            # Put in queue for analysis
            self.audio_queue.put(temp_audio_path)
            counter += 1
        
        audio.terminate()
    
    def audio_analysis_loop(self):
        """Process audio for emotion analysis"""
        while self.running:
            if not self.audio_queue.empty():
                audio_path = self.audio_queue.get()
                
                try:
                    # Load audio
                    data, sample_rate = sf.read(audio_path)
                    
                    # Simple audio features
                    if len(data) > 0:
                        # Zero crossing rate
                        zcr = np.sum(np.abs(np.diff(np.sign(data)))) / (2 * len(data))
                        
                        # Energy
                        energy = np.sum(data**2) / len(data)
                        
                        # Update UI labels
                        self.zcr_label.configure(text=f"{zcr:.4f}")
                        self.energy_label.configure(text=f"{energy:.6f}")
                        
                        # Simple rule-based classification
                        emotions = ["neutral", "calm", "happy", "sad", "angry", "fearful"]
                        
                        # Simple heuristics based on ZCR and energy
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
                        
                        # Save audio results
                        timestamp = int(time.time())
                        with open(os.path.join(self.results_dir, f"voice_emotion_{timestamp}.txt"), "w") as f:
                            f.write(f"Timestamp: {timestamp}\n")
                            f.write(f"Emotion: {self.voice_emotion}\n")
                            f.write(f"ZCR: {zcr:.6f}\n")
                            f.write(f"Energy: {energy:.6f}\n")
                except Exception as e:
                    print(f"Audio analysis error: {e}")
                
                # Clean up
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            
            # Sleep to reduce CPU usage
            time.sleep(0.1)

def main():
    analyzer = RealTimeInterviewAnalyzer()
    # Start the tkinter main loop
    analyzer.root.mainloop()

if __name__ == "__main__":
    main() 