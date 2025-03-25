import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
import tempfile

# Try to import from tests directory if the module exists there
try:
    # Add parent directory to path
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(parent_dir)
    
    # Try to import from tests directory
    from tests.advanced_interview_analyzer import AdvancedInterviewAnalyzer as OriginalAnalyzer
    ORIGINAL_ANALYZER_AVAILABLE = True
except ImportError:
    # Create a mock if the original analyzer is not available
    ORIGINAL_ANALYZER_AVAILABLE = False

class AdvancedInterviewAnalyzer:
    """
    Adapter for the AdvancedInterviewAnalyzer to standardize the interface
    for integration with the career guidance platform.
    """
    
    def __init__(self, models_dir=None, use_mock=False):
        """
        Initialize the interview analyzer adapter
        
        Args:
            models_dir (str): Directory containing analyzer models
            use_mock (bool): Whether to use mock data instead of real analyzer
        """
        self.models_dir = Path(models_dir) if models_dir else Path("models")
        self.use_mock = use_mock or not ORIGINAL_ANALYZER_AVAILABLE
        
        if not self.use_mock and ORIGINAL_ANALYZER_AVAILABLE:
            # Use the original analyzer
            self.analyzer = OriginalAnalyzer()
        else:
            # Create a mock version
            self.analyzer = None
            print("Warning: Using mock interview analyzer. Real analysis not available.")
    
    def analyze_interview(self, video_path, max_duration=None):
        """
        Analyze an interview video and extract facial, voice, and text insights
        
        Args:
            video_path (str): Path to the video file
            max_duration (int): Maximum duration to analyze in seconds
            
        Returns:
            dict: Analysis results including facial emotions, voice analysis, and sentiment
        """
        if not self.use_mock and ORIGINAL_ANALYZER_AVAILABLE:
            # Call the original analyzer but transform results to our format
            try:
                # The original analyzer might have a different method signature or return format
                original_results = self.analyzer.analyze_video(video_path)
                return self._transform_original_results(original_results)
            except Exception as e:
                print(f"Error using original analyzer: {str(e)}")
                # Fall back to mock data if original analyzer fails
                return self._generate_mock_results(video_path)
        else:
            # Use mock data
            return self._generate_mock_results(video_path)
    
    def _transform_original_results(self, original_results):
        """Transform results from the original analyzer to our standardized format"""
        # Create a standardized result structure
        transformed = {
            "facial_emotions": {},
            "voice_emotions": {},
            "sentiment_analysis": {},
            "problem_solving": {
                "score": 0,
                "strengths": [],
                "areas_for_improvement": []
            }
        }
        
        # Map facial emotions - structure may vary in original analyzer
        if "facial_analysis" in original_results:
            facial = original_results["facial_analysis"]
            if isinstance(facial, dict):
                # Direct mapping if compatible
                transformed["facial_emotions"] = {
                    "happiness": facial.get("happy", 0),
                    "sadness": facial.get("sad", 0),
                    "anger": facial.get("angry", 0),
                    "surprise": facial.get("surprise", 0),
                    "fear": facial.get("fear", 0),
                    "disgust": facial.get("disgust", 0),
                    "neutral": facial.get("neutral", 0)
                }
        
        # Map voice emotions
        if "audio_analysis" in original_results:
            audio = original_results["audio_analysis"]
            if isinstance(audio, dict):
                transformed["voice_emotions"] = {
                    "confidence": audio.get("confidence", 0),
                    "clarity": audio.get("clarity", 0),
                    "nervousness": audio.get("nervousness", 0),
                    "enthusiasm": audio.get("enthusiasm", 0),
                    "pace": audio.get("pace", 0)
                }
        
        # Map sentiment analysis
        if "sentiment" in original_results:
            sentiment = original_results["sentiment"]
            if isinstance(sentiment, dict):
                transformed["sentiment_analysis"] = {
                    "positive": sentiment.get("positive", 0),
                    "negative": sentiment.get("negative", 0),
                    "neutral": sentiment.get("neutral", 0)
                }
        
        # Map problem solving if available
        if "problem_solving" in original_results:
            problem = original_results["problem_solving"]
            if isinstance(problem, dict):
                transformed["problem_solving"] = {
                    "score": problem.get("score", 0),
                    "strengths": problem.get("strengths", []),
                    "areas_for_improvement": problem.get("areas_for_improvement", [])
                }
            elif isinstance(problem, (int, float)):
                transformed["problem_solving"]["score"] = problem
        
        return transformed
    
    def _generate_mock_results(self, video_path):
        """Generate mock analysis results for testing"""
        # Check if video exists
        video_exists = os.path.exists(video_path) if video_path else False
        
        # If video exists, extract a few real metrics
        real_metrics = {}
        if video_exists:
            try:
                # Open video file
                cap = cv2.VideoCapture(video_path)
                
                # Get basic video info
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                
                real_metrics = {
                    "fps": fps,
                    "duration": duration,
                    "frame_count": frame_count
                }
                
                # Release video capture
                cap.release()
            except Exception as e:
                print(f"Error extracting video metrics: {str(e)}")
        
        # Generate mock analysis results
        mock_results = {
            "facial_emotions": {
                "happiness": 0.45,
                "sadness": 0.05,
                "anger": 0.03,
                "surprise": 0.12,
                "fear": 0.02,
                "disgust": 0.01,
                "neutral": 0.32
            },
            "voice_emotions": {
                "confidence": 0.68,
                "clarity": 0.72,
                "nervousness": 0.35,
                "enthusiasm": 0.58,
                "pace": 0.65
            },
            "sentiment_analysis": {
                "positive": 0.62,
                "negative": 0.18,
                "neutral": 0.20
            },
            "problem_solving": {
                "score": 75,
                "strengths": [
                    "Logical reasoning",
                    "Creative thinking"
                ],
                "areas_for_improvement": [
                    "Time management",
                    "Systematic approach"
                ]
            },
            "metadata": {
                "is_mock": True,
                "video_path": video_path,
                "video_exists": video_exists,
                **real_metrics
            }
        }
        
        return mock_results
    
    def get_module_status(self):
        """Return the status of the interview analyzer module"""
        return {
            "available": ORIGINAL_ANALYZER_AVAILABLE,
            "using_mock": self.use_mock,
            "models_directory": str(self.models_dir)
        }


# For testing
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = AdvancedInterviewAnalyzer(use_mock=True)
    
    # Test with a mock video or a sample video if available
    sample_video = None
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Check for sample videos in common locations
    potential_videos = [
        os.path.join(parent_dir, "data", "interviews", "sample.mp4"),
        os.path.join(parent_dir, "data", "videos", "sample.mp4"),
        os.path.join(parent_dir, "tests", "data", "sample.mp4")
    ]
    
    for video_path in potential_videos:
        if os.path.exists(video_path):
            sample_video = video_path
            break
    
    # Create a temp video file if no sample exists
    if not sample_video:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp:
            temp.close()
            sample_video = temp.name
            print(f"Created temporary mock video file: {sample_video}")
    
    # Run analysis
    print(f"Analyzing video: {sample_video}")
    results = analyzer.analyze_interview(sample_video)
    
    # Print results
    print("\nINTERVIEW ANALYSIS RESULTS:")
    print(json.dumps(results, indent=2))
    
    # Clean up temporary file if created
    if "tempfile" in sample_video:
        try:
            os.unlink(sample_video)
            print(f"Removed temporary video file: {sample_video}")
        except:
            pass 