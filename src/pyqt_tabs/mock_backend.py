import os
import time
import random
import json

class MockAnalyzerBackend:
    """Mock implementation of the analyzer backend for UI testing"""
    
    def __init__(self):
        """Initialize the mock backend"""
        self.mock_responses = self._load_mock_responses()
    
    def _load_mock_responses(self):
        """Load mock responses or generate if not available"""
        mock_data = {
            "cv_analysis": {
                "candidate_info": {
                    "name": "John Smith",
                    "email": "john.smith@example.com",
                    "phone": "+1 (555) 123-4567"
                },
                "skills": [
                    {"name": "Python", "type": "technical"},
                    {"name": "Machine Learning", "type": "technical"},
                    {"name": "Data Analysis", "type": "technical"},
                    {"name": "SQL", "type": "technical"},
                    {"name": "JavaScript", "type": "technical"},
                    {"name": "Communication", "type": "soft"},
                    {"name": "Problem Solving", "type": "soft"},
                    {"name": "Team Leadership", "type": "soft"}
                ],
                "career_matches": [
                    {"career": "Data Scientist", "match_percentage": 92},
                    {"career": "Machine Learning Engineer", "match_percentage": 87},
                    {"career": "Software Engineer", "match_percentage": 78},
                    {"career": "Business Analyst", "match_percentage": 65}
                ],
                "recommendations": [
                    {
                        "type": "career_options",
                        "options": [
                            {"career": "Data Scientist", "match_percentage": 92},
                            {"career": "Machine Learning Engineer", "match_percentage": 87}
                        ]
                    },
                    {
                        "type": "skills_to_develop",
                        "skills": [
                            "Cloud Computing (AWS/Azure)",
                            "Deep Learning",
                            "Statistical Analysis",
                            "Public Speaking"
                        ]
                    }
                ]
            }
        }
        return mock_data
    
    def extract_text_from_file(self, file_path):
        """Mock method to extract text from a file"""
        # Return a mock CV text
        if file_path.lower().endswith(('.pdf', '.docx', '.doc', '.txt')):
            return """
JOHN SMITH
-----------
john.smith@example.com | +1 (555) 123-4567 | LinkedIn: /in/johnsmith

SUMMARY
-------
Experienced data scientist with 5+ years of experience in machine learning, 
statistical analysis, and data visualization. Skilled in Python, SQL, and various 
ML frameworks. Passionate about solving complex problems and deriving actionable 
insights from data.

SKILLS
------
Technical: Python, R, SQL, TensorFlow, PyTorch, Scikit-learn, Pandas, NumPy, Matplotlib
Soft: Communication, Problem Solving, Team Leadership, Critical Thinking

EXPERIENCE
----------
SENIOR DATA SCIENTIST | ABC TECHNOLOGIES | 2020 - Present
- Led a team of 3 data scientists to develop and deploy machine learning models
- Reduced customer churn by 15% through predictive analytics
- Implemented NLP solutions for automatic text classification
- Created interactive dashboards for business stakeholders

DATA SCIENTIST | XYZ CORP | 2018 - 2020
- Developed predictive models for sales forecasting
- Created ETL pipelines for data processing
- Conducted A/B tests to optimize website conversion

EDUCATION
---------
Master's in Data Science | University of Technology | 2018
Bachelor's in Computer Science | State University | 2016
"""
        else:
            return "Could not extract text from the selected file format."
    
    def analyze_cv(self, file_path):
        """Mock method to analyze a CV file"""
        # Simulate processing time
        time.sleep(2)
        
        # Return mock analysis results
        return self.mock_responses["cv_analysis"]
    
    def generate_interview_questions(self, cv_analysis=None):
        """Generate mock interview questions based on CV analysis"""
        # If we have CV analysis, we could customize questions, but for mock we'll return standard ones
        return [
            "Tell me about yourself and your background.",
            "What motivated you to apply for this position?",
            "Describe a challenging project you worked on and how you handled it.",
            "How do you keep up with the latest trends in your field?",
            "Tell me about a time when you had to learn a new skill quickly.",
            "How do you handle working under pressure or tight deadlines?",
            "Give an example of a time you made a mistake and how you handled it.",
            "Where do you see yourself professionally in five years?"
        ]
    
    def analyze_interview_response(self, question, audio_file=None, video_frame=None):
        """Mock method to analyze an interview response"""
        # Simulate processing time
        time.sleep(1)
        
        # Generate random scores for the response
        confidence_score = random.randint(65, 95)
        clarity_score = random.randint(70, 95)
        relevance_score = random.randint(75, 98)
        
        # Mock feedback based on the question
        improvement_tips = []
        
        if "yourself" in question.lower():
            improvement_tips = [
                "Keep your introduction concise and focused on professional achievements",
                "Include specific accomplishments rather than general statements",
                "Consider structuring your response chronologically for clarity"
            ]
        elif "challenging" in question.lower() or "mistake" in question.lower():
            improvement_tips = [
                "Use the STAR method (Situation, Task, Action, Result) for clarity",
                "Focus more on what you learned from the experience",
                "Be more specific about the actions you took to resolve the situation"
            ]
        elif "pressure" in question.lower() or "deadline" in question.lower():
            improvement_tips = [
                "Provide a more concrete example with specific steps you took",
                "Explain your prioritization strategy more clearly",
                "Include how you communicated with stakeholders during the process"
            ]
        else:
            improvement_tips = [
                "Provide more specific examples to illustrate your points",
                "Try to maintain more consistent eye contact",
                "Consider structuring your response with a clear beginning, middle, and end"
            ]
        
        # Return mock analysis
        return {
            "question": question,
            "scores": {
                "confidence": confidence_score,
                "clarity": clarity_score,
                "relevance": relevance_score
            },
            "improvement_tips": improvement_tips,
            "transcript": "This is a mock transcript of what the person said in response to the question."
        }
    
    def generate_final_report(self, interview_results):
        """Generate a final report from the interview results"""
        # In a real implementation, this would create a comprehensive analysis
        # For mock, we'll just add some overall recommendations
        
        # Calculate overall score
        total_confidence = 0
        total_clarity = 0
        total_relevance = 0
        
        for q in interview_results.get("questions", []):
            scores = q.get("scores", {})
            total_confidence += scores.get("confidence", 0)
            total_clarity += scores.get("clarity", 0)
            total_relevance += scores.get("relevance", 0)
        
        num_questions = len(interview_results.get("questions", []))
        if num_questions > 0:
            avg_confidence = total_confidence / num_questions
            avg_clarity = total_clarity / num_questions
            avg_relevance = total_relevance / num_questions
            overall_score = (avg_confidence + avg_clarity + avg_relevance) / 3
        else:
            overall_score = 0
        
        # Mock recommendations
        recommendations = [
            "Focus on providing more specific examples from your experience",
            "Practice maintaining consistent eye contact during interviews",
            "Work on reducing filler words to sound more confident",
            "Consider preparing more structured responses with clear beginning, middle, and end",
            "Try to vary your tone and pace to keep the interviewer engaged"
        ]
        
        # Return mock report
        return {
            "overall_score": int(overall_score),
            "recommendations": recommendations,
            "detailed_analysis": interview_results.get("questions", [])
        } 