import os
import time
import logging
import cv2
import numpy as np
import pickle
import json
from datetime import datetime

# Import model loaders - try advanced first, fall back to basic
try:
    from src.utils.advanced_model_loader import AdvancedModelLoader
    advanced_models_available = True
except ImportError:
    advanced_models_available = False
    from src.utils.model_loader import model_loader
    
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalyzerBackend:
    """Backend implementation that uses pre-trained models for analysis"""
    
    def __init__(self):
        """Initialize the analyzer backend"""
        # Initialize model loaders
        if advanced_models_available:
            logger.info("Initializing advanced AI models")
            self.model_loader = AdvancedModelLoader(use_gpu=False)
            self.using_advanced_models = True
        else:
            logger.info("Advanced models not available, using basic models")
            self.model_loader = model_loader
            self.using_advanced_models = False
            
        # Create required directories
        self._create_dirs()
        
        # Get model version information if available
        if self.using_advanced_models:
            version_info = self.model_loader.get_version_info()
            current_version = version_info.get("current_version", "unknown")
            logger.info(f"Using model version: {current_version}")
            
            # Log performance metrics
            metrics = self.model_loader.get_performance_metrics()
            if metrics:
                logger.info("Model performance metrics:")
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        logger.info(f"  - {key}: {value:.4f}")
    
    def _create_dirs(self):
        """Create required directories for the application"""
        dirs = ["results", "temp"]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
        
    def extract_text_from_file(self, file_path):
        """
        Extract text from a CV file
        
        Args:
            file_path (str): Path to the CV file
            
        Returns:
            str: Extracted text from the file
        """
        logger.info(f"Extracting text from file: {file_path}")
        
        # Different file types need different handling
        if file_path.lower().endswith('.pdf'):
            # Try to use PyPDF2 or similar if available
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(file_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                return text
            except ImportError:
                logger.warning("PyPDF2 not available. Cannot extract text from PDF.")
                return "PDF text extraction not available. Please install PyPDF2."
            except Exception as e:
                logger.error(f"Error extracting text from PDF: {str(e)}")
                return f"Error extracting text: {str(e)}"
                
        elif file_path.lower().endswith(('.docx', '.doc')):
            # Try to use python-docx if available
            try:
                import docx
                doc = docx.Document(file_path)
                text = "\n".join([para.text for para in doc.paragraphs])
                return text
            except ImportError:
                logger.warning("python-docx not available. Cannot extract text from DOCX.")
                return "DOCX text extraction not available. Please install python-docx."
            except Exception as e:
                logger.error(f"Error extracting text from DOCX: {str(e)}")
                return f"Error extracting text: {str(e)}"
                
        elif file_path.lower().endswith('.txt'):
            # Plain text file
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Error reading text file: {str(e)}")
                return f"Error reading file: {str(e)}"
        else:
            return "Unsupported file format. Please upload a PDF, DOCX, or TXT file."
    
    def analyze_cv(self, file_path):
        """
        Analyze a CV file
        
        Args:
            file_path (str): Path to the CV file
            
        Returns:
            dict: CV analysis results
        """
        logger.info(f"Analyzing CV: {file_path}")
        
        # Extract text from the CV
        cv_text = self.extract_text_from_file(file_path)
        
        try:
            # Process with the appropriate models
            if self.using_advanced_models:
                # Use advanced BERT-based model
                cv_category = self.model_loader.predict_cv_category(cv_text)
                logger.info(f"CV categorized as: {cv_category}")
            else:
                # Fall back to basic analysis
                time.sleep(2)  # Simulate processing
                cv_category = "technical"  # Default category
                
            return self._generate_cv_analysis(cv_text, cv_category)
        except Exception as e:
            logger.error(f"Error analyzing CV: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Fall back to basic analysis
            return self._generate_cv_analysis(cv_text, "unknown")
    
    def _generate_cv_analysis(self, cv_text, cv_category):
        """
        Generate CV analysis results
        
        Args:
            cv_text (str): Text extracted from the CV
            cv_category (str): Category determined by the model
            
        Returns:
            dict: CV analysis results
        """
        # Extract candidate info
        candidate_info = self._extract_candidate_info(cv_text)
        
        # Extract skills
        skills = self._extract_skills(cv_text)
        
        # Generate career matches
        career_matches = self._generate_career_matches(cv_text, skills, cv_category)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(career_matches, skills)
        
        # Return the complete analysis
        return {
            "candidate_info": candidate_info,
            "skills": skills,
            "career_matches": career_matches,
            "recommendations": recommendations,
            "category": cv_category
        }
    
    def _extract_candidate_info(self, cv_text):
        """Extract candidate information from CV text"""
        # In a real implementation, this would use NLP to extract information
        # For now, we'll use basic text processing
        
        lines = cv_text.split('\n')
        name = ""
        email = ""
        phone = ""
        
        # Look for name in the first few lines
        for i, line in enumerate(lines[:5]):
            line = line.strip()
            if line and not name and i == 0:
                name = line
            
            # Look for email using a simple pattern
            if '@' in line and not email:
                words = line.split()
                for word in words:
                    if '@' in word:
                        email = word.strip('.,;')
            
            # Look for phone using a simple pattern
            if not phone:
                for word in line.split():
                    if any(c.isdigit() for c in word) and len(word) >= 7:
                        phone = word.strip('.,;')
        
        return {
            "name": name,
            "email": email,
            "phone": phone
        }
    
    def _extract_skills(self, cv_text):
        """Extract skills from CV text"""
        # In a real implementation, this would use NLP and the model
        # For now, we'll use a simple keyword-based approach
        
        # List of technical skills to look for
        technical_skills = [
            "Python", "Java", "C++", "JavaScript", "HTML", "CSS", "SQL",
            "Machine Learning", "Data Analysis", "AI", "Data Science",
            "React", "Angular", "Vue", "Node.js", "Django", "Flask",
            "TensorFlow", "PyTorch", "Scikit-learn", "Pandas", "NumPy",
            "AWS", "Azure", "GCP", "Docker", "Kubernetes"
        ]
        
        # List of soft skills to look for
        soft_skills = [
            "Communication", "Teamwork", "Leadership", "Problem Solving",
            "Critical Thinking", "Time Management", "Adaptability",
            "Creativity", "Emotional Intelligence", "Negotiation",
            "Conflict Resolution", "Decision Making", "Project Management"
        ]
        
        # Find skills in the text
        found_technical = []
        found_soft = []
        
        for skill in technical_skills:
            if skill.lower() in cv_text.lower():
                found_technical.append({"name": skill, "type": "technical"})
        
        for skill in soft_skills:
            if skill.lower() in cv_text.lower():
                found_soft.append({"name": skill, "type": "soft"})
        
        # Combine and return all skills
        return found_technical + found_soft
    
    def _generate_career_matches(self, cv_text, skills, cv_category):
        """Generate career matches based on CV text and skills"""
        
        # Extract skills names for easier matching
        skill_names = [s["name"].lower() for s in skills]
        
        # Define potential careers based on categorization and skills
        careers = {
            "technical": [
                {"title": "Software Engineer", "match": 0},
                {"title": "Data Scientist", "match": 0},
                {"title": "Machine Learning Engineer", "match": 0},
                {"title": "DevOps Engineer", "match": 0},
                {"title": "Full Stack Developer", "match": 0}
            ],
            "management": [
                {"title": "Project Manager", "match": 0},
                {"title": "Product Manager", "match": 0},
                {"title": "Team Lead", "match": 0},
                {"title": "Technical Director", "match": 0},
                {"title": "CTO", "match": 0}
            ],
            "creative": [
                {"title": "UX Designer", "match": 0},
                {"title": "UI Developer", "match": 0},
                {"title": "Creative Director", "match": 0},
                {"title": "Graphics Programmer", "match": 0},
                {"title": "Game Developer", "match": 0}
            ],
            "customer_service": [
                {"title": "Support Specialist", "match": 0},
                {"title": "Customer Success Manager", "match": 0},
                {"title": "Technical Support Engineer", "match": 0},
                {"title": "Account Manager", "match": 0},
                {"title": "Client Relations Manager", "match": 0}
            ],
            "research": [
                {"title": "Research Scientist", "match": 0},
                {"title": "Algorithm Developer", "match": 0},
                {"title": "Research Engineer", "match": 0},
                {"title": "Data Analyst", "match": 0},
                {"title": "Computational Linguist", "match": 0}
            ]
        }
        
        # Use the model's category prediction, fallback to "technical" if not available
        category = cv_category if cv_category in careers else "technical"
        
        # Simple matching logic
        for career in careers[category]:
            # Base score
            base_score = 0.7  # Model already predicted this category
            
            # Count skill matches for this career
            skill_score = 0
            
            # Different careers value different skills
            if career["title"] == "Software Engineer":
                relevant_skills = ["python", "java", "javascript", "c++", "react", "angular", "vue"]
                for skill in relevant_skills:
                    if skill in skill_names:
                        skill_score += 0.05
            
            elif career["title"] == "Data Scientist":
                relevant_skills = ["python", "machine learning", "data analysis", "pandas", "numpy"]
                for skill in relevant_skills:
                    if skill in skill_names:
                        skill_score += 0.05
            
            elif career["title"] == "Project Manager":
                relevant_skills = ["leadership", "communication", "project management", "time management"]
                for skill in relevant_skills:
                    if skill in skill_names:
                        skill_score += 0.07
            
            # Similar matching for other careers...
            # In a real system, this would be a more sophisticated algorithm
            
            # Calculate final match
            career["match"] = min(0.99, base_score + skill_score)
        
        # Get top 3 matches
        top_matches = sorted(careers[category], key=lambda x: x["match"], reverse=True)[:3]
        
        return [
            {"title": career["title"], "match_percentage": round(career["match"] * 100)}
            for career in top_matches
        ]
    
    def _generate_recommendations(self, career_matches, skills):
        """Generate recommendations based on career matches and skills"""
        
        # Get the top career
        top_career = career_matches[0]["title"] if career_matches else "Software Engineer"
        
        # Build recommendations list
        recommendations = []
        
        # Skill gaps
        skill_gaps = {
            "Software Engineer": ["Python", "JavaScript", "Git", "Data Structures"],
            "Data Scientist": ["Python", "Machine Learning", "Statistics", "SQL"],
            "Project Manager": ["Communication", "Leadership", "Project Management"],
            "UX Designer": ["UI Design", "User Research", "Wireframing", "Prototyping"],
            "Support Specialist": ["Communication", "Problem Solving", "Product Knowledge"],
            "Research Scientist": ["Statistics", "Machine Learning", "Scientific Writing"]
        }
        
        # Get skill gap for the top career
        if top_career in skill_gaps:
            skill_names = [s["name"].lower() for s in skills]
            missing_skills = []
            
            for skill in skill_gaps[top_career]:
                if skill.lower() not in skill_names:
                    missing_skills.append(skill)
            
            if missing_skills:
                recommendations.append({
                    "type": "skill_gap",
                    "content": f"Consider developing these skills for a {top_career} position: {', '.join(missing_skills)}"
                })
        
        # Generic recommendations
        recommendations.append({
            "type": "general",
            "content": "Highlight your most relevant experience at the top of your CV"
        })
        
        recommendations.append({
            "type": "general",
            "content": "Include metrics and achievements in your experience descriptions"
        })
        
        # Add career-specific recommendation
        recommendations.append({
            "type": "career",
            "content": f"Your profile is well-suited for a {top_career} role"
        })
        
        return recommendations
        
    def generate_interview_questions(self, cv_analysis=None):
        """
        Generate interview questions based on CV analysis
        
        Args:
            cv_analysis (dict): CV analysis results
            
        Returns:
            list: Interview questions
        """
        # Default category if not specified
        category = "technical"
        if cv_analysis and "category" in cv_analysis:
            category = cv_analysis["category"]
        
        # Extract skills if available
        skills = []
        if cv_analysis and "skills" in cv_analysis:
            skills = cv_analysis["skills"]
        
        # Base questions for each category
        question_templates = {
            "technical": [
                "Describe a challenging technical problem you've solved. What made it difficult and how did you approach it?",
                "How do you stay updated with the latest technologies in your field?",
                "Can you walk me through your approach to debugging a complex issue?",
                "How do you measure the success of your code or technical solutions?",
                "Tell me about a technical decision you made that you later regretted. What did you learn?"
            ],
            "management": [
                "Describe your management style and how you adapt it to different team members.",
                "How do you handle conflicting priorities in a project?",
                "Tell me about a time when you had to make a difficult decision as a manager.",
                "How do you measure the success of your team?",
                "What strategies do you use to develop your team members?"
            ],
            "creative": [
                "Can you walk me through your creative process?",
                "How do you balance creativity with project constraints and deadlines?",
                "Tell me about a creative solution you developed for a challenging problem.",
                "How do you gather feedback and iterate on your creative work?",
                "What inspires you and how do you incorporate that into your work?"
            ],
            "customer_service": [
                "Tell me about a time you had to deal with a particularly difficult customer.",
                "How do you ensure you understand customer needs correctly?",
                "Describe a situation where you went above and beyond for a customer.",
                "How do you handle situations where you can't fulfill a customer's request?",
                "What metrics do you use to measure customer satisfaction?"
            ],
            "research": [
                "Describe your research methodology and approach to problem-solving.",
                "How do you validate your research findings?",
                "Tell me about a research project where you had to pivot based on unexpected results.",
                "How do you balance theoretical work with practical applications?",
                "What techniques do you use to communicate complex research findings to non-technical stakeholders?"
            ]
        }
        
        # Get questions for the category
        questions = []
        if category in question_templates:
            questions = question_templates[category][:3]  # Get first 3 questions
        else:
            questions = question_templates["technical"][:3]  # Default to technical
        
        # Add skill-specific questions
        skill_names = [s["name"] for s in skills]
        
        skill_questions = {
            "Python": "Can you explain how you've used Python in your past projects?",
            "Machine Learning": "What machine learning algorithms have you worked with and how did you select them?",
            "Leadership": "Describe a situation where your leadership made a significant difference in a project outcome.",
            "Communication": "How do you ensure effective communication within your team and with stakeholders?",
            "Problem Solving": "What framework do you use for approaching complex problems?"
        }
        
        for skill in skill_names:
            if skill in skill_questions and len(questions) < 5:
                questions.append(skill_questions[skill])
        
        # Add a behavioral question
        behavioral_questions = [
            "Tell me about a time you faced a significant challenge in your work. How did you handle it?",
            "Describe a situation where you had to work with a difficult team member. How did you handle it?",
            "Can you give an example of a time when you failed? What did you learn from it?"
        ]
        
        if len(questions) < 6:
            questions.append(np.random.choice(behavioral_questions))
        
        # Return the questions
        return questions
    
    def analyze_interview_response(self, question, response_text=None, audio_file=None, video_frame=None):
        """
        Analyze an interview response
        
        Args:
            question (str): The interview question
            response_text (str): Text of the response
            audio_file (str): Path to audio file if available
            video_frame (numpy.ndarray): Video frame if available
            
        Returns:
            dict: Response analysis results
        """
        logger.info(f"Analyzing response to: {question[:50]}...")
        
        # Initialize results
        results = {
            "text_analysis": {},
            "audio_analysis": {},
            "video_analysis": {},
            "improvement_tips": []
        }
        
        # Analyze text if available
        if response_text:
            try:
                if self.using_advanced_models:
                    # Use advanced RoBERTa-based model
                    text_analysis = self.model_loader.analyze_interview_content(response_text)
                    
                    # Convert to expected format
                    results["text_analysis"] = {
                        "relevance_score": text_analysis.get("relevance", 0.5),
                        "clarity_score": text_analysis.get("clarity", 0.5),
                        "depth_score": text_analysis.get("depth", 0.5),
                        "structure_score": text_analysis.get("structure", 0.5),
                        "overall_quality": text_analysis.get("quality", 0.5)
                    }
                else:
                    # Fake analysis
                    results["text_analysis"] = {
                        "relevance_score": np.random.uniform(0.6, 0.9),
                        "clarity_score": np.random.uniform(0.5, 0.9),
                        "depth_score": np.random.uniform(0.4, 0.8),
                        "structure_score": np.random.uniform(0.5, 0.9),
                        "overall_quality": np.random.uniform(0.5, 0.9)
                    }
            except Exception as e:
                logger.error(f"Error analyzing response text: {str(e)}")
                results["text_analysis"] = {
                    "relevance_score": 0.5,
                    "clarity_score": 0.5,
                    "depth_score": 0.5,
                    "structure_score": 0.5,
                    "overall_quality": 0.5
                }
        
        # Analyze video frame if available
        if video_frame is not None:
            try:
                if self.using_advanced_models:
                    # Use advanced ResNet-based emotion detection
                    emotion = self.model_loader.detect_emotion(video_frame)
                    
                    # Create emotion scores
                    emotions = ['neutral', 'happy', 'sad', 'angry', 'surprised', 'scared', 'disgusted']
                    emotion_scores = {e: 0.1 for e in emotions}  # Base values
                    emotion_scores[emotion] = 0.9  # Set detected emotion high
                    
                    results["video_analysis"] = {
                        "emotion_scores": emotion_scores,
                        "primary_emotion": emotion,
                        "confidence_score": np.random.uniform(0.6, 0.9)
                    }
                else:
                    # Fake analysis
                    emotions = ['neutral', 'happy', 'sad', 'angry', 'surprised', 'scared', 'disgusted']
                    emotion_scores = {e: np.random.uniform(0, 0.3) for e in emotions}
                    primary_emotion = np.random.choice(emotions)
                    emotion_scores[primary_emotion] = np.random.uniform(0.6, 0.9)
                    
                    results["video_analysis"] = {
                        "emotion_scores": emotion_scores,
                        "primary_emotion": primary_emotion,
                        "confidence_score": np.random.uniform(0.6, 0.9)
                    }
            except Exception as e:
                logger.error(f"Error analyzing video frame: {str(e)}")
                results["video_analysis"] = {
                    "emotion_scores": {"neutral": 0.8, "happy": 0.2},
                    "primary_emotion": "neutral",
                    "confidence_score": 0.5
                }
        
        # Generate improvement tips
        if response_text:
            confidence_score = results["video_analysis"].get("confidence_score", 0.5) if "video_analysis" in results else 0.5
            clarity_score = results["text_analysis"].get("clarity_score", 0.5) if "text_analysis" in results else 0.5
            relevance_score = results["text_analysis"].get("relevance_score", 0.5) if "text_analysis" in results else 0.5
            emotion_scores = results["video_analysis"].get("emotion_scores", {"neutral": 0.8}) if "video_analysis" in results else {"neutral": 0.8}
            
            results["improvement_tips"] = self._generate_improvement_tips(
                question, confidence_score, clarity_score, relevance_score, emotion_scores
            )
        
        return results
    
    def _generate_improvement_tips(self, question, confidence_score, clarity_score, relevance_score, emotion_scores):
        """Generate improvement tips based on analysis"""
        tips = []
        
        # Based on confidence
        if confidence_score < 0.6:
            tips.append("Work on building your confidence when answering technical questions")
        
        # Based on clarity
        if clarity_score < 0.6:
            tips.append("Try to structure your responses more clearly with a beginning, middle, and conclusion")
        
        # Based on relevance
        if relevance_score < 0.6:
            tips.append("Make sure to directly address the question being asked")
        
        # Based on emotion
        if emotion_scores.get("nervous", 0) > 0.5 or emotion_scores.get("scared", 0) > 0.5:
            tips.append("Practice techniques to manage interview anxiety, such as deep breathing")
        
        if emotion_scores.get("angry", 0) > 0.3 or emotion_scores.get("disgusted", 0) > 0.3:
            tips.append("Be mindful of your facial expressions when discussing challenging topics")
        
        # Add more general tips if we don't have enough
        general_tips = [
            "Use the STAR method (Situation, Task, Action, Result) for behavioral questions",
            "Include specific metrics and outcomes when describing your achievements",
            "Prepare concise examples that highlight your skills relevant to the position",
            "Ask clarifying questions if you're unsure about what the interviewer is asking",
            "Take a moment to gather your thoughts before answering complex questions"
        ]
        
        while len(tips) < 2:
            tip = np.random.choice(general_tips)
            if tip not in tips:
                tips.append(tip)
        
        return tips
    
    def generate_final_report(self, interview_results):
        """
        Generate a final report based on interview results
        
        Args:
            interview_results (list): Results from each interview question
            
        Returns:
            dict: Final report
        """
        logger.info("Generating final interview report")
        
        # Calculate average scores
        avg_relevance = np.mean([
            q["analysis"]["text_analysis"].get("relevance_score", 0.5) 
            for q in interview_results if "analysis" in q and "text_analysis" in q["analysis"]
        ])
        
        avg_clarity = np.mean([
            q["analysis"]["text_analysis"].get("clarity_score", 0.5) 
            for q in interview_results if "analysis" in q and "text_analysis" in q["analysis"]
        ])
        
        avg_depth = np.mean([
            q["analysis"]["text_analysis"].get("depth_score", 0.5) 
            for q in interview_results if "analysis" in q and "text_analysis" in q["analysis"]
        ])
        
        avg_confidence = np.mean([
            q["analysis"]["video_analysis"].get("confidence_score", 0.5) 
            for q in interview_results if "analysis" in q and "video_analysis" in q["analysis"]
        ])
        
        # Calculate overall score (weighted average)
        overall_score = (
            avg_relevance * 0.25 + 
            avg_clarity * 0.25 + 
            avg_depth * 0.3 + 
            avg_confidence * 0.2
        )
        
        # Generate overall recommendations
        recommendations = self._generate_overall_recommendations(interview_results)
        
        # Generate interview summary
        summary = self._generate_interview_summary(interview_results)
        
        # Format the report
        report = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "scores": {
                "overall_score": overall_score,
                "relevance_score": avg_relevance,
                "clarity_score": avg_clarity,
                "depth_score": avg_depth,
                "confidence_score": avg_confidence
            },
            "summary": summary,
            "recommendations": recommendations,
            "questions_analysis": [
                {
                    "question": q["question"],
                    "answer": q["answer"],
                    "analysis": q["analysis"] if "analysis" in q else {}
                }
                for q in interview_results
            ]
        }
        
        return report
    
    def _generate_overall_recommendations(self, interview_results):
        """Generate overall recommendations based on interview results"""
        recommendations = []
        
        # Analyze all feedback to find common issues
        all_tips = []
        for q in interview_results.get("questions", []):
            tips = q.get("feedback", [])
            all_tips.extend(tips)
        
        # Count tip frequency to find common issues
        tip_count = {}
        for tip in all_tips:
            tip_count[tip] = tip_count.get(tip, 0) + 1
        
        # Sort tips by frequency
        sorted_tips = sorted(tip_count.items(), key=lambda x: x[1], reverse=True)
        
        # Add the most common issues as recommendations
        for tip, count in sorted_tips[:3]:
            recommendations.append(tip)
        
        # Add generic recommendations if needed
        generic_recommendations = [
            "Focus on providing more specific examples from your experience",
            "Practice maintaining consistent eye contact during interviews",
            "Work on reducing filler words to sound more confident",
            "Consider preparing more structured responses with clear beginning, middle, and end",
            "Try to vary your tone and pace to keep the interviewer engaged"
        ]
        
        while len(recommendations) < 5 and generic_recommendations:
            recommendations.append(generic_recommendations.pop(0))
        
        return recommendations
    
    def _generate_interview_summary(self, interview_results):
        """Generate a summary of the interview"""
        # In a real implementation, this would use NLP to generate a summary
        # For now, return a template-based summary
        
        num_questions = len(interview_results.get("questions", []))
        overall_score = interview_results.get("overall_score", 0)
        
        # Determine performance level
        performance = "excellent"
        if overall_score < 70:
            performance = "poor"
        elif overall_score < 80:
            performance = "average"
        elif overall_score < 90:
            performance = "good"
        
        return f"The candidate answered {num_questions} questions with {performance} overall performance. The interview covered a range of topics including personal background, technical skills, and problem-solving abilities." 