import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# Ensure preprocessing modules are in path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import CV parser and interview analyzer
from src.preprocessing.cv_parser import CVParser
from .interview_analyzer import AdvancedInterviewAnalyzer

class CandidateEvaluator:
    """
    Comprehensive candidate evaluation system that combines CV analysis with
    interview assessment for the career guidance platform.
    """
    
    def __init__(self, models_dir="models", output_dir="results", config_file=None):
        """
        Initialize the candidate evaluator
        
        Args:
            models_dir (str): Directory containing models
            output_dir (str): Directory to save results
            config_file (str): Optional path to configuration file
        """
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Initialize components
        self.cv_parser = CVParser(
            language=self.config.get("language", "fr"),
            skills_file=str(self.models_dir / "nlp" / "skills_taxonomy.json")
        )
        
        self.interview_analyzer = AdvancedInterviewAnalyzer()
        
        # Weights for different evaluation components
        self.weights = self.config.get("weights", {
            "cv_skills_match": 0.25,
            "cv_experience": 0.15,
            "interview_facial": 0.15,
            "interview_vocal": 0.15,
            "interview_sentiment": 0.15,
            "problem_solving": 0.15
        })
        
        # Career areas and their required skills
        self.career_areas = self.config.get("career_areas", {
            "software_development": [
                "Python", "Java", "JavaScript", "React", "Angular", 
                "Git", "Problem Solving", "Communication"
            ],
            "data_science": [
                "Python", "R", "Machine Learning", "SQL", "Statistics",
                "Data Visualization", "Analytical Thinking"
            ],
            "tourism": [
                "Customer Service", "Communication", "Foreign Languages",
                "Cultural Knowledge", "Problem Solving", "Hospitality"
            ],
            "agribusiness": [
                "Agricultural Knowledge", "Business Management", "Sustainability",
                "Supply Chain", "Technical Skills", "Economics"
            ],
            "renewable_energy": [
                "Engineering", "Technical Knowledge", "Project Management",
                "Sustainability", "Innovation", "Problem Solving"
            ]
        })
        
    def _load_config(self, config_file):
        """Load configuration from file or use defaults"""
        default_config = {
            "language": "fr",
            "weights": {
                "cv_skills_match": 0.25,
                "cv_experience": 0.15,
                "interview_facial": 0.15,
                "interview_vocal": 0.15,
                "interview_sentiment": 0.15,
                "problem_solving": 0.15
            },
            "min_experience_years": 0,
            "career_areas": {
                "software_development": [
                    "Python", "Java", "JavaScript", "React", "Angular", 
                    "Git", "Problem Solving", "Communication"
                ],
                "data_science": [
                    "Python", "R", "Machine Learning", "SQL", "Statistics",
                    "Data Visualization", "Analytical Thinking"
                ],
                "tourism": [
                    "Customer Service", "Communication", "Foreign Languages",
                    "Cultural Knowledge", "Problem Solving", "Hospitality"
                ],
                "agribusiness": [
                    "Agricultural Knowledge", "Business Management", "Sustainability",
                    "Supply Chain", "Technical Skills", "Economics"
                ],
                "renewable_energy": [
                    "Engineering", "Technical Knowledge", "Project Management",
                    "Sustainability", "Innovation", "Problem Solving"
                ]
            },
            "regions": {
                "Casablanca-Settat": ["software_development", "data_science"],
                "Tanger-Tétouan-Al Hoceïma": ["tourism"],
                "Souss-Massa": ["tourism", "agribusiness"],
                "Oriental": ["renewable_energy", "agribusiness"],
                "Guelmim-Oued Noun": ["renewable_energy", "agribusiness"]
            }
        }
        
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
                # Merge configs, with loaded values taking precedence
                for key, value in loaded_config.items():
                    default_config[key] = value
        
        return default_config
    
    def evaluate_candidate(self, cv_file, interview_video=None, candidate_region=None):
        """
        Perform comprehensive evaluation of a candidate
        
        Args:
            cv_file (str): Path to CV file
            interview_video (str): Optional path to interview video
            candidate_region (str): Optional region of the candidate
            
        Returns:
            dict: Comprehensive evaluation results
        """
        results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "candidate_info": {},
            "cv_analysis": {},
            "interview_analysis": {},
            "career_matches": {},
            "overall_score": 0,
            "recommendations": []
        }
        
        # Process CV
        if cv_file and os.path.exists(cv_file):
            cv_data = self.cv_parser.extract_from_file(cv_file)
            results["cv_analysis"] = cv_data
            
            # Extract candidate info from CV
            if "personal_info" in cv_data:
                results["candidate_info"] = cv_data["personal_info"]
            
            # Calculate career matches based on CV
            results["career_matches"] = self._calculate_career_matches(cv_data, candidate_region)
        
        # Process interview if available
        if interview_video and os.path.exists(interview_video):
            interview_results = self.interview_analyzer.analyze_interview(interview_video)
            results["interview_analysis"] = interview_results
        
        # Calculate overall score
        results["overall_score"] = self._calculate_overall_score(results)
        
        # Generate recommendations
        results["recommendations"] = self._generate_recommendations(results)
        
        # Save results
        output_file = self.output_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        
        return results
    
    def _calculate_career_matches(self, cv_data, candidate_region=None):
        """Calculate match percentages for different career paths"""
        # Extract candidate skills (handle both string skills and dictionary skills)
        candidate_skills = []
        for skill in cv_data.get("skills", []):
            if isinstance(skill, dict) and "name" in skill:
                candidate_skills.append(skill["name"].lower())
            elif isinstance(skill, str):
                candidate_skills.append(skill.lower())
        
        # Calculate match for each career
        matches = []
        
        # Calculate match for each career area
        for area, required_skills in self.career_areas.items():
            required_skills_lower = [skill.lower() for skill in required_skills]
            
            # Count matching skills
            matching_skills = [skill for skill in candidate_skills if skill in required_skills_lower]
            
            # Calculate match percentage
            if required_skills:
                match_percentage = (len(matching_skills) / len(required_skills)) * 100
            else:
                match_percentage = 0
            
            matches.append({
                "career": area,
                "match_percentage": match_percentage,
                "matching_skills": matching_skills,
                "missing_skills": [skill for skill in required_skills if skill.lower() not in candidate_skills]
            })
        
        # Adjust scores based on region if provided
        if candidate_region and candidate_region in self.config.get("regions", {}):
            regional_focus = self.config["regions"][candidate_region]
            
            # Boost scores for regionally relevant careers
            for match in matches:
                if match["career"] in regional_focus:
                    # Add 15% boost for regionally relevant careers
                    match["match_percentage"] = min(100, match["match_percentage"] * 1.15)
                    match["regional_fit"] = True
        
        # Sort careers by match percentage
        matches = sorted(matches, key=lambda x: x["match_percentage"], reverse=True)
        
        return matches
    
    def _calculate_overall_score(self, results):
        """Calculate overall candidate score"""
        # Initialize components with default values
        components = {
            "cv_skills_match": 0,
            "cv_experience": 0,
            "interview_facial": 0,
            "interview_vocal": 0,
            "interview_sentiment": 0,
            "problem_solving": 0
        }
        
        # CV skills match - based on top career match
        if results.get("career_matches"):
            top_career = results["career_matches"][0]["career"]
            components["cv_skills_match"] = results["career_matches"][0]["match_percentage"] / 100
        
        # CV experience score
        if "experience" in results.get("cv_analysis", {}):
            experience = results["cv_analysis"]["experience"]
            # Calculate based on years and relevance
            total_years = 0
            for exp in experience:
                if "start_year" in exp and "end_year" in exp:
                    start = exp.get("start_year")
                    end = exp.get("end_year")
                    
                    if start and start.isdigit():
                        if end and end.isdigit():
                            total_years += int(end) - int(start)
                        elif end and end.lower() == "présent":
                            current_year = datetime.now().year
                            total_years += current_year - int(start)
            
            # Score based on years (capped at 10 years = 1.0)
            components["cv_experience"] = min(1.0, total_years / 10)
        
        # Interview scores if available
        interview_data = results.get("interview_analysis", {})
        
        if "facial_emotions" in interview_data:
            # Calculate facial emotion score (confidence, positive emotions)
            emotions = interview_data["facial_emotions"]
            # Higher scores for positive emotions like happiness, lower for negative like anger
            components["interview_facial"] = min(1.0, (
                emotions.get("happiness", 0) * 1.0 + 
                emotions.get("surprise", 0) * 0.5 -
                emotions.get("anger", 0) * 0.8 -
                emotions.get("disgust", 0) * 0.7 -
                emotions.get("fear", 0) * 0.6 -
                emotions.get("sadness", 0) * 0.5
            ))
        
        if "voice_emotions" in interview_data:
            # Calculate voice emotion score
            voice = interview_data["voice_emotions"]
            # Higher scores for confidence, clarity
            components["interview_vocal"] = min(1.0, (
                voice.get("confidence", 0) * 1.0 +
                voice.get("clarity", 0) * 0.8 -
                voice.get("nervousness", 0) * 0.7
            ))
        
        if "sentiment_analysis" in interview_data:
            # Calculate sentiment score
            sentiment = interview_data["sentiment_analysis"]
            components["interview_sentiment"] = min(1.0, (
                sentiment.get("positive", 0) * 1.0 -
                sentiment.get("negative", 0) * 0.8
            ))
        
        if "problem_solving" in interview_data:
            # Problem solving score
            components["problem_solving"] = min(1.0, interview_data["problem_solving"].get("score", 0) / 100)
        
        # Calculate weighted score
        overall_score = 0
        for component, score in components.items():
            overall_score += score * self.weights.get(component, 0)
        
        # Scale to 0-100
        overall_score = overall_score * 100
        
        return overall_score
    
    def _generate_recommendations(self, results):
        """Generate personalized recommendations based on evaluation"""
        recommendations = []
        
        # Career recommendations
        if results.get("career_matches"):
            top_careers = results["career_matches"][:3]  # Top 3 matches
            
            # Recommend top career paths
            career_rec = {
                "type": "career_paths",
                "message": f"Based on your skills and experience, consider these career paths:",
                "options": []
            }
            
            for match in top_careers:
                career_rec["options"].append({
                    "career": match["career"],
                    "match_percentage": match["match_percentage"],
                    "regional_fit": match.get("regional_fit", False)
                })
            
            recommendations.append(career_rec)
            
            # Skill improvement recommendations
            top_career = top_careers[0]["career"]
            missing_skills = top_careers[0]["missing_skills"]
            
            if missing_skills:
                skill_rec = {
                    "type": "skill_improvement",
                    "message": f"To improve your prospects in {top_career}, consider developing these skills:",
                    "skills": missing_skills[:5]  # Top 5 missing skills
                }
                recommendations.append(skill_rec)
        
        # Interview improvement recommendations
        interview_data = results.get("interview_analysis", {})
        if interview_data:
            interview_recs = []
            
            # Facial expression recommendations
            if "facial_emotions" in interview_data:
                emotions = interview_data["facial_emotions"]
                if emotions.get("happiness", 0) < 0.3:
                    interview_recs.append("Show more positive expressions during interviews")
                if emotions.get("anger", 0) > 0.1 or emotions.get("disgust", 0) > 0.1:
                    interview_recs.append("Avoid negative expressions like anger or disgust")
            
            # Voice recommendations
            if "voice_emotions" in interview_data:
                voice = interview_data["voice_emotions"]
                if voice.get("confidence", 0) < 0.4:
                    interview_recs.append("Work on speaking with more confidence")
                if voice.get("clarity", 0) < 0.4:
                    interview_recs.append("Practice speaking more clearly")
                if voice.get("nervousness", 0) > 0.6:
                    interview_recs.append("Try relaxation techniques to reduce nervousness")
            
            # Communication recommendations
            if "sentiment_analysis" in interview_data:
                sentiment = interview_data["sentiment_analysis"]
                if sentiment.get("negative", 0) > 0.4:
                    interview_recs.append("Use more positive language in your responses")
            
            if interview_recs:
                rec = {
                    "type": "interview_improvement",
                    "message": "Here are some ways to improve your interview performance:",
                    "suggestions": interview_recs
                }
                recommendations.append(rec)
        
        # Educational recommendations
        if "education" in results.get("cv_analysis", {}):
            education = results["cv_analysis"]["education"]
            
            # If no higher education, suggest courses
            if not education or all("baccalauréat" in (e.get("degree", "").lower()) for e in education):
                edu_rec = {
                    "type": "education",
                    "message": "Consider these educational opportunities to enhance your qualifications:",
                    "suggestions": [
                        "Certificate programs in your area of interest",
                        "Online courses through platforms like Coursera or edX",
                        "Vocational training programs"
                    ]
                }
                recommendations.append(edu_rec)
        
        # Final overall recommendation
        overall_score = results.get("overall_score", 0)
        overall_rec = {
            "type": "overall",
            "message": ""
        }
        
        if overall_score >= 80:
            overall_rec["message"] = "You have a strong profile! Focus on refining your interview skills to maximize your opportunities."
        elif overall_score >= 60:
            overall_rec["message"] = "You have a good foundation. Develop the recommended skills to enhance your career prospects."
        else:
            overall_rec["message"] = "Consider focusing on the suggested skill improvements and interview techniques to strengthen your profile."
        
        recommendations.append(overall_rec)
        
        return recommendations


# Example usage
if __name__ == "__main__":
    evaluator = CandidateEvaluator()
    
    # Example evaluation - replace with actual file paths
    # results = evaluator.evaluate_candidate(
    #     cv_file="path/to/cv.pdf",
    #     interview_video="path/to/interview.mp4",
    #     candidate_region="Casablanca-Settat"
    # )
    
    # Print sample usage
    print("CandidateEvaluator initialized successfully")
    print("Sample usage:")
    print("  evaluator = CandidateEvaluator()")
    print("  results = evaluator.evaluate_candidate(")
    print("      cv_file='data/cvs/candidate1.pdf',")
    print("      interview_video='data/interviews/candidate1.mp4',")
    print("      candidate_region='Casablanca-Settat'")
    print("  )")
    print("  print(results['overall_score'])")
    print("  print(results['recommendations'])") 