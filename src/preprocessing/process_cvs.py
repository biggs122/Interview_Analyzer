import os
import json
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm
from cv_parser import CVParser, create_skills_taxonomy
from src.utils.model_version_manager import get_performance_comparison

class CVProcessor:
    def __init__(self, input_dir, output_dir, language="fr"):
        """
        Initialize the CV batch processor
        
        Args:
            input_dir (str): Directory containing CV files (PDF, DOCX)
            output_dir (str): Directory to save processed results
            language (str): Primary language for CV parsing
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.language = language
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create skills taxonomy if it doesn't exist
        skills_file = Path("models/nlp/skills_taxonomy.json")
        if not skills_file.exists():
            print("Creating skills taxonomy...")
            skills = create_skills_taxonomy(domain_specific=True)
            skills_file.parent.mkdir(parents=True, exist_ok=True)
            with open(skills_file, "w", encoding="utf-8") as f:
                json.dump(skills, f, ensure_ascii=False, indent=4)
        
        # Initialize CV parser
        self.parser = CVParser(language=language, skills_file=str(skills_file))
        
        # Statistics
        self.stats = {
            "total_files": 0,
            "processed_files": 0,
            "failed_files": 0,
            "skills_detected": {},
            "top_experiences": {}
        }
    
    def process_all_cvs(self):
        """Process all CVs in the input directory"""
        cv_files = []
        
        # Find all PDF and DOCX files
        for ext in ["*.pdf", "*.PDF", "*.docx", "*.DOCX"]:
            cv_files.extend(list(self.input_dir.glob(ext)))
        
        self.stats["total_files"] = len(cv_files)
        print(f"Found {self.stats['total_files']} CV files")
        
        # Process each CV
        results = []
        for cv_file in tqdm(cv_files, desc="Processing CVs"):
            try:
                cv_data = self.parser.extract_from_file(str(cv_file))
                
                # Add filename and file path
                cv_data["filename"] = cv_file.name
                cv_data["file_path"] = str(cv_file)
                
                # Save individual result
                output_file = self.output_dir / f"{cv_file.stem}.json"
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(cv_data, f, ensure_ascii=False, indent=4)
                
                # Update statistics
                self.update_stats(cv_data)
                
                # Add to results list
                results.append(cv_data)
                self.stats["processed_files"] += 1
                
            except Exception as e:
                print(f"Error processing {cv_file.name}: {str(e)}")
                self.stats["failed_files"] += 1
        
        # Create summary DataFrame
        self.create_summary_dataframe(results)
        
        # Save statistics
        self.save_stats()
        
        return results
    
    def update_stats(self, cv_data):
        """Update statistics based on CV data"""
        # Count skills
        if "skills" in cv_data:
            for skill in cv_data["skills"]:
                skill_lower = skill.lower()
                if skill_lower in self.stats["skills_detected"]:
                    self.stats["skills_detected"][skill_lower] += 1
                else:
                    self.stats["skills_detected"][skill_lower] = 1
        
        # Count job titles
        if "experience" in cv_data:
            for exp in cv_data["experience"]:
                if "job_title" in exp:
                    job_title = exp["job_title"].lower()
                    if job_title in self.stats["top_experiences"]:
                        self.stats["top_experiences"][job_title] += 1
                    else:
                        self.stats["top_experiences"][job_title] = 1
    
    def create_summary_dataframe(self, results):
        """Create a summary DataFrame from all processed CVs"""
        summary_data = []
        
        for cv in results:
            # Extract basic information
            cv_summary = {
                "filename": cv.get("filename", ""),
                "name": cv.get("personal_info", {}).get("name", ""),
                "email": cv.get("personal_info", {}).get("email", ""),
                "phone": cv.get("personal_info", {}).get("phone", ""),
                "linkedin": cv.get("personal_info", {}).get("linkedin", ""),
                "skills_count": len(cv.get("skills", [])),
                "experience_count": len(cv.get("experience", [])),
                "education_count": len(cv.get("education", [])),
                "languages": ", ".join(cv.get("languages", [])),
                "all_skills": ", ".join(cv.get("skills", []))
            }
            
            # Add highest education
            if cv.get("education"):
                cv_summary["highest_education"] = cv["education"][0].get("degree", "")
                cv_summary["institution"] = cv["education"][0].get("institution", "")
            
            # Add most recent experience
            if cv.get("experience"):
                cv_summary["recent_job"] = cv["experience"][0].get("job_title", "")
                cv_summary["recent_company"] = cv["experience"][0].get("company", "")
                
                # Calculate total years of experience
                total_years = 0
                for exp in cv.get("experience", []):
                    start = exp.get("start_year")
                    end = exp.get("end_year")
                    
                    if start and end and start.isdigit():
                        if end.isdigit():
                            total_years += int(end) - int(start)
                        elif end.lower() == "présent":
                            import datetime
                            current_year = datetime.datetime.now().year
                            total_years += current_year - int(start)
                
                cv_summary["years_experience"] = total_years
            
            summary_data.append(cv_summary)
        
        # Create DataFrame
        df = pd.DataFrame(summary_data)
        
        # Save to CSV
        df.to_csv(self.output_dir / "cv_summary.csv", index=False, encoding="utf-8")
        
        # Return DataFrame
        return df
    
    def save_stats(self):
        """Save processing statistics"""
        # Sort skills and experiences by frequency
        self.stats["skills_detected"] = dict(sorted(
            self.stats["skills_detected"].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:20])  # Top 20 skills
        
        self.stats["top_experiences"] = dict(sorted(
            self.stats["top_experiences"].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:20])  # Top 20 job titles
        
        # Add processing summary
        self.stats["success_rate"] = f"{(self.stats['processed_files'] / self.stats['total_files'] * 100):.1f}%" if self.stats['total_files'] > 0 else "0%"
        
        # Save statistics to JSON
        with open(self.output_dir / "processing_stats.json", "w", encoding="utf-8") as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=4)
        
        # Print summary
        print(f"\nProcessing complete:")
        print(f"  - Processed: {self.stats['processed_files']} of {self.stats['total_files']} files ({self.stats['success_rate']})")
        print(f"  - Failed: {self.stats['failed_files']} files")
        
        if self.stats["skills_detected"]:
            print("\nTop 5 skills detected:")
            for i, (skill, count) in enumerate(list(self.stats["skills_detected"].items())[:5]):
                print(f"  {i+1}. {skill} ({count} occurrences)")

def main():
    """Main function to run CV processor"""
    parser = argparse.ArgumentParser(description="Process CV files for the career guidance platform")
    parser.add_argument("--input", "-i", required=True, help="Directory containing CV files")
    parser.add_argument("--output", "-o", required=True, help="Directory to save processed results")
    parser.add_argument("--language", "-l", default="fr", choices=["fr", "ar", "en"], help="Primary language for CV parsing")
    args = parser.parse_args()
    
    processor = CVProcessor(
        input_dir=args.input,
        output_dir=args.output,
        language=args.language
    )
    
    processor.process_all_cvs()

    # Get performance comparison
    comparison = get_performance_comparison()
    print(comparison)

if __name__ == "__main__":
    main() 