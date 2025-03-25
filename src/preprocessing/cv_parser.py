import re
import os
import json
import spacy
import pandas as pd
from pathlib import Path
import pdfplumber
import docx2txt
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords

# Download necessary NLTK resources
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    print("Warning: Could not download NLTK resources. Some features may be limited.")

class CVParser:
    def __init__(self, language="fr", skills_file=None):
        """
        Initialize the CV Parser
        
        Args:
            language (str): The primary language of CVs to parse ('fr', 'ar', 'en')
            skills_file (str): Path to a JSON file containing skill taxonomies
        """
        self.language = language
        
        # Load appropriate spaCy model based on language with fallback options
        self.nlp = self._load_spacy_model(language)
        
        # Load skills taxonomy if provided
        self.skills = []
        if skills_file and os.path.exists(skills_file):
            try:
                with open(skills_file, 'r', encoding='utf-8') as f:
                    self.skills = json.load(f)
            except:
                print(f"Warning: Could not load skills file at {skills_file}. Using empty skills list.")
        
        # Initialize regex patterns
        self._init_patterns()
        
        # Initialize transformers NER model for better extraction
        try:
            self.ner_model = pipeline(
                "token-classification", 
                model="jean-baptiste/camembert-ner",
                aggregation_strategy="simple"
            )
        except:
            print("Warning: Could not load NER model. Using basic extraction methods.")
            self.ner_model = None
        
        # TF-IDF vectorizer for skills matching
        self.vectorizer = TfidfVectorizer(min_df=1, analyzer='word', ngram_range=(1, 2))
        
    def _load_spacy_model(self, language):
        """Load spaCy model with fallbacks for missing models"""
        # Try to load the requested language model
        try:
            if language == "fr":
                return spacy.load("fr_core_news_md", disable=["parser"])
            elif language == "ar":
                return spacy.load("xx_ent_wiki_sm")  # Multilingual model
            else:  # Default to English
                return spacy.load("en_core_web_md", disable=["parser"])
        except IOError:
            # First fallback: try english if other language fails
            try:
                print(f"Warning: Could not load {language} model. Trying English model...")
                return spacy.load("en_core_web_md", disable=["parser"])
            except IOError:
                # Second fallback: try the small model
                try:
                    print("Warning: Could not load medium English model. Trying small English model...")
                    return spacy.load("en_core_web_sm", disable=["parser"])
                except IOError:
                    # Final fallback: use blank model
                    print("Warning: No spaCy models available. Using blank model with limited functionality.")
                    return spacy.blank("en")

    def _init_patterns(self):
        """Initialize regex patterns for information extraction"""
        # Contact information patterns
        self.email_pattern = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
        self.phone_pattern = re.compile(r'(?:(?:\+|00)212|0)\s*(?:5|6|7)(?:[\s.-]*\d{2}){4}')
        self.linkedin_pattern = re.compile(r'linkedin\.com/in/[\w-]+')
        
        # Education and experience patterns
        self.education_headers = [
            'formation', 'education', 'études', 'scolarité', 'diplômes', 'diplome', 'cursus'
        ]
        self.experience_headers = [
            'expérience', 'expériences', 'parcours professionnel', 'emplois', 'postes'
        ]
        self.skills_headers = [
            'compétences', 'aptitudes', 'qualifications', 'savoir-faire', 'connaissances'
        ]
        
    def extract_from_file(self, file_path):
        """
        Extract information from a CV file
        
        Args:
            file_path (str): Path to the CV file (PDF or DOCX)
            
        Returns:
            dict: Extracted information including personal details, education, experience, skills
        """
        text = self._extract_text(file_path)
        if not text:
            return {"error": "Could not extract text from file"}
        
        # Process the text with NLP
        doc = self.nlp(text)
        
        # Extract information
        result = {
            "personal_info": self._extract_personal_info(text, doc),
            "education": self._extract_education(text, doc),
            "experience": self._extract_experience(text, doc),
            "skills": self._extract_skills(text, doc),
            "languages": self._extract_languages(text, doc)
        }
        
        return result
    
    def _extract_text(self, file_path):
        """Extract text from PDF or DOCX file"""
        try:
            file_path = Path(file_path)
            
            if file_path.suffix.lower() == '.pdf':
                try:
                    # Try to extract with structure preservation
                    text = ""
                    sections = {}
                    with pdfplumber.open(file_path) as pdf:
                        for page_num, page in enumerate(pdf.pages):
                            page_text = page.extract_text() or ""
                            text += page_text
                            
                            # Try to detect sections on the first page
                            if page_num == 0:
                                # Look for contact info at the top
                                lines = page_text.split('\n')
                                if len(lines) > 5:
                                    sections['header'] = '\n'.join(lines[:5])
                                
                                # Check for common section headers
                                for pattern in ['EXPERIENCE', 'EDUCATION', 'SKILLS', 'LANGUAGES', 'CERTIFICATIONS']:
                                    if pattern in page_text:
                                        start_idx = page_text.find(pattern)
                                        if start_idx != -1:
                                            # Get content after the section header
                                            section_content = page_text[start_idx:]
                                            # Find next section if it exists
                                            next_section = None
                                            for next_pattern in ['EXPERIENCE', 'EDUCATION', 'SKILLS', 'LANGUAGES', 'CERTIFICATIONS']:
                                                if next_pattern != pattern and next_pattern in section_content[len(pattern):]:
                                                    next_idx = section_content[len(pattern):].find(next_pattern)
                                                    next_section = section_content[len(pattern):][next_idx]
                                                    sections[pattern] = section_content[len(pattern):next_idx].strip()
                                                    break
                                            if next_section is None:
                                                sections[pattern] = section_content[len(pattern):].strip()
                    
                    # If we extracted some structure, note it for later use
                    if sections:
                        self.cv_structure = sections
                    
                    return text
                except Exception as e:
                    print(f"Error extracting PDF text: {str(e)}")
                    # Try fallback PDF extraction if available
                    try:
                        import PyPDF2
                        text = ""
                        with open(file_path, 'rb') as file:
                            reader = PyPDF2.PdfReader(file)
                            for page_num in range(len(reader.pages)):
                                text += reader.pages[page_num].extract_text() or ""
                        return text
                    except Exception as pdf_fallback_error:
                        print(f"Fallback PDF extraction also failed: {str(pdf_fallback_error)}")
            
            elif file_path.suffix.lower() == '.docx':
                try:
                    return docx2txt.process(file_path)
                except Exception as e:
                    print(f"Error extracting DOCX text: {str(e)}")
            
            elif file_path.suffix.lower() == '.txt':
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return f.read()
                except UnicodeDecodeError:
                    # Try with different encodings
                    for encoding in ['latin-1', 'cp1252', 'ISO-8859-1']:
                        try:
                            with open(file_path, 'r', encoding=encoding) as f:
                                return f.read()
                        except:
                            pass
                    print(f"Error: Could not decode text file with any common encoding")
            
            else:
                print(f"Warning: Unsupported file format: {file_path.suffix}")
        
        except Exception as e:
            print(f"Error in text extraction: {str(e)}")
        
        return ""
    
    def _extract_personal_info(self, text, doc):
        """Extract name, email, phone, and LinkedIn profile"""
        personal_info = {
            "name": "",
            "email": "",
            "phone": "",
            "linkedin": ""
        }
        
        # Check if CV structure was detected in PDF extraction
        if hasattr(self, 'cv_structure') and 'header' in self.cv_structure:
            header_text = self.cv_structure['header']
        else:
            # Use first 5 lines as header if available
            lines = text.split('\n')
            header_text = '\n'.join(lines[:min(5, len(lines))])
        
        # Find email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_match = re.search(email_pattern, text)
        if email_match:
            personal_info["email"] = email_match.group(0)
        
        # Find phone
        phone_patterns = [
            r'\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',  # +1 (123) 456-7890
            r'\b(?:\+\d{1,3}[-.\s]?)?\d{10}\b',  # +1234567890
            r'\b(?:\+\d{1,3}[-.\s]?)?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',  # +1 123-456-7890
            r'\b(?:\+\d{1,3}[-.\s]?)?\d{2}[-.\s]?\d{2}[-.\s]?\d{2}[-.\s]?\d{2}[-.\s]?\d{2}\b'  # +1 12-34-56-78-90
        ]
        
        for pattern in phone_patterns:
            phone_match = re.search(pattern, text)
            if phone_match:
                personal_info["phone"] = phone_match.group(0)
                break
        
        # Find LinkedIn
        linkedin_pattern = r'(?:linkedin\.com/in/|linkedin\.com/profile/view\?id=)([a-zA-Z0-9_-]+)'
        linkedin_match = re.search(linkedin_pattern, text)
        if linkedin_match:
            personal_info["linkedin"] = f"linkedin.com/in/{linkedin_match.group(1)}"
        
        # Try to find name
        try:
            # First try with NER if available
            name = ""
            if doc is not None:
                for ent in doc.ents:
                    if ent.label_ == "PERSON":
                        name = ent.text
                        break
            
            # If name not found with NER, try other approaches
            if not name:
                # Look for name patterns in header
                # Case 1: Names in all caps at start of document
                cap_name_match = re.search(r'^([A-Z][A-Z\s]+)(?:\r?\n|$)', header_text)
                if cap_name_match:
                    name = cap_name_match.group(1).strip()
                    
                # Case 2: CV or Resume or Curriculum Vitae of...
                cv_of_match = re.search(r'(?:CV|[Rr]esume|[Cc]urriculum [Vv]itae)(?:\s+of|\s*:)?\s+([A-Z][a-z]+\s+[A-Z][a-z]+)', text)
                if cv_of_match:
                    name = cv_of_match.group(1)
                    
                # Case 3: First line that's not a title or CV
                if not name:
                    for line in header_text.split('\n'):
                        line = line.strip()
                        # Skip if line contains job titles or words like CV, Resume
                        if line and not re.search(r'(?:CV|Resume|Curriculum Vitae|Consultant|Engineer|Developer|INSTRUCTOR|CONSULTANT|METHODOLOGIST)', line):
                            if re.search(r'[A-Z][a-z]+ [A-Z][a-z]+|[A-Z][A-Z\s]+', line):
                                name = line
                                break
                
                # Last resort: use the first line if it's not too long
                if not name and len(header_text.split('\n')) > 0:
                    first_line = header_text.split('\n')[0].strip()
                    if 3 <= len(first_line.split()) <= 5:  # Reasonable name length
                        name = first_line
            
            personal_info["name"] = name.strip()
        except Exception as e:
            print(f"Error extracting name: {str(e)}")
            # If we get here, we couldn't extract a name, use a placeholder
            personal_info["name"] = "Unknown Candidate"
        
        return personal_info
    
    def _extract_education(self, text, doc):
        """Extract education information"""
        education = []
        
        # Find education section using headers
        education_section = self._find_section(text, self.education_headers)
        if not education_section:
            return education
        
        # Process education section with NER for institutions and dates
        edu_doc = self.nlp(education_section)
        
        # Extract education entities
        current_entry = {}
        lines = education_section.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Use pattern matching to identify degrees
            if re.search(r'(diplôme|master|licence|baccalauréat|ingénieur|doctorat)', line.lower()):
                if current_entry and "degree" in current_entry:
                    education.append(current_entry)
                    current_entry = {}
                current_entry["degree"] = line
                
                # Look for dates in the same line
                date_match = re.search(r'(\d{4})\s*[à-]\s*(\d{4}|\bprésent\b)', line, re.IGNORECASE)
                if date_match:
                    current_entry["start_year"] = date_match.group(1)
                    current_entry["end_year"] = date_match.group(2)
            
            # Look for institutions
            elif any(org in line for org in ["université", "école", "institut", "faculty"]):
                current_entry["institution"] = line
        
        # Add the last entry if not empty
        if current_entry and "degree" in current_entry:
            education.append(current_entry)
            
        return education
    
    def _extract_experience(self, text, doc):
        """Extract work experience information"""
        experiences = []
        
        # Find experience section
        experience_section = self._find_section(text, self.experience_headers)
        if not experience_section:
            return experiences
        
        # Process experience section
        current_entry = {}
        lines = experience_section.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for date patterns indicating new entries
            date_match = re.search(r'(\d{4})\s*[à-]\s*(\d{4}|\bprésent\b)', line, re.IGNORECASE)
            if date_match:
                if current_entry and "job_title" in current_entry:
                    experiences.append(current_entry)
                    current_entry = {}
                current_entry["start_year"] = date_match.group(1)
                current_entry["end_year"] = date_match.group(2)
                
                # Extract job title and company from the same line
                job_info = re.sub(r'(\d{4})\s*[à-]\s*(\d{4}|\bprésent\b)', '', line).strip()
                if job_info:
                    if ":" in job_info:
                        parts = job_info.split(":")
                        current_entry["job_title"] = parts[0].strip()
                        current_entry["company"] = parts[1].strip()
                    else:
                        current_entry["job_title"] = job_info
            
            # If line contains common job title indicators
            elif any(title in line.lower() for title in ["ingénieur", "développeur", "consultant", "manager", "directeur", "chef"]):
                current_entry["job_title"] = line
                
            # Look for company names
            elif any(corp in line.lower() for corp in ["sarl", "sa", "inc", "llc", "ltd", "inc."]):
                current_entry["company"] = line
                
            # Add description if entry has started
            elif current_entry and "job_title" in current_entry and "description" not in current_entry:
                current_entry["description"] = line
        
        # Add the last entry if not empty
        if current_entry and "job_title" in current_entry:
            experiences.append(current_entry)
            
        return experiences
    
    def _extract_skills(self, text, doc):
        """Extract skills from CV using NLP and skills taxonomy"""
        skills = []
        
        try:
            # First attempt: look for dedicated skills section if document structure was detected
            skills_section = ""
            if hasattr(self, 'cv_structure') and 'SKILLS' in self.cv_structure:
                skills_section = self.cv_structure['SKILLS']
            else:
                # Try to locate skills section using regex
                skills_patterns = [
                    r'(?:SKILLS|COMPETENCES|COMPETENCIES|TECHNOLOGIES|EXPERTISE|TECHNICAL SKILLS|PROFESSIONAL SKILLS)[:\s]*\n(.*?)(?:^\s*$|\Z|\n\s*[A-Z]{2,})',
                    r'(?:Skills|Competences|Competencies|Technologies|Expertise|Technical Skills|Professional Skills)[:\s]*\n(.*?)(?:^\s*$|\Z|\n\s*[A-Z][a-z]+\s+[A-Z][a-z]+:)'
                ]
                
                for pattern in skills_patterns:
                    skills_match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
                    if skills_match:
                        skills_section = skills_match.group(1)
                        break
            
            # Process the skills section if found
            if skills_section:
                # Split by common separators like commas, bullets, newlines
                skill_candidates = re.split(r'[,•\n]+', skills_section)
                for candidate in skill_candidates:
                    candidate = candidate.strip()
                    if candidate and 2 <= len(candidate) <= 50:  # Reasonable skill length
                        skills.append({"name": candidate, "level": "Not specified"})
            
            # Second attempt: Match against a skills taxonomy
            # Common technical skills
            tech_skills = [
                # Programming Languages
                r'\b(?:Python|Java|JavaScript|C\+\+|C#|Ruby|PHP|Swift|Kotlin|Go|Rust|TypeScript|Scala)\b',
                # Web Development
                r'\b(?:HTML5?|CSS3?|React|Angular|Vue\.js|Node\.js|Express|Django|Flask|Ruby on Rails|jQuery|Bootstrap|SASS|LESS|WordPress)\b',
                # Data Science & ML
                r'\b(?:TensorFlow|PyTorch|Keras|Scikit-learn|Pandas|NumPy|SciPy|R|MATLAB|Tableau|Power BI|Hadoop|Spark|Big Data|Machine Learning|Deep Learning|NLP|Computer Vision)\b',
                # Databases
                r'\b(?:SQL|MySQL|PostgreSQL|Oracle|MongoDB|Firebase|Redis|Cassandra|SQLite|NoSQL|GraphQL)\b',
                # DevOps & Tools
                r'\b(?:Git|Docker|Kubernetes|Jenkins|AWS|Azure|GCP|Terraform|Ansible|Linux|Unix|Bash|PowerShell|CI/CD|Agile|Scrum|JIRA)\b',
                # Microsoft Office
                r'\b(?:Excel|Word|PowerPoint|Outlook|Microsoft Office|MS Office)\b',
                # Design
                r'\b(?:Photoshop|Illustrator|InDesign|Sketch|Figma|XD|UI/UX|Graphic Design)\b'
            ]
            
            # Soft skills
            soft_skills = [
                r'\b(?:Communication|Leadership|Teamwork|Problem[\s-]Solving|Critical[\s-]Thinking|Time[\s-]Management|Adaptability|Creativity|Collaboration|Negotiation|Presentation|Customer[\s-]Service|Project[\s-]Management|Analytical[\s-]Skills|Decision[\s-]Making|Organization|Attention[\s-]to[\s-]Detail|Flexibility)\b'
            ]
            
            # Industry-specific skills
            industry_skills = [
                # Finance
                r'\b(?:Accounting|Financial[\s-]Analysis|Budgeting|Forecasting|Risk[\s-]Management|Investment|Banking|Financial[\s-]Reporting|Taxation|Auditing)\b',
                # Marketing
                r'\b(?:Digital[\s-]Marketing|SEO|SEM|Content[\s-]Marketing|Social[\s-]Media|Brand[\s-]Management|Market[\s-]Research|Email[\s-]Marketing|Growth[\s-]Hacking|Analytics)\b',
                # Healthcare
                r'\b(?:Patient[\s-]Care|Medical[\s-]Records|Clinical|Diagnostics|Healthcare[\s-]Management|Pharmaceutical|Nursing|Therapy|Rehabilitation)\b',
                # Education
                r'\b(?:Teaching|Curriculum[\s-]Development|E-Learning|Assessment|Training|Instructional[\s-]Design|Coaching|Mentoring)\b',
                # Hospitality
                r'\b(?:Customer[\s-]Service|Reservations|Hotel[\s-]Management|Food[\s-]Service|Event[\s-]Planning|Catering|Hospitality[\s-]Management)\b',
                # Agriculture
                r'\b(?:Farming|Cultivation|Irrigation|Crop[\s-]Management|Livestock|Sustainable[\s-]Agriculture|Organic[\s-]Farming|Agricultural[\s-]Engineering)\b'
            ]
            
            # Combine all skill patterns
            all_patterns = tech_skills + soft_skills + industry_skills
            
            # Find matches in text
            for pattern in all_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    skill_name = match.group(0)
                    # Check if already added
                    if not any(skill["name"].lower() == skill_name.lower() for skill in skills):
                        skills.append({"name": skill_name, "level": "Not specified"})
            
            # Third attempt: Try to use NLP if available
            if self.nlp and doc is not None:
                try:
                    # Extract noun chunks as potential skills
                    for chunk in doc.noun_chunks:
                        if 2 <= len(chunk.text.split()) <= 4:  # Reasonable skill length
                            skill_candidate = chunk.text.strip()
                            # Filter out common non-skill phrases
                            if not re.search(r'\b(?:I|me|my|year|month|day|time|experience|education|university|college|school)\b', 
                                           skill_candidate, re.IGNORECASE):
                                if not any(skill["name"].lower() == skill_candidate.lower() for skill in skills):
                                    skills.append({"name": skill_candidate, "level": "Not specified"})
                except Exception as e:
                    print(f"Warning: Error in NLP-based skill extraction: {str(e)}")
        
        except Exception as e:
            print(f"Error in skill extraction: {str(e)}")
            # Return some default skills from the regex patterns as fallback
            for pattern in tech_skills[:3] + soft_skills + industry_skills[:2]:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    skill_name = match.group(0)
                    if not any(skill["name"].lower() == skill_name.lower() for skill in skills):
                        skills.append({"name": skill_name, "level": "Not specified"})
        
        # Limit to top skills by frequency in the document
        skill_occurrences = {}
        for skill in skills:
            skill_name = skill["name"].lower()
            count = len(re.findall(r'\b' + re.escape(skill_name) + r'\b', text.lower()))
            skill_occurrences[skill["name"]] = count
        
        # Sort skills by occurrence count (highest first)
        sorted_skills = sorted(skills, key=lambda x: skill_occurrences.get(x["name"], 0), reverse=True)
        
        return sorted_skills
    
    def _extract_languages(self, text, doc):
        """Extract languages from CV"""
        languages = []
        
        try:
            # Common languages in Morocco
            language_patterns = [
                r'\b(arabe|arabic)\b',
                r'\b(français|french)\b',
                r'\b(anglais|english)\b',
                r'\b(espagnol|spanish)\b',
                r'\bamazigh\b'
            ]
            
            # Look for language section
            language_section = self._find_section(text, ["langues", "languages", "compétences linguistiques"])
            
            if language_section:
                for pattern in language_patterns:
                    try:
                        matches = re.findall(pattern, language_section, re.IGNORECASE)
                        languages.extend(matches)
                    except:
                        continue
            else:
                # If no dedicated section, look throughout the document
                for pattern in language_patterns:
                    try:
                        matches = re.findall(pattern, text, re.IGNORECASE)
                        languages.extend(matches)
                    except:
                        continue
                        
            # If no languages found, add a default based on the document language
            if not languages:
                if self.language == "fr":
                    languages = ["français"]
                elif self.language == "ar":
                    languages = ["arabe"]
                elif self.language == "en":
                    languages = ["anglais"]
        except Exception as e:
            print(f"Warning: Error in language extraction: {e}")
            # Return a default language based on the document's specified language
            if self.language == "fr":
                return ["français"]
            elif self.language == "ar":
                return ["arabe"]
            elif self.language == "en":
                return ["anglais"]
        
        return list(set(languages))
    
    def _find_section(self, text, headers):
        """Find a specific section in the CV text based on headers"""
        lines = text.split('\n')
        section_start = -1
        
        # Find the section start based on headers
        for i, line in enumerate(lines):
            if any(header.lower() in line.lower() for header in headers):
                section_start = i
                break
        
        if section_start == -1:
            return ""
        
        # Find the section end (next header or end of text)
        section_end = len(lines)
        for i in range(section_start + 1, len(lines)):
            # Check if this line could be a new section header
            if re.match(r'^[A-Z\s]{2,}$', lines[i].strip()) or any(re.search(r'\b' + header + r'\b', lines[i], re.IGNORECASE) for header in self.education_headers + self.experience_headers + self.skills_headers):
                if i > section_start + 1:  # Ensure we've captured some content
                    section_end = i
                    break
        
        # Extract the section content
        section_content = '\n'.join(lines[section_start:section_end])
        return section_content


def create_skills_taxonomy(domain_specific=True):
    """Create a skills taxonomy JSON file for Moroccan job market"""
    skills = {
        "technical": {
            "software_development": [
                "Python", "Java", "JavaScript", "PHP", "C++", "C#", ".NET",
                "React", "Angular", "Vue.js", "Node.js", "Django", "Flask",
                "SQL", "MongoDB", "PostgreSQL", "MySQL", "Oracle",
                "Git", "Docker", "Kubernetes", "Jenkins", "CI/CD",
                "HTML", "CSS", "SASS", "Bootstrap", "Tailwind CSS",
                "RESTful API", "GraphQL", "Microservices"
            ],
            "data_science": [
                "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
                "TensorFlow", "PyTorch", "Keras", "Scikit-learn",
                "Pandas", "NumPy", "Data Mining", "Statistical Analysis",
                "R", "SPSS", "SAS", "Tableau", "Power BI", "Data Visualization"
            ],
            "agriculture": [
                "Agribusiness", "Irrigation Systems", "Crop Management",
                "Sustainable Agriculture", "Agricultural Economics",
                "Soil Science", "Hydroponics", "Precision Agriculture"
            ],
            "tourism": [
                "Hospitality Management", "Tourism Marketing",
                "Event Planning", "Customer Service", "Food & Beverage",
                "Tour Guide", "Destination Management", "Hotel Operations"
            ],
            "renewable_energy": [
                "Solar Energy", "Wind Energy", "Hydroelectric Power",
                "Energy Storage", "Green Building", "Energy Efficiency",
                "Environmental Impact Assessment", "Sustainability"
            ]
        },
        "soft_skills": [
            "Communication", "Leadership", "Teamwork", "Problem Solving",
            "Critical Thinking", "Time Management", "Adaptability",
            "Creativity", "Emotional Intelligence", "Negotiation",
            "Conflict Resolution", "Decision Making", "Organization"
        ],
        "languages": [
            "Arabic", "French", "English", "Spanish", "Amazigh"
        ],
        "certifications": [
            "PMP", "PRINCE2", "ITIL", "Scrum Master", "AWS Certified",
            "Azure Certified", "Google Cloud Certified", "Cisco CCNA",
            "CompTIA A+", "TOGAF", "ISO", "Six Sigma", "ACCA", "CFA"
        ]
    }
    
    if domain_specific:
        # Add Morocco-specific skills and certifications
        morocco_specific = {
            "regional_expertise": [
                "Casablanca Finance City", "Tangier Med Port",
                "Marrakech Tourism", "Agadir Fishing Industry",
                "Rabat-Salé-Kénitra Industrial Zone",
                "Dakhla Renewable Energy Projects"
            ],
            "local_certifications": [
                "OFPPT Certification", "ISCAE Diploma", "EMI Engineering",
                "ENCG Management", "ANAPEC Training"
            ]
        }
        skills.update(morocco_specific)
    
    # Flatten the skills dictionary into a list for easier matching
    flat_skills = []
    for category in skills:
        if isinstance(skills[category], list):
            flat_skills.extend(skills[category])
        else:
            for subcategory in skills[category]:
                flat_skills.extend(skills[category][subcategory])
    
    return flat_skills


if __name__ == "__main__":
    # Example usage
    skills = create_skills_taxonomy(domain_specific=True)
    
    # Save skills to JSON file
    with open("models/nlp/skills_taxonomy.json", "w", encoding="utf-8") as f:
        json.dump(skills, f, ensure_ascii=False, indent=4)
    
    # Create parser instance
    parser = CVParser(language="fr", skills_file="models/nlp/skills_taxonomy.json")
    
    # Example: Extract information from a CV
    # cv_info = parser.extract_from_file("path/to/cv.pdf")
    # print(json.dumps(cv_info, indent=4, ensure_ascii=False)) 