import os
import sys
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit, 
    QFileDialog, QGroupBox, QGridLayout, QSplitter, QFrame,
    QMessageBox, QScrollArea
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QColor

class CVAnalysisTab(QWidget):
    """CV Analysis tab for the PyQt Interview Analyzer application"""
    
    # Signals
    analysis_completed = pyqtSignal(bool)  # Signal to indicate CV analysis completion
    
    def __init__(self, backend, parent=None):
        super().__init__(parent)
        self.backend = backend
        self.main_app = parent
        self.cv_path = ""
        self.analysis_results = None
        self.analysis_complete = False
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the CV Analysis tab UI"""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Create horizontal splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel for CV upload and preview
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # CV Upload section
        upload_group = QGroupBox("CV Upload")
        upload_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        upload_layout = QVBoxLayout(upload_group)
        
        # Upload button and file path display
        self.upload_button = QPushButton("Upload CV")
        self.upload_button.setStyleSheet(
            "QPushButton { background-color: #007BFF; color: white; padding: 8px; }"
            "QPushButton:hover { background-color: #0056b3; }"
        )
        self.upload_button.clicked.connect(self.upload_cv)
        
        self.file_path_label = QLabel("No file selected")
        self.file_path_label.setStyleSheet("color: #6c757d;")
        
        upload_layout.addWidget(self.upload_button)
        upload_layout.addWidget(self.file_path_label)
        
        # CV Preview section
        preview_group = QGroupBox("CV Preview")
        preview_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        preview_layout = QVBoxLayout(preview_group)
        
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setPlaceholderText("CV content will be displayed here")
        
        preview_layout.addWidget(self.preview_text)
        
        # Action buttons
        action_layout = QHBoxLayout()
        
        self.analyze_button = QPushButton("ANALYZE CV")
        self.analyze_button.setStyleSheet(
            "QPushButton { background-color: #007BFF; color: white; padding: 10px; font-weight: bold; }"
            "QPushButton:hover { background-color: #0056b3; }"
            "QPushButton:disabled { background-color: #cccccc; }"
        )
        self.analyze_button.clicked.connect(self.analyze_cv)
        self.analyze_button.setEnabled(False)
        
        self.proceed_button = QPushButton("PROCEED TO INTERVIEW")
        self.proceed_button.setStyleSheet(
            "QPushButton { background-color: #28A745; color: white; padding: 10px; font-weight: bold; }"
            "QPushButton:hover { background-color: #218838; }"
            "QPushButton:disabled { background-color: #cccccc; }"
        )
        self.proceed_button.clicked.connect(self.proceed_to_interview)
        self.proceed_button.setEnabled(False)
        
        action_layout.addWidget(self.analyze_button)
        action_layout.addWidget(self.proceed_button)
        
        # Add all sections to left panel
        left_layout.addWidget(upload_group)
        left_layout.addWidget(preview_group)
        left_layout.addLayout(action_layout)
        
        # Right panel for analysis results
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Results section
        results_group = QGroupBox("CV Analysis Results")
        results_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        
        # Create a scroll area for the results
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        # Create content widget for the scroll area
        results_content = QWidget()
        results_layout = QVBoxLayout(results_content)
        
        # Personal info section
        personal_info_frame = QFrame()
        personal_info_frame.setFrameShape(QFrame.StyledPanel)
        personal_info_frame.setStyleSheet("QFrame { background-color: #f8f9fa; border-radius: 5px; }")
        personal_info_layout = QVBoxLayout(personal_info_frame)
        
        personal_info_title = QLabel("Personal Information")
        personal_info_title.setStyleSheet("font-weight: bold; color: #007BFF;")
        personal_info_layout.addWidget(personal_info_title)
        
        personal_info_grid = QGridLayout()
        
        labels = ["Name:", "Email:", "Phone:"]
        self.info_values = {}
        
        for i, label_text in enumerate(labels):
            label = QLabel(label_text)
            label.setStyleSheet("font-weight: bold; color: #495057;")
            value = QLabel("")
            value.setStyleSheet("color: #212529;")
            
            personal_info_grid.addWidget(label, i, 0)
            personal_info_grid.addWidget(value, i, 1)
            
            self.info_values[label_text] = value
        
        personal_info_layout.addLayout(personal_info_grid)
        results_layout.addWidget(personal_info_frame)
        
        # Skills section
        skills_frame = QFrame()
        skills_frame.setFrameShape(QFrame.StyledPanel)
        skills_frame.setStyleSheet("QFrame { background-color: #f8f9fa; border-radius: 5px; }")
        skills_layout = QVBoxLayout(skills_frame)
        
        skills_title = QLabel("Skills Analysis")
        skills_title.setStyleSheet("font-weight: bold; color: #007BFF;")
        skills_layout.addWidget(skills_title)
        
        tech_skills_label = QLabel("Technical Skills:")
        tech_skills_label.setStyleSheet("font-weight: bold; color: #495057;")
        skills_layout.addWidget(tech_skills_label)
        
        self.tech_skills_text = QTextEdit()
        self.tech_skills_text.setReadOnly(True)
        self.tech_skills_text.setMaximumHeight(100)
        skills_layout.addWidget(self.tech_skills_text)
        
        soft_skills_label = QLabel("Soft Skills:")
        soft_skills_label.setStyleSheet("font-weight: bold; color: #495057;")
        skills_layout.addWidget(soft_skills_label)
        
        self.soft_skills_text = QTextEdit()
        self.soft_skills_text.setReadOnly(True)
        self.soft_skills_text.setMaximumHeight(100)
        skills_layout.addWidget(self.soft_skills_text)
        
        results_layout.addWidget(skills_frame)
        
        # Career matches section
        career_frame = QFrame()
        career_frame.setFrameShape(QFrame.StyledPanel)
        career_frame.setStyleSheet("QFrame { background-color: #f8f9fa; border-radius: 5px; }")
        career_layout = QVBoxLayout(career_frame)
        
        career_title = QLabel("Career Matches")
        career_title.setStyleSheet("font-weight: bold; color: #007BFF;")
        career_layout.addWidget(career_title)
        
        self.career_text = QTextEdit()
        self.career_text.setReadOnly(True)
        career_layout.addWidget(self.career_text)
        
        results_layout.addWidget(career_frame)
        
        # Recommendations section
        recommendations_frame = QFrame()
        recommendations_frame.setFrameShape(QFrame.StyledPanel)
        recommendations_frame.setStyleSheet("QFrame { background-color: #f8f9fa; border-radius: 5px; }")
        recommendations_layout = QVBoxLayout(recommendations_frame)
        
        recommendations_title = QLabel("Recommendations")
        recommendations_title.setStyleSheet("font-weight: bold; color: #007BFF;")
        recommendations_layout.addWidget(recommendations_title)
        
        self.recommendations_text = QTextEdit()
        self.recommendations_text.setReadOnly(True)
        recommendations_layout.addWidget(self.recommendations_text)
        
        results_layout.addWidget(recommendations_frame)
        
        # Save button
        save_layout = QHBoxLayout()
        save_layout.addStretch()
        
        self.save_button = QPushButton("SAVE ANALYSIS")
        self.save_button.setStyleSheet(
            "QPushButton { background-color: #007BFF; color: white; padding: 8px; font-weight: bold; }"
            "QPushButton:hover { background-color: #0056b3; }"
            "QPushButton:disabled { background-color: #cccccc; }"
        )
        self.save_button.clicked.connect(self.save_cv_analysis)
        self.save_button.setEnabled(False)
        
        save_layout.addWidget(self.save_button)
        
        # Add the scroll area to the results layout
        scroll_area.setWidget(results_content)
        results_group_layout = QVBoxLayout(results_group)
        results_group_layout.addWidget(scroll_area)
        
        # Add all sections to right panel
        right_layout.addWidget(results_group)
        right_layout.addLayout(save_layout)
        
        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([int(self.width() * 0.4), int(self.width() * 0.6)])
        
        # Add splitter to main layout
        main_layout.addWidget(splitter)
    
    def upload_cv(self):
        """Handle the upload CV button click"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select CV File", "", 
            "CV Files (*.pdf *.docx *.doc *.txt);;All Files (*)", 
            options=options
        )
        
        if file_path:
            self.cv_path = file_path
            self.file_path_label.setText(os.path.basename(file_path))
            self.preview_cv()
            self.analyze_button.setEnabled(True)
            
            if self.main_app:
                self.main_app.status_bar.showMessage(f"CV file loaded: {os.path.basename(file_path)}")
    
    def preview_cv(self):
        """Preview the selected CV file"""
        if not self.cv_path:
            return
            
        self.preview_text.setPlainText("Loading CV preview...")
        
        try:
            # Use backend to extract text
            cv_text = self.backend.extract_text_from_file(self.cv_path)
            
            if cv_text:
                self.preview_text.setPlainText(cv_text)
            else:
                self.preview_text.setPlainText("Could not extract text from this file format.")
        except Exception as e:
            self.preview_text.setPlainText(f"Error loading CV: {str(e)}")
    
    def analyze_cv(self):
        """Analyze the CV file"""
        if not self.cv_path:
            return
            
        self.preview_text.append("\n\nAnalyzing CV, please wait...")
        
        if self.main_app:
            self.main_app.status_bar.showMessage("Analyzing CV...")
            
        try:
            # Use backend to analyze the CV
            results = self.backend.analyze_cv(self.cv_path)
            
            if results:
                self.display_analysis_results(results)
            else:
                QMessageBox.warning(self, "Analysis Failed", 
                                   "Could not analyze the CV. Please try a different file.")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error analyzing CV: {str(e)}")
            if self.main_app:
                self.main_app.status_bar.showMessage("CV analysis failed")
    
    def display_analysis_results(self, results):
        """Display the CV analysis results"""
        self.analysis_results = results
        
        # Update personal info
        personal_info = results.get("candidate_info", {})
        self.info_values["Name:"].setText(personal_info.get("name", ""))
        self.info_values["Email:"].setText(personal_info.get("email", ""))
        self.info_values["Phone:"].setText(personal_info.get("phone", ""))
        
        # Update skills
        skills = results.get("skills", [])
        tech_skills = []
        soft_skills = []
        
        for skill in skills:
            if isinstance(skill, dict):
                skill_name = skill.get("name", "")
                skill_type = skill.get("type", "").lower()
                
                if skill_type == "technical":
                    tech_skills.append(skill_name)
                elif skill_type == "soft":
                    soft_skills.append(skill_name)
            elif isinstance(skill, str):
                # Default to technical skill if type not specified
                tech_skills.append(skill)
        
        self.tech_skills_text.setPlainText("• " + "\n• ".join(tech_skills) if tech_skills else "No technical skills found")
        self.soft_skills_text.setPlainText("• " + "\n• ".join(soft_skills) if soft_skills else "No soft skills found")
        
        # Update career matches
        career_matches = results.get("career_matches", [])
        career_text = ""
        
        for i, career in enumerate(career_matches):
            if isinstance(career, dict):
                career_name = career.get("career", "")
                match_percentage = career.get("match_percentage", 0)
                career_text += f"• {career_name} - {match_percentage}% match\n"
            elif isinstance(career, str):
                career_text += f"• {career}\n"
        
        self.career_text.setPlainText(career_text if career_text else "No career matches found")
        
        # Update recommendations
        recommendations = results.get("recommendations", [])
        recommendations_text = ""
        
        for recommendation in recommendations:
            if isinstance(recommendation, dict):
                rec_type = recommendation.get("type", "")
                options = recommendation.get("options", [])
                skills = recommendation.get("skills", [])
                
                if rec_type and options:
                    recommendations_text += f"• {rec_type.replace('_', ' ').title()}:\n"
                    for option in options:
                        if isinstance(option, dict):
                            opt_text = option.get("career", "")
                            opt_match = option.get("match_percentage", "")
                            if opt_text:
                                if opt_match:
                                    recommendations_text += f"  - {opt_text} ({opt_match}%)\n"
                                else:
                                    recommendations_text += f"  - {opt_text}\n"
                        elif isinstance(option, str):
                            recommendations_text += f"  - {option}\n"
                
                if rec_type and skills:
                    recommendations_text += f"• Recommended skills to develop:\n"
                    for skill in skills:
                        recommendations_text += f"  - {skill}\n"
                            
            elif isinstance(recommendation, str):
                recommendations_text += f"• {recommendation}\n"
        
        self.recommendations_text.setPlainText(recommendations_text if recommendations_text else "No recommendations available")
        
        # Enable buttons
        self.proceed_button.setEnabled(True)
        self.save_button.setEnabled(True)
        
        # Mark analysis as complete
        self.analysis_complete = True
        
        # Signal to main app
        if self.main_app:
            self.main_app.set_cv_analysis_complete(True)
            self.main_app.status_bar.showMessage("CV analysis complete. Ready to proceed to interview.")
        
    def proceed_to_interview(self):
        """Proceed to the interview tab"""
        if not self.analysis_complete:
            QMessageBox.warning(self, "Analysis Required", 
                               "Please complete CV analysis first")
            return
            
        # Switch to interview tab
        if self.main_app and hasattr(self.main_app, 'tab_widget'):
            self.main_app.tab_widget.setCurrentIndex(1)
    
    def save_cv_analysis(self):
        """Save the CV analysis results to a file"""
        if not self.analysis_complete or not self.analysis_results:
            return
            
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save CV Analysis", "", 
            "Text Files (*.txt);;All Files (*)", 
            options=options
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write("CV ANALYSIS REPORT\n")
                    f.write("=================\n\n")
                    
                    # Write personal info
                    f.write("PERSONAL INFORMATION\n")
                    f.write(f"Name: {self.info_values['Name:'].text()}\n")
                    f.write(f"Email: {self.info_values['Email:'].text()}\n")
                    f.write(f"Phone: {self.info_values['Phone:'].text()}\n\n")
                    
                    # Write skills
                    f.write("TECHNICAL SKILLS\n")
                    f.write(f"{self.tech_skills_text.toPlainText()}\n\n")
                    
                    f.write("SOFT SKILLS\n")
                    f.write(f"{self.soft_skills_text.toPlainText()}\n\n")
                    
                    # Write career matches
                    f.write("CAREER MATCHES\n")
                    f.write(f"{self.career_text.toPlainText()}\n\n")
                    
                    # Write recommendations
                    f.write("RECOMMENDATIONS\n")
                    f.write(f"{self.recommendations_text.toPlainText()}\n")
                
                if self.main_app:
                    self.main_app.status_bar.showMessage(f"CV analysis saved to {os.path.basename(file_path)}")
                    
                QMessageBox.information(self, "Save Successful", 
                                       f"CV analysis saved to {file_path}")
                    
            except Exception as e:
                QMessageBox.critical(self, "Save Error", 
                                    f"Error saving CV analysis: {str(e)}") 