"""
PyQt5 Tab Components for Interview Analyzer Application

This package contains the tab components used in the PyQt5 version of the Interview Analyzer application.
"""

from .cv_analysis_tab import CVAnalysisTab
from .interview_tab import InterviewTab
from .mock_backend import MockAnalyzerBackend

__all__ = [
    'CVAnalysisTab', 
    'InterviewTab',
    'MockAnalyzerBackend'
] 