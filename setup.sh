#!/bin/bash
# Script d'installation pour Interview Analyzer

echo "Installation des dépendances..."
pip install -r requirements.txt

echo "Création des répertoires nécessaires..."
mkdir -p data/{facial,audio,cv,processed}
mkdir -p models/{nlp,facial,audio}
mkdir -p results
mkdir -p temp

echo "Installation terminée. Vous pouvez maintenant exécuter:"
echo "python tests/enhanced_interview_analyzer.py"
