import os
import PyPDF2  # Ensure PyPDF2 is installed using: pip install PyPDF2
import pytesseract
from PIL import Image
import cv2
import numpy as np

# Définir les chemins
input_dir = "/Users/abderrahim_boussyf/interview_analyzer/data/cv/selected"  # Ajustez si vous utilisez un autre sous-dossier
output_dir = "/Users/abderrahim_boussyf/interview_analyzer/data/processed/cv"
os.makedirs(output_dir, exist_ok=True)

# Fonction pour extraire le texte d'un PDF
def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text.strip()
    except Exception as e:
        print(f"Erreur lors de l'extraction du PDF {pdf_path} : {e}")
        return None

# Fonction pour extraire le texte d'une image (PNG/JPG)
def extract_text_from_image(image_path):
    try:
        # Charger l'image avec OpenCV
        image = cv2.imread(image_path)
        # Convertir en niveaux de gris
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Appliquer un seuillage pour améliorer la lisibilité
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        # Utiliser Tesseract pour extraire le texte
        text = pytesseract.image_to_string(thresh)
        return text.strip()
    except Exception as e:
        print(f"Erreur lors de l'extraction de l'image {image_path} : {e}")
        return None

# Fonction de pré-traitement
def preprocess_cv(file_path, output_path):
    filename = os.path.basename(file_path)
    if filename.lower().endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
    elif filename.lower().endswith((".png", ".jpg")):
        text = extract_text_from_image(file_path)
    else:
        print(f"Format non supporté : {file_path}")
        return

    if text:
        # Sauvegarder le texte extrait dans un fichier .txt
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"CV traité : {output_path}")
    else:
        print(f"Aucun texte extrait de {file_path}")

# Traiter tous les fichiers dans le dossier d'entrée
for filename in os.listdir(input_dir):
    if filename.lower().endswith((".pdf", ".png", ".jpg")):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"processed_{filename}.txt")
        preprocess_cv(input_path, output_path)