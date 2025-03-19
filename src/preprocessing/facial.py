import cv2
import mediapipe as mp
import numpy as np
import os

# Initialiser Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,    # Mode pour images fixes
    max_num_faces=1,           # Détecter un seul visage par image
    min_detection_confidence=0.5  # Seuil de confiance minimal
)

# Définir les chemins
input_dir = "/Users/abderrahim_boussyf/interview_analyzer/data/facial/selected"  # Ajustez si vous utilisez un autre sous-dossier (ex. fer2013/train/)
output_dir = "/Users/abderrahim_boussyf/interview_analyzer/data/processed/facial"
os.makedirs(output_dir, exist_ok=True)

# Fonction de pré-traitement
def preprocess_image(image_path, output_path):
    # Charger l'image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erreur : Impossible de charger {image_path}")
        return

    # Convertir en RGB (Mediapipe utilise RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Détecter les repères faciaux
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        # Redimensionner à 64x64 (léger pour M1 et suffisant pour DeepFace)
        image_resized = cv2.resize(image, (64, 64))
        # Normaliser les valeurs entre 0 et 1
        image_normalized = image_resized / 255.0
        # Sauvegarder l'image traitée (reconverti en 0-255 pour stockage)
        cv2.imwrite(output_path, image_normalized * 255)
        print(f"Image traitée : {output_path}")
    else:
        print(f"Aucun visage détecté dans {image_path}")

# Traiter toutes les images dans le dossier d'entrée
for filename in os.listdir(input_dir):
    if filename.lower().endswith((".jpg", ".png")):  # Insensible à la casse
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"processed_{filename}")
        preprocess_image(input_path, output_path)

# Libérer les ressources
face_mesh.close()