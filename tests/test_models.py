import os
import cv2
from transformers import pipeline
from deepface import DeepFace
import spacy
import re
from speechbrain.inference import EncoderClassifier
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import soundfile as sf
from scipy.io import wavfile
import warnings

# Chemins des données pré-traitées
facial_dir = "/Users/abderrahim_boussyf/interview_analyzer/data/processed/facial"
audio_dir = "/Users/abderrahim_boussyf/interview_analyzer/data/processed/audio"
cv_dir = "/Users/abderrahim_boussyf/interview_analyzer/data/processed/cv"

nlp_en = spacy.load("en_core_web_sm")
nlp_fr = spacy.load("fr_core_news_sm")


# 1. Tester DeepFace pour les données faciales (inchangé)
def test_facial():
    print("\n=== Test DeepFace (Facial) ===")
    for filename in os.listdir(facial_dir):
        if filename.endswith((".jpg", ".png")):
            image_path = os.path.join(facial_dir, filename)
            try:
                result = DeepFace.analyze(image_path, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']
                print(f"{filename} : Émotion prédite = {emotion}")
            except Exception as e:
                print(f"Erreur sur {filename} : {e}")

# 2. Tester Audio (Efficient Approach)
def test_audio_efficient():
    print("\n=== Test Audio (Efficient Approach) ===")
    
    warnings.filterwarnings('ignore')
    
    # Simple features extraction approach
    for filename in os.listdir(audio_dir):
        if filename.endswith(".wav"):
            audio_path = os.path.join(audio_dir, filename)
            try:
                # Try multiple methods to load the audio
                audio_loaded = False
                
                # Method 1: Try with soundfile
                try:
                    data, sample_rate = sf.read(audio_path)
                    audio_loaded = True
                    print(f"{filename}: Loaded with soundfile - SR={sample_rate}")
                except Exception as e:
                    print(f"{filename}: soundfile loading failed: {str(e)}")
                
                # Method 2: Try with scipy if method 1 failed
                if not audio_loaded:
                    try:
                        sample_rate, data = wavfile.read(audio_path)
                        # Convert to float
                        if data.dtype == np.int16:
                            data = data.astype(np.float32) / 32768.0
                        audio_loaded = True
                        print(f"{filename}: Loaded with scipy - SR={sample_rate}")
                    except Exception as e:
                        print(f"{filename}: scipy loading failed: {str(e)}")
                
                # If audio loaded successfully, extract features
                if audio_loaded:
                    # Basic feature: zero-crossing rate (very lightweight)
                    zero_crossings = np.sum(np.abs(np.diff(np.sign(data)))) / (2 * len(data))
                    
                    # Simple energy calculation
                    energy = np.sum(data**2) / len(data)
                    
                    # Mock emotion prediction
                    emotions = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
                    emotion_idx = min(int(zero_crossings * 10) % len(emotions), len(emotions) - 1)
                    print(f"{filename}: ZCR={zero_crossings:.4f}, Energy={energy:.6f}")
                    print(f"{filename}: Predicted emotion = {emotions[emotion_idx]}")
                else:
                    print(f"{filename}: Failed to load audio with all methods")
                    
            except Exception as e:
                print(f"Erreur sur {filename}: {str(e)}")

# 3. Tester DistilBERT pour les CVs (corrigé avec un modèle de classification)
def test_cv():
    print("\n=== Test CV : Extraction et Génération de Questions ===")
    generator = pipeline("text-generation", model="distilgpt2")

    for filename in os.listdir(cv_dir):
        if filename.endswith(".txt"):
            cv_path = os.path.join(cv_dir, filename)
            try:
                with open(cv_path, "r", encoding="utf-8") as f:
                    text = f.read()

                # Utiliser spaCy pour analyser le texte
                doc = nlp_en(text.lower())
                info = {"skills": set(), "experience": set(), "domain": ""}
                for ent in doc.ents:
                    if ent.label_ in ("SKILL", "NORP", "ORG") or "skill" in ent.text:
                        info["skills"].add(ent.text)
                    if "experience" in ent.text or "year" in ent.text:
                        info["experience"].add(ent.text)
                    if "domain" in ent.text or "specialty" in ent.text or "field" in ent.text:
                        info["domain"] = ent.text

                # Nettoyage des compétences/experiences bruitées
                info["skills"] = {s for s in info["skills"] if len(s.split()) <= 3 and "skill" not in s}
                info["experience"] = {e for e in info["experience"] if len(e) > 10}

                print(f"\nCV : {filename}")
                print(f"Compétences extraites : {list(info['skills'])[:5]}")
                print(f"Expériences extraites : {list(info['experience'])[:3]}")
                print(f"Domaine extrait : {info['domain']}")

                # Générer des questions avec des prompts améliorés
                questions = []
                for skill in list(info["skills"])[:3]:
                    prompt = f"Generate a clear and concise interview question about the skill '{skill}' in a professional context."
                    q = generator(prompt, max_length=50, num_return_sequences=1)[0]["generated_text"].replace(prompt, "").strip()
                    questions.append(q)
                if info["experience"]:
                    exp = list(info["experience"])[0]
                    prompt = f"Generate a specific interview question about the experience '{exp}' related to its impact."
                    q = generator(prompt, max_length=50, num_return_sequences=1)[0]["generated_text"].replace(prompt, "").strip()
                    questions.append(q)
                if info["domain"]:
                    prompt = f"Generate an interview question about working in the domain '{info['domain']}'."
                    q = generator(prompt, max_length=50, num_return_sequences=1)[0]["generated_text"].replace(prompt, "").strip()
                    questions.append(q)

                print("Questions générées :")
                for i, q in enumerate(questions, 1):
                    print(f"{i}. {q}")

            except Exception as e:
                print(f"Erreur sur {filename} : {e}")

# Exécuter les tests
if __name__ == "__main__":
    test_facial()
    test_audio_efficient()
    test_cv()