import os
import soundfile as sf
from speechbrain.inference import EncoderClassifier
import torchaudio
import torch
import numpy as np

# Initialiser SpeechBrain pour le pré-traitement (modèle d'extraction de caractéristiques)
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    savedir="models/audio/speechbrain"
)

# Définir les chemins
input_dir = "/Users/abderrahim_boussyf/interview_analyzer/data/audio/selected"  # Ajustez si vous utilisez un autre sous-dossier (ex. ravdess/)
output_dir = "/Users/abderrahim_boussyf/interview_analyzer/data/processed/audio"
os.makedirs(output_dir, exist_ok=True)

# Fonction de pré-traitement
def preprocess_audio(audio_path, output_path, target_sr=16000):
    # Charger l'audio
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform is None:
        print(f"Erreur : Impossible de charger {audio_path}")
        return

    # Convertir en mono si stéréo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resampler à 16 kHz (standard pour SpeechBrain)
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
        waveform = resampler(waveform)

    # Normaliser l'amplitude
    waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)

    # Convertir en numpy pour sauvegarde
    waveform_np = waveform.squeeze().numpy()

    # Sauvegarder l'audio pré-traité
    sf.write(output_path, waveform_np, target_sr)
    print(f"Audio traité : {output_path}")

# Traiter tous les fichiers audio dans le dossier d'entrée
for filename in os.listdir(input_dir):
    if filename.lower().endswith(".wav"):  # Filtrer les fichiers WAV
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"processed_{filename}")
        preprocess_audio(input_path, output_path)

# Pas de "close()" nécessaire pour SpeechBrain ici, car on ne fait que pré-traiter