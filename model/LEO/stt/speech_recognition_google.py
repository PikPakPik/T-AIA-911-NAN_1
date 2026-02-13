import speech_recognition as sr
import sys
import os

def transcribe_audio(audio_file_path):
    """
    Transcrit un fichier audio en texte et retourne le texte.
    Retourne None en cas d'erreur.
    """
    # Init
    r = sr.Recognizer()

    # Check if the audio file exists
    if not os.path.exists(audio_file_path):
        print(f"Erreur : Le fichier '{audio_file_path}' n'a pas été trouvé.")
        return None

    with sr.AudioFile(audio_file_path) as source:
        # Load audio file
        audio_data = r.record(source)
        
        try:
            # Send to google API (free, internet required)
            # le problème c'est que ça ne marche pas avec les fichiers de plus de 1 minute, il faudrait les découper en morceaux de 1 minute
            texte = r.recognize_google(audio_data, language="fr-FR")
            return texte
        except sr.UnknownValueError:
            print(f"L'audio du fichier '{os.path.basename(audio_file_path)}' n'a pas été compris.")
            return None
        except sr.RequestError as e:
            print(f"Erreur de connexion à Google pour le fichier '{os.path.basename(audio_file_path)}': {e}")
            return None

if __name__ == "__main__":
    audio_directory = "./wav/"

    # Créer un tableau des fichiers .wav dans le dossier
    try:
        wav_files = [f for f in os.listdir(audio_directory) if f.endswith(".wav")]
    except FileNotFoundError:
        print(f"Erreur : Le dossier '{audio_directory}' n'existe pas.")
        sys.exit(1)

    if not wav_files:
        print(f"Aucun fichier .wav trouvé dans le dossier '{audio_directory}'.")
    else:
        print(f"\n Fichiers .wav trouvés : {wav_files}\n")
        
        # Boucler sur chaque fichier pour le transcrire
        for filename in wav_files:
            full_path = os.path.join(audio_directory, filename)
            print(f"--- Transcription de '{filename}' ---")
            
            transcription = transcribe_audio(full_path)
            
            if transcription:
                print(transcription, "\n")
            else:
                print("Transcription échouée.\n")