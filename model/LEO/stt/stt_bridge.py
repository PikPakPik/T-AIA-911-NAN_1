import os
import subprocess
import sys
import tempfile
import importlib.util
from pathlib import Path
from typing import Optional, Tuple

def _load_audio_transcribe():
    module_path = Path(__file__).resolve().parent / "audio_transcribe.py"
    spec = importlib.util.spec_from_file_location("leo_audio_transcribe", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Impossible de charger audio_transcribe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def transcribe_wav_bytes(
    audio_bytes: bytes,
    language: str = "fr-FR",
) -> Tuple[Optional[str], Optional[str]]:
    """
    Transcrit des bytes WAV en deleguant a audio_transcribe.transcribe_audio.
    `language` est accepte pour compatibilite d'interface.
    Retourne (texte, erreur).
    """
    if not audio_bytes:
        return None, "Aucun audio fourni."

    tmp_path = None
    try:
        audio_transcribe_module = _load_audio_transcribe()
        transcribe_audio = getattr(audio_transcribe_module, "transcribe_audio", None)
        if transcribe_audio is None:
            return None, "Fonction `transcribe_audio` absente de audio_transcribe.py."
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        text = transcribe_audio(tmp_path)
        if not text:
            return None, "Transcription echouee."
        return text, None
    except Exception as exc:
        return None, f"Erreur STT WAV: {exc}"
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def transcribe_microphone() -> Tuple[Optional[str], Optional[str]]:
    """
    Lance microphone_recognition.py et recupere la ligne Google Speech Recognition.
    Retourne (texte, erreur).
    """
    script_path = Path(__file__).resolve().parent / "microphone_recognition.py"
    if not script_path.is_file():
        return None, f"Fichier introuvable: {script_path}"

    try:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:
        return None, f"Impossible de lancer microphone_recognition.py: {exc}"

    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    output = output.strip()
    marker = "Google Speech Recognition thinks you said "
    for line in output.splitlines():
        if marker in line:
            text = line.split(marker, 1)[1].strip()
            if text:
                return text, None

    if output:
        return None, output
    return None, "Aucune transcription microphone recue."
