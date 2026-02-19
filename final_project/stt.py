"""
STT — Speech-to-Text bridge (optionnel).

Charge dynamiquement le module STT de Leo et expose des fonctions
de transcription pour fichier WAV et microphone.
"""

import importlib.util
from pathlib import Path
from typing import Optional, Tuple


def load_stt_module(repo_root: Path):
    """
    Charge dynamiquement le module stt_bridge.py de Leo.
    Retourne (module, None) en cas de succès, (None, message_erreur) sinon.
    """
    stt_path = repo_root / "model" / "LEO" / "stt" / "stt_bridge.py"
    if not stt_path.is_file():
        return None, f"Fichier STT introuvable : {stt_path}"
    try:
        spec = importlib.util.spec_from_file_location("leo_microphone_recognition", stt_path)
        if spec is None or spec.loader is None:
            return None, "Impossible de charger le module STT."
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, None
    except Exception as exc:
        return None, f"Erreur de chargement du module STT : {exc}"


def transcribe_wav_bytes(
    audio_bytes: bytes,
    stt_module,
    language: str = "fr-FR",
) -> Tuple[Optional[str], Optional[str]]:
    """
    Transcrit des bytes WAV en texte.
    Retourne (texte, None) ou (None, message_erreur).
    """
    if stt_module is None:
        return None, "Module STT indisponible."
    transcriber = getattr(stt_module, "transcribe_wav_bytes", None)
    if transcriber is None:
        return None, "Fonction `transcribe_wav_bytes` absente de stt_bridge.py."
    try:
        return transcriber(audio_bytes, language=language)
    except Exception as exc:
        return None, f"Erreur pendant la transcription WAV : {exc}"


def transcribe_microphone(
    stt_module,
    language: str = "fr-FR",
) -> Tuple[Optional[str], Optional[str]]:
    """
    Transcrit depuis le microphone du serveur.
    Retourne (texte, None) ou (None, message_erreur).
    """
    if stt_module is None:
        return None, "Module STT indisponible."
    transcriber = getattr(stt_module, "transcribe_microphone", None)
    if transcriber is None:
        return None, "Fonction `transcribe_microphone` absente de stt_bridge.py."
    try:
        return transcriber(language=language)
    except Exception as exc:
        return None, f"Erreur pendant la transcription microphone : {exc}"
