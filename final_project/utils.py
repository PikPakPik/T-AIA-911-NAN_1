"""
Fonctions utilitaires partagées entre les modules.
"""

import re
import unicodedata
from typing import Union

import torch


def get_device() -> Union[int, torch.device]:
    """Détecte le meilleur device disponible (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return 0
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return -1


def device_label(device: Union[int, torch.device]) -> str:
    """Nom lisible du device."""
    if isinstance(device, torch.device):
        return f"MPS (Apple Silicon GPU)"
    if device == 0:
        return "CUDA"
    return "CPU"


def format_duration(minutes: float) -> str:
    """Formate une durée en minutes en '2h35' ou '12min'."""
    h = int(minutes // 60)
    m = int(minutes % 60)
    if h > 0:
        return f"{h}h{m:02d}"
    return f"{m}min"


def normalize_text(text: str) -> str:
    """Normalise un texte : minuscules, sans accents, espaces condensés."""
    raw = text.lower().strip()
    no_accents = "".join(
        c for c in unicodedata.normalize("NFD", raw) if unicodedata.category(c) != "Mn"
    )
    return re.sub(r"\s+", " ", no_accents)
