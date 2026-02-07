#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chat interactif : tu envoies une phrase, le modèle renvoie les LOC (lieux) détectés.
Usage: python3 chat_roberta_loc.py [--model ./roberta_ner_model]
"""

import argparse
import torch
from transformers import pipeline


def extract_locations(entities, phrase=""):
    """
    Extrait les entités LOC et fusionne les tokens consécutifs (ex. Tour + - + Effiel → Tour-Effiel).
    Si phrase est fournie, utilise les positions start/end pour récupérer le texte exact.
    """
    loc_entities = []
    for entity in entities:
        label = entity.get("entity_group", entity.get("label", ""))
        label_upper = label.upper()
        if "LOC" in label_upper or label_upper in ("B-LOC", "I-LOC"):
            start = entity.get("start", 0)
            end = entity.get("end", 0)
            word = entity.get("word", entity.get("text", "")).strip()
            if word:
                loc_entities.append((start, end, word))

    if not loc_entities:
        return []

    loc_entities.sort(key=lambda x: x[0])

    # Fusionner les LOC consécutives (ex. "Tour", "-", "Effiel" → "Tour-Effiel")
    merged = []
    cur_start, cur_end, cur_word = loc_entities[0]
    for start, end, word in loc_entities[1:]:
        if start <= cur_end + 5:  # adjacent ou quelques caractères (espace, tiret)
            cur_end = max(cur_end, end)
        else:
            if phrase and cur_start < len(phrase) and cur_end <= len(phrase):
                merged.append(phrase[cur_start:cur_end].strip())
            else:
                merged.append(cur_word)
            cur_start, cur_end, cur_word = start, end, word
    if phrase and cur_start < len(phrase) and cur_end <= len(phrase):
        merged.append(phrase[cur_start:cur_end].strip())
    else:
        merged.append(cur_word)
    return merged


# Contexte recette / pâtisserie : ne pas interpréter Paris-Brest, religieuse, etc. comme trajet
RECIPE_CUES = (
    "recette", "cuisine", "pâtisserie", "dessert", "gâteau", "gateau",
    "un paris-brest", "une religieuse", "un monaco", "boire un monaco",
)
def filter_non_travel_locs(phrase: str, locs: list[str]) -> list[str]:
    """
    Filtre les LOC quand la phrase parle de recette/cuisine (ex. Paris-Brest = pâtisserie).
    """
    if not locs:
        return locs
    phrase_lower = phrase.lower().strip()
    for cue in RECIPE_CUES:
        if cue in phrase_lower:
            # Paris-Brest (pâtisserie) : deux LOC Paris + Brest dans ce contexte → pas un trajet
            if len(locs) >= 2 and "paris" in phrase_lower and "brest" in phrase_lower:
                return []
            # "un monaco" / "boire un monaco" : ne pas retourner Monaco comme lieu
            if "monaco" in phrase_lower and ("monaco" in [l.lower() for l in locs]):
                return [l for l in locs if l.lower() != "monaco"]
    return locs


def main():
    parser = argparse.ArgumentParser(description="Chat : phrase → LOC détectés")
    parser.add_argument(
        "--model",
        default="./roberta_ner_model",
        help="Chemin du modèle NER (défaut: ./roberta_ner_model)",
    )
    args = parser.parse_args()

    # Device : MPS (Mac ARM64), CUDA, ou CPU
    def _get_device():
        if torch.cuda.is_available():
            return 0
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        return -1

    _device = _get_device()
    if isinstance(_device, torch.device):
        print("Device: MPS (Apple Silicon GPU)")
    elif _device == 0:
        print("Device: CUDA")
    else:
        print("Device: CPU")
    print(f"Chargement du modèle: {args.model}...")
    ner = pipeline(
        "ner",
        model=args.model,
        tokenizer=args.model,
        aggregation_strategy="simple",
        device=_device,
    )
    print("Modèle prêt. Envoie une phrase pour extraire les lieux (LOC).")
    print("  Commande: quit / exit / q  pour quitter.\n")

    while True:
        try:
            phrase = input("Toi > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not phrase:
            continue
        if phrase.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break

        try:
            entities = ner(phrase)
            locs = extract_locations(entities, phrase)
            locs = filter_non_travel_locs(phrase, locs)
            if locs:
                print(f"LOC  > {locs}")
                if len(locs) >= 2:
                    print(f"      (départ: {locs[0]}, arrivée: {locs[-1]})")
            else:
                if any(c in phrase.lower() for c in ("recette", "cuisine", "paris-brest", "boire un")):
                    print("LOC  > (aucun trajet — contexte recette/cuisine)")
                else:
                    print("LOC  > (aucun lieu détecté)")
        except Exception as e:
            print(f"LOC  > Erreur: {e}")

        print()


if __name__ == "__main__":
    main()
