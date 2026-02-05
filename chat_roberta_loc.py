#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chat interactif : tu envoies une phrase, le modèle renvoie les LOC (lieux) détectés.
Usage: python3 chat_roberta_loc.py [--model ./roberta_ner_model]
"""

import argparse
import os
import torch
from transformers import pipeline, AutoTokenizer, BertForTokenClassification


def extract_dep_arr(entities, phrase=""):
    """
    Extrait Départ et Arrivée depuis les entités B-DEP/I-DEP et B-ARR/I-ARR (modèle BERT Europeana).
    Retourne (départ, arrivée) ou (None, None) si non trouvés.
    """
    dep_words, arr_words = [], []
    for entity in entities:
        label = entity.get("entity_group", entity.get("label", ""))
        label_upper = label.upper()
        start = entity.get("start", 0)
        end = entity.get("end", 0)
        word = (entity.get("word", entity.get("text", "")) or "").strip()
        if "DEP" in label_upper:
            dep_words.append((start, end, word))
        elif "ARR" in label_upper:
            arr_words.append((start, end, word))
    dep_words.sort(key=lambda x: x[0])
    arr_words.sort(key=lambda x: x[0])
    dep_str = None
    arr_str = None
    if dep_words and phrase:
        dep_str = phrase[dep_words[0][0]:dep_words[-1][1]].strip()
    elif dep_words:
        dep_str = " ".join(w[2] for w in dep_words)
    if arr_words and phrase:
        arr_str = phrase[arr_words[0][0]:arr_words[-1][1]].strip()
    elif arr_words:
        arr_str = " ".join(w[2] for w in arr_words)
    return (dep_str, arr_str)


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
    # Modèle BERT Europeana : sauvegardé en TensorFlow (tf_model.h5), on charge en PyTorch avec from_tf=True
    model_path = os.path.abspath(args.model)
    tf_weights = os.path.join(model_path, "tf_model.h5")
    if os.path.isfile(tf_weights):
        # low_cpu_mem_usage=False évite les "meta tensors" qui empêchent .to(device) après from_tf=True
        model = BertForTokenClassification.from_pretrained(
            model_path, from_tf=True, low_cpu_mem_usage=False
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        ner = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=_device,
        )
        use_dep_arr = True  # labels B-DEP / B-ARR
    else:
        ner = pipeline(
            "ner",
            model=args.model,
            tokenizer=args.model,
            aggregation_strategy="simple",
            device=_device,
        )
        use_dep_arr = False
    print("Modèle prêt. Envoie une phrase pour extraire les lieux (LOC) ou Départ/Arrivée.")
    print("  Commande: quit / exit / q  pour quitter.\n")

    while True:
        try:
            phrase = input("Toi  > ").strip()
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
            if use_dep_arr:
                dep, arr = extract_dep_arr(entities, phrase)
                if dep or arr:
                    print(f"Départ  > {dep or '—'}")
                    print(f"Arrivée > {arr or '—'}")
                else:
                    print("Départ  > (aucun)")
                    print("Arrivée > (aucun)")
            else:
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
