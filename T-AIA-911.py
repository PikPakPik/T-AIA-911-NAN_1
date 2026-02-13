#!/usr/bin/env python3
"""
T-AIA-911 - Extraction NER de lieux (LOC) et pathfinding.

Point d'entrée principal pour :
- extraction NER depuis fichier/URL/texte ;
- mode chat terminal avec pathfinding ;
- lancement de l'UI Streamlit.
"""

import argparse
import csv
import os
import sys
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urlparse
from urllib.request import urlopen

import torch
from transformers import pipeline

# Chemins du projet
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset"
GEO_CSV_PATHS = [
    DATASET_DIR / "geo.csv",
    DATASET_DIR / "old" / "geo.csv",
]
VALENTIN_PATH = PROJECT_ROOT / "model" / "VALENTIN"

# Contexte recette / pâtisserie : ne pas interpréter Paris-Brest, religieuse, etc. comme trajet
RECIPE_CUES = (
    "recette", "cuisine", "pâtisserie", "dessert", "gâteau", "gateau",
    "un paris-brest", "une religieuse", "un monaco", "boire un monaco",
)


def _get_device() -> int | torch.device:
    """Détecte le meilleur device disponible (CUDA, MPS, CPU)."""
    if torch.cuda.is_available():
        return 0
    if getattr(torch.backends, "mps", None) is not None:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
    return -1


def _resolve_geo_csv() -> Path:
    """Retourne le chemin vers geo.csv (dataset/geo.csv ou dataset/old/geo.csv)."""
    for p in GEO_CSV_PATHS:
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"Aucun geo.csv trouvé. Créez {GEO_CSV_PATHS[0]} ou utilisez {GEO_CSV_PATHS[1]}"
    )


def extract_locations(
    entities: List[dict],
    phrase: str = "",
) -> List[str]:
    """
    Extrait les entités LOC et fusionne les tokens consécutifs (ex. Tour + - + Eiffel → Tour-Eiffel).
    Si phrase est fournie, utilise les positions start/end pour récupérer le texte exact.
    """
    # Labels supportés : LOC (CamemBERT), ORIGIN/DEST (modèle YANN départ/arrivée)
    LOC_LABELS = ("B-LOC", "I-LOC", "B-ORIGIN", "I-ORIGIN", "B-DEST", "I-DEST", "ORIGIN", "DEST")
    loc_entities: List[Tuple[int, int, str, str]] = []
    for entity in entities:
        label = entity.get("entity_group", entity.get("label", "")) or ""
        label_upper = str(label).upper()
        if "LOC" in label_upper or label_upper in LOC_LABELS:
            start = entity.get("start", 0)
            end = entity.get("end", 0)
            word = entity.get("word", entity.get("text", "")).strip()
            if word:
                loc_entities.append((start, end, word, label_upper))

    if not loc_entities:
        return []

    loc_entities.sort(key=lambda x: x[0])

    # Type de base pour regroupement (ORIGIN/DEST ne se fusionnent jamais entre eux)
    def _base_type(lbl: str) -> str:
        if lbl in ("ORIGIN", "B-ORIGIN", "I-ORIGIN"):
            return "ORIGIN"
        if lbl in ("DEST", "B-DEST", "I-DEST"):
            return "DEST"
        return "LOC"

    # Fusionner les LOC consécutives de même type (ex. "Tour", "-", "Eiffel" → "Tour-Eiffel")
    merged: List[str] = []
    cur_start, cur_end, cur_words, cur_type = (
        loc_entities[0][0], loc_entities[0][1], [loc_entities[0][2]], _base_type(loc_entities[0][3])
    )

    for start, end, word, label in loc_entities[1:]:
        same_type = _base_type(label) == cur_type
        if same_type and start <= cur_end + 5:  # adjacent ou quelques caractères (espace, tiret)
            cur_end = max(cur_end, end)
            cur_words.append(word)
        else:
            if phrase and cur_start < len(phrase) and cur_end <= len(phrase):
                merged.append(phrase[cur_start:cur_end].strip())
            else:
                merged.append("".join(w.lstrip("##") for w in cur_words))
            cur_start, cur_end, cur_words, cur_type = start, end, [word], _base_type(label)

    if phrase and cur_start < len(phrase) and cur_end <= len(phrase):
        merged.append(phrase[cur_start:cur_end].strip())
    else:
        merged.append("".join(w.lstrip("##") for w in cur_words))
    return merged


def filter_non_travel_locs(phrase: str, locs: List[str]) -> List[str]:
    """
    Filtre les LOC quand la phrase parle de recette/cuisine (ex. Paris-Brest = pâtisserie).
    """
    if not locs:
        return locs
    phrase_lower = phrase.lower().strip()
    for cue in RECIPE_CUES:
        if cue in phrase_lower:
            if len(locs) >= 2 and "paris" in phrase_lower and "brest" in phrase_lower:
                return []
            if "monaco" in phrase_lower and any(l.lower() == "monaco" for l in locs):
                return [l for l in locs if l.lower() != "monaco"]
    return locs


def read_lines_from_source(source: str) -> List[str]:
    """Lit les lignes depuis un fichier ou une URL."""
    parsed = urlparse(source)
    if parsed.scheme in ("http", "https"):
        with urlopen(source) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            text = resp.read().decode(charset)
            return text.splitlines()

    path = Path(source)
    if not path.is_file():
        print(f"Fichier introuvable : {source}", file=sys.stderr)
        sys.exit(1)

    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    cleaned_lines: List[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            cleaned_lines.append("")
            continue
        if "," in line:
            parts = line.split(",", 1)
            if len(parts) == 2 and parts[0].strip().isdigit():
                cleaned_lines.append(parts[1].strip())
            else:
                cleaned_lines.append(line)
        else:
            cleaned_lines.append(line)
    return cleaned_lines


def run_file_or_url(
    source: str,
    output_path: Path,
    ner_pipeline,
) -> None:
    """Traite un fichier ou une URL et écrit le CSV."""
    lines = read_lines_from_source(source)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        for idx, line in enumerate(lines, 1):
            phrase = line.strip()
            if not phrase:
                writer.writerow([idx, "INVALID", ""])
                continue

            try:
                entities = ner_pipeline(phrase)
            except Exception as e:
                print(f"Erreur ligne {idx} : {e}", file=sys.stderr)
                writer.writerow([idx, "INVALID", ""])
                continue

            lieux = extract_locations(entities, phrase)
            departure = lieux[0] if len(lieux) > 0 else "INVALID"
            arrival = lieux[-1] if len(lieux) > 1 else "INVALID"
            steps = lieux[1:-1] if len(lieux) > 2 else []
            writer.writerow([idx, departure] + steps + [arrival])

    print(f"Extraction terminée. {len(lines)} entrées → {output_path}")


def run_chat(ner_pipeline, model_name: str) -> None:
    """Mode chat interactif avec pathfinding."""
    if str(VALENTIN_PATH) not in sys.path:
        sys.path.insert(0, str(VALENTIN_PATH))
    from pathfinding import PathFinder

    geo_path = _resolve_geo_csv()
    pathfinder = PathFinder(str(geo_path))

    print("Modèle prêt. Envoie une phrase pour extraire les lieux (LOC).")
    print("Commande : quit / exit / q  pour quitter.\n")

    while True:
        try:
            phrase = input("Toi > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nInterruption du programme.")
            break
        if not phrase:
            continue
        if phrase.lower() in ("quit", "exit", "q"):
            print("Arrêt du programme.")
            break

        try:
            entities = ner_pipeline(phrase)
            locs = extract_locations(entities, phrase)
            locs = filter_non_travel_locs(phrase, locs)
            if locs:
                print(f"LOC  > {locs}")
                if len(locs) >= 2:
                    print(f"      (départ : {locs[0]}, arrivée : {locs[-1]})")
                    route = pathfinder.find_shortest_path(
                        locs[0], locs[-1], locs[1:-1]
                    )
                    print(f"      Route optimisée : {route}")
        except Exception as e:
            print(f"Erreur lors de l'extraction des entités : {e}")


def run_text(phrase: str, ner_pipeline) -> None:
    """Traite un texte direct en ligne de commande."""
    if not phrase:
        print("Erreur : texte vide.", file=sys.stderr)
        sys.exit(1)

    entities = ner_pipeline(phrase)
    locs = extract_locations(entities, phrase)
    locs = filter_non_travel_locs(phrase, locs)

    if locs:
        print(f"Phrase : {phrase}")
        print(f"LOC : {locs}")
        if len(locs) >= 2:
            print(f"Départ : {locs[0]}, Arrivée : {locs[-1]}")
            if len(locs) > 2:
                print(f"Étapes : {locs[1:-1]}")
    else:
        print(f"Phrase : {phrase}")
        print("Aucun lieu détecté.")


def run_ui() -> None:
    """Lance l'UI Streamlit de pathfinding."""
    ui_script = PROJECT_ROOT / "model" / "ALEXANDRE" / "pathfinding" / "chat_pathfinding_app.py"
    if not ui_script.is_file():
        print(f"Script UI introuvable : {ui_script}", file=sys.stderr)
        sys.exit(1)
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(ui_script)],
            check=True,
        )
    except FileNotFoundError:
        print("Streamlit non installé. Installez avec : pip install streamlit", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Échec du lancement de l'UI : {e}", file=sys.stderr)
        sys.exit(e.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extraction d'entités LOC depuis un texte (RoBERTa/CamemBERT)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", help="Chemin vers le fichier texte")
    group.add_argument("--url", help="URL du fichier texte (une phrase par ligne)")
    group.add_argument("--text", help="Texte direct (une phrase)")
    group.add_argument(
        "--chat",
        help="Mode chat interactif (tapez les phrases une par une, 'exit' pour quitter)",
        action="store_true",
    )
    group.add_argument("--ui", help="Lancer l'UI Streamlit de pathfinding", action="store_true")

    parser.add_argument("--output", help="Chemin du fichier CSV de sortie", default=None)
    parser.add_argument(
        "--model",
        help="Modèle NER (RoBERTa/CamemBERT). Ex: Jean-Baptiste/camembert-ner, model/YANN/best_model, ./roberta_ner_model",
        default="Jean-Baptiste/camembert-ner",
    )
    args = parser.parse_args()

    if args.ui:
        run_ui()
        return

    if args.output and not (args.file or args.url):
        parser.error("--output ne peut être utilisé qu'avec --file ou --url")

    output_path: Optional[Path] = None
    if args.file or args.url:
        output_path = Path(args.output) if args.output else DATASET_DIR / "dataset_output_roberta.csv"
        if not output_path.is_absolute():
            output_path = DATASET_DIR / output_path.name

    # Device
    device = _get_device()
    if isinstance(device, torch.device):
        print("Device : MPS (Apple Silicon GPU)")
    elif device == 0:
        print("Device : CUDA")
    else:
        print("Device : CPU")

    print(f"Chargement du modèle : {args.model}...")
    try:
        ner_pipeline = pipeline(
            "ner",
            model=args.model,
            tokenizer=args.model,
            aggregation_strategy="simple",
            device=device,
        )
        print("Modèle chargé avec succès.")
    except Exception as e:
        err_msg = str(e)
        if "endswith" in err_msg or "NoneType" in err_msg:
            print(
                "Bug tokenizer CamemBERT détecté. Essayez : pip install -U transformers sentencepiece",
                file=sys.stderr,
            )
        print(f"Attention : impossible de charger le modèle NER : {e}", file=sys.stderr)
        raise

    if args.file or args.url:
        run_file_or_url(
            args.file or args.url,
            output_path,
            ner_pipeline,
        )
    elif args.chat:
        run_chat(ner_pipeline, args.model)
    elif args.text:
        run_text(args.text.strip(), ner_pipeline)


if __name__ == "__main__":
    main()
