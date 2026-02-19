#!/usr/bin/env python3
"""
CLI — Point d'entrée en ligne de commande.

Usage :
    # Extraction NER depuis un fichier (format sujet : sentenceID,sentence)
    python -m final_project.cli --file input.txt --output results.csv

    # Extraction NER depuis une URL
    python -m final_project.cli --url https://example.com/sentences.txt

    # Extraction NER sur un texte direct
    python -m final_project.cli --text "Je veux aller de Paris à Lyon"

    # Mode chat terminal (NER + pathfinding)
    python -m final_project.cli --chat

    # Lancer l'UI Streamlit
    python -m final_project.cli --ui

Options :
    --model   Modèle NER (nom HF ou chemin local). Défaut : PikPak44/yann_best_model
    --output  Chemin du CSV de sortie (avec --file ou --url)
"""

import argparse
import csv
import sys
import subprocess
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse
from urllib.request import urlopen

from utils import get_device, device_label, format_duration
from ner import (
    HF_HUB_AVAILABLE,
    download_hf_model,
    load_ner,
    extract_cities,
    extract_locations_from_ner,
    extract_dep_arr_from_ner,
    normalize_locations,
    filter_non_travel_locs,
)

# Chemins du projet
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
GTFS_DIR = DATASET_DIR / "sncf"
GEO_CSV = DATASET_DIR / "geo.csv"

DEFAULT_HF_MODEL = "PikPak44/yann_best_model"


# ─── Lecture d'entrée ─────────────────────────────────────────────────────────

def read_lines_from_source(source: str) -> List[str]:
    """Lit les lignes depuis un fichier local ou une URL."""
    parsed = urlparse(source)
    if parsed.scheme in ("http", "https"):
        with urlopen(source) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            return resp.read().decode(charset).splitlines()

    path = Path(source)
    if not path.is_file():
        print(f"Fichier introuvable : {source}", file=sys.stderr)
        sys.exit(1)

    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    cleaned: List[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            cleaned.append("")
            continue
        if "," in line:
            parts = line.split(",", 1)
            if len(parts) == 2 and parts[0].strip().isdigit():
                cleaned.append(parts[1].strip())
            else:
                cleaned.append(line)
        else:
            cleaned.append(line)
    return cleaned


# ─── Extraction d'entités (format sujet) ─────────────────────────────────────

def extract_locations_for_cli(entities: list, phrase: str) -> List[str]:
    """
    Extrait les lieux depuis les entités NER.
    Compatible avec les labels LOC et DEP/ARR/ORIGIN/DEST.
    """
    # Labels supportés
    LOC_LABELS = ("B-LOC", "I-LOC", "B-ORIGIN", "I-ORIGIN", "B-DEST", "I-DEST", "ORIGIN", "DEST")
    loc_entities = []
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

    def _base_type(lbl: str) -> str:
        if lbl in ("ORIGIN", "B-ORIGIN", "I-ORIGIN"):
            return "ORIGIN"
        if lbl in ("DEST", "B-DEST", "I-DEST"):
            return "DEST"
        return "LOC"

    merged: List[str] = []
    cur_start, cur_end, cur_words, cur_type = (
        loc_entities[0][0], loc_entities[0][1], [loc_entities[0][2]], _base_type(loc_entities[0][3])
    )

    for start, end, word, label in loc_entities[1:]:
        same_type = _base_type(label) == cur_type
        if same_type and start <= cur_end + 5:
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


# ─── Modes d'exécution ───────────────────────────────────────────────────────

def run_file_or_url(source: str, output_path: Path, ner_pipeline) -> None:
    """
    Traite un fichier ou une URL et écrit le CSV de sortie.
    Format : sentenceID,Departure,[Steps...],Destination
    """
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

            lieux = extract_locations_for_cli(entities, phrase)
            departure = lieux[0] if len(lieux) > 0 else "INVALID"
            arrival = lieux[-1] if len(lieux) > 1 else "INVALID"
            steps = lieux[1:-1] if len(lieux) > 2 else []
            writer.writerow([idx, departure] + steps + [arrival])

    print(f"Extraction terminée. {len(lines)} entrées → {output_path}")


def run_text(phrase: str, ner_pipeline) -> None:
    """Traite un texte direct et affiche le résultat."""
    if not phrase:
        print("Erreur : texte vide.", file=sys.stderr)
        sys.exit(1)

    entities = ner_pipeline(phrase)
    lieux = extract_locations_for_cli(entities, phrase)
    lieux = filter_non_travel_locs(phrase, lieux)

    if lieux:
        print(f"Phrase : {phrase}")
        print(f"LOC    : {lieux}")
        if len(lieux) >= 2:
            print(f"Départ : {lieux[0]}")
            print(f"Arrivée: {lieux[-1]}")
            if len(lieux) > 2:
                print(f"Étapes : {lieux[1:-1]}")
    else:
        print(f"Phrase : {phrase}")
        print("Aucun lieu détecté.")


def run_chat(ner_bundle: dict) -> None:
    """Mode chat terminal avec NER + pathfinding SNCF."""
    from pathfinder import SNCFPathFinder

    pf = SNCFPathFinder(
        gtfs_dir=str(GTFS_DIR),
        geo_csv_path=str(GEO_CSV) if GEO_CSV.is_file() else None,
    )
    pf.build()

    print("Modèle NER prêt. Envoie une phrase pour trouver un itinéraire.")
    print("Commandes : quit / exit / q  pour quitter.\n")

    while True:
        try:
            phrase = input("Toi > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nArrêt.")
            break
        if not phrase:
            continue
        if phrase.lower() in ("quit", "exit", "q"):
            print("Arrêt du programme.")
            break

        try:
            cities = extract_cities(phrase, ner_bundle)
            if cities:
                print(f"LOC  > {cities}")
                if len(cities) >= 2:
                    print(f"       (départ : {cities[0]}, arrivée : {cities[-1]})")
                    route = pf.route_with_waypoints(cities)
                    names = " → ".join(route.station_names)
                    print(f"       Route : {names}")
                    print(f"       Temps : {format_duration(route.total_time_min)}")

                    schedule = pf.find_schedule_for_route(route.station_ids)
                    if schedule and schedule.legs:
                        for leg in schedule.legs:
                            t = leg.train_type or "Train"
                            print(f"       {t} n°{leg.train_number} : ", end="")
                            print(f"{leg.stops[0].station_name} → {leg.stops[-1].station_name}")
                        if schedule.connections:
                            for conn in schedule.connections:
                                print(
                                    f"       🔄 Correspondance à {conn.station_name} "
                                    f"({format_duration(conn.wait_min)} d'attente)"
                                )
                else:
                    print("       Une seule ville détectée, il en faut au moins 2.")
            else:
                print("       Aucun lieu détecté.")
        except Exception as e:
            print(f"Erreur : {e}")


def run_ui() -> None:
    """Lance l'UI Streamlit."""
    app_script = Path(__file__).resolve().parent / "app.py"
    if not app_script.is_file():
        print(f"Script UI introuvable : {app_script}", file=sys.stderr)
        sys.exit(1)
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(app_script)],
            check=True,
        )
    except FileNotFoundError:
        print("Streamlit non installé. Installe avec : pip install streamlit", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)


# ─── Résolution du modèle ────────────────────────────────────────────────────

def resolve_model(model_arg: str) -> str:
    """
    Si le modèle ressemble à un repo HF (contient /), télécharge-le.
    Sinon, le retourne tel quel (chemin local).
    """
    if "/" in model_arg and not Path(model_arg).exists():
        print(f"Téléchargement du modèle depuis Hugging Face : {model_arg}…")
        return download_hf_model(model_arg)
    return model_arg


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="T-AIA-911 — Travel Order Resolver (NER + Pathfinding SNCF)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", help="Fichier texte d'entrée (sentenceID,sentence)")
    group.add_argument("--url", help="URL du fichier texte")
    group.add_argument("--text", help="Texte direct (une phrase)")
    group.add_argument("--chat", action="store_true", help="Mode chat terminal interactif")
    group.add_argument("--ui", action="store_true", help="Lancer l'UI Streamlit")

    parser.add_argument("--output", help="Chemin du CSV de sortie (avec --file/--url)", default=None)
    parser.add_argument(
        "--model",
        help=f"Modèle NER (nom HF ou chemin local). Défaut : {DEFAULT_HF_MODEL}",
        default=DEFAULT_HF_MODEL,
    )
    args = parser.parse_args()

    # Mode UI : pas besoin de charger le modèle
    if args.ui:
        run_ui()
        return

    if args.output and not (args.file or args.url):
        parser.error("--output ne peut être utilisé qu'avec --file ou --url")

    # Résoudre le modèle
    model_path = resolve_model(args.model)

    # Détection du device
    device = get_device()
    print(f"Device : {device_label(device)}")
    print(f"Modèle : {model_path}")

    # Chargement NER
    print("Chargement du modèle NER…")
    ner_bundle = load_ner(model_path)
    print("Modèle chargé avec succès.")

    # Dispatch
    if args.file or args.url:
        output_path = (
            Path(args.output) if args.output
            else DATASET_DIR / "dataset_output.csv"
        )
        run_file_or_url(args.file or args.url, output_path, ner_bundle["ner"])
    elif args.chat:
        run_chat(ner_bundle)
    elif args.text:
        run_text(args.text.strip(), ner_bundle["ner"])


if __name__ == "__main__":
    main()
