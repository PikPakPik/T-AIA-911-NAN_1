import argparse
import torch
from transformers import pipeline
import csv
import os
import sys
from urllib.parse import urlparse
from urllib.request import urlopen

parser = argparse.ArgumentParser(description="Extract LOC entities from a text file using RoBERTa/CamemBERT")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--file", help="Path to input text file")
group.add_argument("--url", help="URL to input text file (one sentence per line)")
group.add_argument("--text", help="Direct input text (one sentence)")
group.add_argument("--chat", help="Interactive chat mode (type sentences one by one, 'exit' to quit)", action="store_true")

parser.add_argument("--output", help="Path to output CSV file", default=None)
parser.add_argument(
    "--model",
    help="RoBERTa/CamemBERT model to load. Options: Jean-Baptiste/camembert-ner (default), benjamin/roberta-base-wechsel-french (needs fine-tuning), ./roberta_ner_model (fine-tuned model)",
    choices=[
        "Jean-Baptiste/camembert-ner",
        "benjamin/roberta-base-wechsel-french",
        "./roberta_ner_model"
        ],
    default="Jean-Baptiste/camembert-ner"
)

args = parser.parse_args()

# Validation : --output uniquement avec --file ou --url
if args.output and not (args.file or args.url):
    parser.error("--output can only be used with --file or --url")

# Valeur par défaut pour --output si --file ou --url est utilisé
if (args.file or args.url) and not args.output:
    args.output = "dataset_output_roberta.csv"

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

try:
    # Essaie de charger comme un modèle NER fine-tuné
    ner_pipeline = pipeline(
        "ner",
        model=args.model,
        tokenizer=args.model,
        aggregation_strategy="simple",
        device=_device
    )
    print("Model loaded successfully as NER model!")
except Exception as e:
    print(f"Warning: Could not load as NER model: {e}")
    print("Attempting to use model with zero-shot NER or alternative approach...")
    # Si le modèle n'est pas fine-tuné pour la NER, on peut essayer une approche différente
    # Pour l'instant, on suggère d'utiliser un modèle NER fine-tuné
    print("Note: This model may not be fine-tuned for NER.")
    print("Consider using a NER fine-tuned model like 'Jean-Baptiste/camembert-ner'")
    print("or fine-tune this model for NER tasks.")
    raise

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

def read_lines_from_source(source: str):
    parsed = urlparse(source)
    if parsed.scheme in ("http", "https"):
        with urlopen(source) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            text = resp.read().decode(charset)
            return text.splitlines()

    if not os.path.isfile(source):
        print(f"File not found: {source}", file=sys.stderr)
        sys.exit(1)

    with open(source, encoding="utf-8") as f:
        lines = f.readlines()
        # Extrait seulement le texte après le numéro et la virgule (format: "numéro,texte")
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                cleaned_lines.append("")
                continue
            # Si la ligne commence par un numéro suivi d'une virgule, extrait le texte après
            if ',' in line:
                parts = line.split(',', 1)
                if len(parts) == 2 and parts[0].strip().isdigit():
                    cleaned_lines.append(parts[1].strip())
                else:
                    cleaned_lines.append(line)
            else:
                cleaned_lines.append(line)
        return cleaned_lines

if args.file or args.url:
    lines = read_lines_from_source(args.file if args.file else args.url)

    with open("./dataset/" + args.output, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        for idx, line in enumerate(lines, 1):
            phrase = line.strip()
    
            # Ignore les lignes vides
            if not phrase:
                writer.writerow([idx, 'INVALID', ''])
                continue
            
            # Traite la phrase avec le modèle NER
            try:
                entities = ner_pipeline(phrase)
            except Exception as e:
                print(f"Error processing line {idx}: {e}", file=sys.stderr)
                writer.writerow([idx, 'INVALID', ''])
                continue
            
            # Extrait les lieux (avec debug pour les premières lignes)
            lieux = extract_locations(entities)
            
            departure = lieux[0] if len(lieux) > 0 else 'INVALID'
            arrival = lieux[-1] if len(lieux) > 1 else 'INVALID'
            steps = lieux[1:-1] if len(lieux) > 2 else []
            
            writer.writerow([idx, departure] + steps + [arrival])
        
    print(f"Extraction completed. Output saved to {args.output} with {len(lines)} entries using RoBERTa model: {args.model}.")

if args.chat:
    print("Modèle prêt. Envoie une phrase pour extraire les lieux (LOC).")
    print("Commande: quit / exit / q  pour quitter.\n")

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
            entities = ner(phrase)
            locs = extract_locations(entities, phrase)
            locs = filter_non_travel_locs(phrase, locs)
            if locs:
                print(f"LOC  > {locs}")
                if len(locs) >= 2:
                    print(f"      (départ: {locs[0]}, arrivée: {locs[-1]})")
        except Exception as e:
            print(f"Erreur lors de l'extraction des entités : {e}")