import csv
import argparse
import os
import sys
from urllib.parse import urlparse
from urllib.request import urlopen
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import torch

# Extraire des lieux (entités LOC) depuis un fichier texte local ou une URL.
# Options :
#  --file  : chemin vers un fichier texte local (une phrase par ligne).
#  --url   : URL HTTP/HTTPS pointant vers un fichier texte (une phrase par ligne).
#  --output: chemin vers le fichier CSV de sortie (par défaut dataset_output.csv).
#  --model : modèle RoBERTa/CamemBERT à utiliser (par défaut benjamin/roberta-base-wechsel-french).
# Les deux options sont mutuellement exclusives et au moins une est requise.
# Le script lit les lignes, identifie les entités LOC et écrit les résultats dans dataset_output.csv.

parser = argparse.ArgumentParser(description="Extract LOC entities from a text file using RoBERTa/CamemBERT")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--file", help="Path to input text file")
group.add_argument("--url", help="URL to input text file (one sentence per line)")

parser.add_argument("--output", help="Path to output CSV file", default="dataset_output_roberta.csv")
parser.add_argument(
    "--model",
    help="RoBERTa/CamemBERT model to load. Options: Jean-Baptiste/camembert-ner (default), benjamin/roberta-base-wechsel-french (needs fine-tuning), ./roberta_ner_model (fine-tuned model)",
    default="Jean-Baptiste/camembert-ner",
)

args = parser.parse_args()

# Lis les lignes depuis une source pouvant être un fichier local ou une URL HTTP/HTTPS.
# Paramètre : source (str) — chemin local ou URL.
# Si source est une URL (http/https) : télécharge la ressource, décode selon l'en-tête charset (ou 'utf-8') et retourne text.splitlines() (liste de lignes sans caractères de fin).
# Sinon : vérifie l'existence du fichier ; si absent, écrit une erreur sur stderr et appelle sys.exit(1) ; sinon ouvre le fichier en utf-8 et retourne f.readlines() (lignes, incluant séparateurs).
# Retour : List[str] — liste de lignes prêtes à être traitées.
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

lines = read_lines_from_source(args.file if args.file else args.url)

# Charge le modèle RoBERTa/CamemBERT pour la NER
print(f"Loading model: {args.model}...")
try:
    # Essaie de charger comme un modèle NER fine-tuné
    ner_pipeline = pipeline(
        "ner",
        model=args.model,
        tokenizer=args.model,
        aggregation_strategy="simple",
        device=0 if torch.cuda.is_available() else -1
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

# Fonction pour extraire les entités LOC depuis les résultats de la NER
def extract_locations(entities, debug=False):
    """
    Extrait les entités de type LOC depuis les résultats de la NER.
    Les modèles peuvent utiliser différents labels pour les lieux :
    - LOC, LOCATION, LOC-PER, MISC (pour certains modèles français), etc.
    """
    locations = []
    unique_labels = set()
    
    for entity in entities:
        label = entity.get('entity_group', entity.get('label', ''))
        unique_labels.add(label)
        
        # Normalise les labels pour détecter les lieux
        label_upper = label.upper()
        # Supporte différents formats de labels NER
        if 'LOC' in label_upper or 'LOCATION' in label_upper or label_upper in ['B-LOC', 'I-LOC']:
            word = entity.get('word', entity.get('text', ''))
            # Nettoie le mot (enlève les espaces en début/fin)
            word = word.strip()
            if word:
                locations.append(word)
    
    if debug and unique_labels:
        print(f"  Debug: Labels détectés: {unique_labels}")
    
    return locations

# Traite chaque ligne pour extraire les entités LOC et écrit les résultats dans dataset_output.csv.
# Chaque ligne du CSV contient: [index, départ, ... étapes ..., arrivée].
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
        
        # Affiche les entités détectées pour debug (seulement les premières lignes)
        if idx <= 5:
            for ent in entities:
                print(f"Line {idx} - Entity: {ent.get('word', ent.get('text', ''))}, Label: {ent.get('entity_group', ent.get('label', ''))}")
        
        # Extrait les lieux (avec debug pour les premières lignes)
        lieux = extract_locations(entities, debug=(idx <= 5))
        
        departure = lieux[0] if len(lieux) > 0 else 'INVALID'
        arrival = lieux[-1] if len(lieux) > 1 else 'INVALID'
        steps = lieux[1:-1] if len(lieux) > 2 else []
        
        writer.writerow([idx, departure] + steps + [arrival])
    
print(f"Extraction completed. Output saved to {args.output} with {len(lines)} entries using RoBERTa model: {args.model}.")
