import spacy
import csv
import argparse
import os
import sys
from urllib.parse import urlparse
from urllib.request import urlopen

# Extraire des lieux (entités LOC) depuis un fichier texte local ou une URL.
# Options :
#  --file  : chemin vers un fichier texte local (une phrase par ligne).
#  --url   : URL HTTP/HTTPS pointant vers un fichier texte (une phrase par ligne).
#  --output: chemin vers le fichier CSV de sortie (par défaut dataset_output.csv).
#  --model : modèle SpaCy à utiliser (fr_core_news_md ou fr_core_news_lg, par défaut fr_core_news_lg).
# Les deux options sont mutuellement exclusives et au moins une est requise.
# Le script lit les lignes, identifie les entités LOC et écrit les résultats dans dataset_output.csv.
parser = argparse.ArgumentParser(description="Extract LOC entities from a text file")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--file", help="Path to input text file")
group.add_argument("--url", help="URL to input text file (one sentence per line)")

parser.add_argument("--output", help="Path to output CSV file", default="dataset_output.csv")
parser.add_argument(
    "--model",
    help="SpaCy model to load",
    choices=["fr_core_news_md", "fr_core_news_lg"],
    default="fr_core_news_lg",
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
        return f.readlines()

lines = read_lines_from_source(args.file if args.file else args.url)

nlp = spacy.load(args.model)

# Traite chaque ligne pour extraire les entités LOC et écrit les résultats dans dataset_output.csv.
# Chaque ligne du CSV contient: [index, départ, ... étapes ..., arrivée].
with open(args.output, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)

    for idx, line in enumerate(lines, 1):
        phrase = line.strip()
        doc = nlp(phrase)
        lieux = [ent.text for ent in doc.ents if ent.label_ == "LOC"]

        departure = lieux[0] if len(lieux) > 0 else 'INVALID'
        arrival = lieux[-1] if len(lieux) > 1 else 'INVALID'
        steps = lieux[1:-1] if len(lieux) > 2 else []
        
        writer.writerow([idx, departure] + steps + [arrival])
    
print(f"Extraction completed. Output saved to {args.output} with {len(lines)} entries on SpaCy model: {args.model}.")