import csv
import argparse
import os
import sys
from urllib.parse import urlparse
from urllib.request import urlopen
from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification, BertForTokenClassification
from transformers import pipeline
from transformers.modeling_tf_pytorch_utils import load_tf2_checkpoint_in_pytorch_model
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

# Device : MPS (Mac ARM64), CUDA, ou CPU
def _get_device():
    if torch.cuda.is_available():
        return 0
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return -1

# Charge le modèle RoBERTa/CamemBERT pour la NER
print(f"Loading model: {args.model}...")
_device = _get_device()
if isinstance(_device, torch.device):
    print(f"Device: MPS (Apple Silicon GPU)")
elif _device == 0:
    print("Device: CUDA")
else:
    print("Device: CPU")
# BERT Europeana : sauvegardé en TensorFlow (tf_model.h5). On crée le modèle depuis le config
# (tensors réels sur CPU) puis on charge les poids TF, pour éviter les "meta tensors" qui
# bloquent .to(device) quand on utilise from_pretrained(..., from_tf=True).
model_path = os.path.abspath(args.model)
tf_weights = os.path.join(model_path, "tf_model.h5")
use_bio_labels = False  # True pour BERT Europeana (B-DEP, I-DEP, B-ARR, I-ARR)
if os.path.isfile(tf_weights):
    use_bio_labels = True
    config = AutoConfig.from_pretrained(model_path)
    model = BertForTokenClassification(config)  # CPU, tensors réels (pas meta)
    load_tf2_checkpoint_in_pytorch_model(model, tf_weights, allow_missing_keys=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    ner_pipeline = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=_device,
    )
    print("Model loaded successfully (BERT Europeana, from TensorFlow weights).")
else:
    try:
        # Essaie de charger comme un modèle NER fine-tuné (PyTorch ou HuggingFace)
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
        print("Note: This model may not be fine-tuned for NER.")
        print("Consider using a NER fine-tuned model like 'Jean-Baptiste/camembert-ner'")
        print("or fine-tune this model for NER tasks.")
        raise

# Mapping possible du config (id2label LABEL_0..4) vers BIO (entraînement: B-ARR:0, B-DEP:1, I-ARR:2, I-DEP:3, O:4)
_LABEL_ID_TO_BIO = {"LABEL_0": "ARR", "LABEL_1": "DEP", "LABEL_2": "ARR", "LABEL_3": "DEP"}

# Mots français courants qui ne sont pas des lieux (faux positifs du modèle)
_BIO_NON_LOCATION_WORDS = frozenset({
    "réserver", "ferroviaire", "connexion", "matin", "train", "horaires", "tarif",
    "distance", "voyage", "billet", "trajet", "voir", "perdu", "ce", "un", "une",
    "des", "du", "la", "le", "les", "et", "en", "à", "a", "pour", "depuis", "entre",
    "aller", "boire", "monaco", "réserver", "circulent", "première", "classe",
})

# Fragments de mots (souvent artefacts de tokenisation, ex. "viaire" depuis "ferroviaire")
_BIO_FRAGMENT_PREFIXES = ("viaire", "naire", "tion", "res", "ent", "tre", "que", "nce")


def _is_likely_non_location(s):
    """True si le texte extrait ressemble à un non-lieu (mot courant ou fragment)."""
    if not s or len(s) < 2:
        return True
    t = s.strip().lower()
    if t in _BIO_NON_LOCATION_WORDS:
        return True
    # Un seul "mot" qui est un fragment connu
    words = t.split()
    if len(words) == 1 and any(t.startswith(p) for p in _BIO_FRAGMENT_PREFIXES) and len(t) <= 6:
        return True
    # Premier mot = non-lieu (ex. "Réserver brens" -> on garde que si tout le span est douteux)
    if words and words[0] in _BIO_NON_LOCATION_WORDS and len(words) == 1:
        return True
    return False


def extract_dep_arr_bio(entities, phrase="", debug=False):
    """
    Extrait Départ et Arrivée depuis les labels BIO (B-DEP/I-DEP, B-ARR/I-ARR).
    Gère aussi les noms génériques LABEL_0..LABEL_3 si le config n'a pas les vrais noms.
    Retourne (départ, arrivée) ; utilise start/end pour le texte exact si phrase fournie.
    """
    dep_words, arr_words = [], []
    unique_labels = set()
    for entity in entities:
        label = entity.get("entity_group", entity.get("label", ""))
        unique_labels.add(label)
        # Normaliser : B-DEP, I-DEP -> DEP ; B-ARR, I-ARR -> ARR ; ou LABEL_1/LABEL_3 -> DEP, LABEL_0/LABEL_2 -> ARR
        lu = label.upper()
        kind = None
        if "DEP" in lu:
            kind = "DEP"
        elif "ARR" in lu:
            kind = "ARR"
        elif label in _LABEL_ID_TO_BIO:
            kind = _LABEL_ID_TO_BIO[label]
        if kind == "DEP":
            dep_words.append((entity.get("start", 0), entity.get("end", 0), (entity.get("word", entity.get("text", "")) or "").strip()))
        elif kind == "ARR":
            arr_words.append((entity.get("start", 0), entity.get("end", 0), (entity.get("word", entity.get("text", "")) or "").strip()))
    dep_words.sort(key=lambda x: x[0])
    arr_words.sort(key=lambda x: x[0])
    dep_str = None
    arr_str = None
    if dep_words and phrase:
        dep_str = phrase[dep_words[0][0] : dep_words[-1][1]].strip()
    elif dep_words:
        dep_str = " ".join(w[2] for w in dep_words if w[2])
    if arr_words and phrase:
        arr_str = phrase[arr_words[0][0] : arr_words[-1][1]].strip()
    elif arr_words:
        arr_str = " ".join(w[2] for w in arr_words if w[2])
    # Filtre des faux positifs : mots courants non-lieux et fragments
    if dep_str and _is_likely_non_location(dep_str):
        dep_str = None
    if arr_str and _is_likely_non_location(arr_str):
        arr_str = None
    # Corriger les fragments type "viaire bâgé-dommartin" (arr commence par un bout de mot)
    if arr_str and phrase and arr_words:
        first = arr_str.split()[0].lower() if arr_str.split() else ""
        if first in _BIO_FRAGMENT_PREFIXES:
            start_arr = arr_words[0][0]
            while start_arr > 0 and phrase[start_arr - 1].isalnum():
                start_arr -= 1
            extended = phrase[start_arr : arr_words[-1][1]].strip()
            words = extended.split()
            if words and _is_likely_non_location(words[0]):
                # Enlever le premier mot (non-lieu) et garder le reste comme arrivée
                arr_str = " ".join(words[1:]).strip() or None
            else:
                arr_str = extended if not _is_likely_non_location(extended) else None
    if debug and unique_labels:
        print(f"  Debug: Labels détectés: {unique_labels}")
    return (dep_str, arr_str)


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
        
        if use_bio_labels:
            # BERT Europeana : labels BIO B-DEP, I-DEP, B-ARR, I-ARR (ou LABEL_0..3)
            departure, arrival = extract_dep_arr_bio(entities, phrase=phrase, debug=(idx <= 5))
            writer.writerow([idx, departure or "INVALID", arrival or "INVALID"])
        else:
            # Modèles LOC / LOCATION
            lieux = extract_locations(entities, debug=(idx <= 5))
            departure = lieux[0] if len(lieux) > 0 else "INVALID"
            arrival = lieux[-1] if len(lieux) > 1 else "INVALID"
            steps = lieux[1:-1] if len(lieux) > 2 else []
            writer.writerow([idx, departure] + steps + [arrival])
    
print(f"Extraction completed. Output saved to {args.output} with {len(lines)} entries using RoBERTa model: {args.model}.")
