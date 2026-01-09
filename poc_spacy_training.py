import re
import requests
import csv
import random
import spacy
from spacy.tokens import DocBin
import unicodedata

with open("./dataset/geo.csv", "r", encoding="utf-8") as file:
    reader = csv.reader(file)
    locations = list(reader)

def remove_accents(text: str) -> str:
    """Remove accents from text."""
    nfkd = unicodedata.normalize('NFD', text)
    return ''.join([c for c in nfkd if not unicodedata.combining(c)])

def generate_variants(name: str) -> set[str]:
    variants = set()

    base = name.strip()

    saint_rules = [
        (r"\bSaint\b", ["Saint", "St"]),
        (r"\bSainte\b", ["Sainte", "Ste"]),
    ]

    def apply_rules(text):
        results = {text}
        for pattern, repls in saint_rules:
            new = set()
            for r in results:
                if re.search(pattern, r, flags=re.IGNORECASE):
                    for repl in repls:
                        new.add(re.sub(pattern, repl, r, flags=re.IGNORECASE))
                else:
                    new.add(r)
            results = new
        return results

    for v in apply_rules(base):
        variants.add(v)
        variants.add(v.replace("-", " "))
        variants.add(v.replace(" ", "-"))
        variants.add(v.replace("'", " "))
        variants.add(v.replace("'", ""))

        no_accent = remove_accents(v)
        variants.add(no_accent)
        variants.add(no_accent.replace("-", " "))
        variants.add(no_accent.replace(" ", "-"))
        variants.add(no_accent.replace("'", " "))
        variants.add(no_accent.replace("'", ""))

    final = set()
    for v in variants:
        final.add(v)
        final.add(v.lower())
        final.add(v.upper())

    return final

TEMPLATES = [
    # Base simple
    "Je veux aller de {a} à {b}",
    "Train entre {a} et {b}",
    "Voyage de {a} vers {b}",
    "Billet {a} {b}",

    # Questions
    "Comment aller de {a} à {b}?",
    "Quel train entre {a} et {b}?",
    "C'est combien pour aller de {a} à {b}?",
    
    # Formulations naturelles
    "Je dois me rendre de {a} à {b}",
    "Je dois aller de {a} à {b} demain",
    "Un aller-retour {a}-{b} s'il vous plaît",
    "Un billet {a} {b} svp",
    
    # Avec contexte temporel
    "Départ de {a} vers {b} ce soir",
    "Je pars de {a} pour {b} lundi prochain",
    "Train de {a} à {b} demain matin",
    "Départ {a} arrivée {b} cet après-midi",
    
    # Variations ordre inversé
    "Trajet depuis {a} vers {b}",
    "Réserver un billet de {a} pour {b}",
    "Je souhaite me rendre à {b} depuis {a}",
    "Je veux aller à {b} en partant de {a}",
    "Aller à {b} au départ de {a}",
    "Direction {b} depuis {a}",
    
    # Formulations familières
    "J'ai besoin d'aller de {a} à {b}",
    "Faut que j'aille de {a} à {b}",
    "Cherche train {a} {b}",
    "Dispo {a}-{b}?",
    
    # Avec détails
    "Réservation au départ de {a} direction {b}",
    "Billet de train de {a} à destination de {b}",
    "Partir de {a} pour rejoindre {b}",
    "Trajet en train de {a} jusqu'à {b}",
    
    # Variations
    "Trajet depuis {a} vers {b}",
    "Réserver un billet de {a} pour {b}",
]

def generate_annotated_sentences(locations, n=25000):
    samples = []

    for _ in range(n):
        (name1, label1), (name2, label2) = random.sample(locations, 2)

        v1 = random.choice(list(generate_variants(name1)))
        v2 = random.choice(list(generate_variants(name2)))

        sentence = random.choice(TEMPLATES).format(a=v1, b=v2)

        start1 = sentence.index(v1)
        end1 = start1 + len(v1)
        start2 = sentence.index(v2)
        end2 = start2 + len(v2)

        samples.append((
            sentence,
            [
                (start1, end1, label1),
                (start2, end2, label2),
            ]
        ))

    return samples

def build_docbin(samples, output_path):
    nlp = spacy.blank("fr")
    db = DocBin()

    for text, entities in samples:
        doc = nlp.make_doc(text)
        spans = []
        for start, end, label in entities:
            span = doc.char_span(start, end, label=label)
            if span:
                spans.append(span)
        doc.ents = spans
        db.add(doc)

    db.to_disk(output_path)

samples = generate_annotated_sentences(locations, n=15000)

random.shuffle(samples)
split_train = int(len(samples) * 0.8)
split_dev = int(len(samples) * 0.9)

train_data = samples[:split_train]
dev_data = samples[split_train:split_dev]
test_data = samples[split_dev:]

build_docbin(train_data, "./training_data/train.spacy")
build_docbin(dev_data, "./training_data/dev.spacy")
build_docbin(test_data, "./training_data/test.spacy")