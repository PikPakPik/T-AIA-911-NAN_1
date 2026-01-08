import re
import requests
import csv
import random
import spacy
from spacy.tokens import DocBin
import unicodedata

regions = requests.get("https://geo.api.gouv.fr/regions").json()
departments = requests.get("https://geo.api.gouv.fr/departements").json()
cities = requests.get("https://geo.api.gouv.fr/communes").json()
associated_delegated_cities = requests.get("https://geo.api.gouv.fr/communes_associees_deleguees").json()

with open('geo.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)

    print("Writing regions")
    for region in regions:
        writer.writerow([region["nom"], "LOC_REGION"])

    print("Writing departments")
    for department in departments:
        writer.writerow([department["nom"], "LOC_DEPARTMENT"])

    print("Writing cities")
    for city in cities:
        writer.writerow([city["nom"], "LOC_CITY"])

    print("Writing associated/delegated cities")
    for city in associated_delegated_cities:
        writer.writerow([city["nom"], "LOC_CITY"])

with open("geo.csv", "r", encoding="utf-8") as file:
    reader = csv.reader(file)
    locations = list(reader)

TEMPLATES = [
    "Je veux aller de {a} à {b}",
    "Train entre {a} et {b}",
    "Voyage de {a} vers {b}",
    "Je souhaite me rendre à {b} depuis {a}",
    "Billet {a} {b}",
]

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

for row in locations:
    print(generate_variants(row[0]))