import re
import requests
import csv
import random
import spacy
from spacy.tokens import DocBin
import unicodedata

# Charger les données géographiques
with open("geo.csv", "r", encoding="utf-8") as file:
    reader = csv.reader(file)
    locations = list(reader)

# Organiser par type
cities = [loc for loc in locations if loc[1] == "LOC_CITY"]
regions = [loc for loc in locations if loc[1] == "LOC_REGION"]
departments = [loc for loc in locations if loc[1] == "LOC_DEPARTMENT"]

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

# Templates origine-destination (comme avant)
TEMPLATES_ORIGIN_DEST = [
    "Je veux aller de {a} à {b}",
    "Train entre {a} et {b}",
    "Voyage de {a} vers {b}",
    "Billet {a} {b}",
    "Comment aller de {a} à {b}?",
    "Quel train entre {a} et {b}?",
    "C'est combien pour aller de {a} à {b}?",
    "Je dois me rendre de {a} à {b}",
    "Je dois aller de {a} à {b} demain",
    "Un aller-retour {a}-{b} s'il vous plaît",
    "Un billet {a} {b} svp",
    "Départ de {a} vers {b} ce soir",
    "Je pars de {a} pour {b} lundi prochain",
    "Train de {a} à {b} demain matin",
    "Départ {a} arrivée {b} cet après-midi",
    "Trajet depuis {a} vers {b}",
    "Réserver un billet de {a} pour {b}",
    "Je souhaite me rendre à {b} depuis {a}",
    "Je veux aller à {b} en partant de {a}",
    "Aller à {b} au départ de {a}",
    "Direction {b} depuis {a}",
    "J'ai besoin d'aller de {a} à {b}",
    "Faut que j'aille de {a} à {b}",
    "Cherche train {a} {b}",
    "Dispo {a}-{b}?",
    "Réservation au départ de {a} direction {b}",
    "Billet de train de {a} à destination de {b}",
    "Partir de {a} pour rejoindre {b}",
    "Trajet en train de {a} jusqu'à {b}",
]

# NOUVEAUX Templates : ville + région/département
TEMPLATES_CITY_REGION = [
    "Je vais à {city} en {region}",
    "Je vais à {city} dans le {region}",
    "Je vais à {city} dans la {region}",
    "Train pour {city} en {region}",
    "Train pour {city} dans le {region}",
    "Voyage vers {city} en {region}",
    "Je me rends à {city} en {region}",
    "Départ vers {city} en {region}",
    "Départ vers {city} dans la {region}",
    "Billet pour {city} en {region}",
    "Comment aller à {city} en {region}?",
    "Quel train pour {city} en {region}?",
    "{city} en {region} demain",
    "{city} dans le {region} ce soir",
    "Direction {city} en {region}",
    "Je dois aller à {city} dans la {region}",
    "Trajet vers {city} en {region}",
    "Réserver pour {city} dans le {region}",
    "Je pars pour {city} en {region}",
    "Aller à {city} dans la {region} demain matin",
]

# Templates : ville + département
TEMPLATES_CITY_DEPT = [
    "Je vais à {city} dans le {dept}",
    "Je vais à {city} dans la {dept}",
    "Train pour {city} dans le {dept}",
    "Voyage vers {city} dans le {dept}",
    "Je me rends à {city} dans le {dept}",
    "Départ vers {city} dans le {dept}",
    "Billet pour {city} dans le {dept}",
    "Comment aller à {city} dans le {dept}?",
    "{city} dans le {dept} demain",
    "Direction {city} dans le {dept}",
]

# Templates : origine + ville en région
TEMPLATES_ORIGIN_CITY_REGION = [
    "De {origin} à {city} en {region}",
    "De {origin} vers {city} en {region}",
    "Train de {origin} pour {city} en {region}",
    "Je pars de {origin} pour {city} en {region}",
    "Voyage de {origin} vers {city} dans la {region}",
    "De {origin} à {city} dans le {region}",
    "Billet de {origin} pour {city} en {region}",
]

def generate_annotated_sentences(locations, cities, regions, departments, n=25000):
    samples = []
    
    # 60% origine-destination (garder la logique existante)
    n_origin_dest = int(n * 0.6)
    for _ in range(n_origin_dest):
        (name1, label1), (name2, label2) = random.sample(locations, 2)
        v1 = random.choice(list(generate_variants(name1)))
        v2 = random.choice(list(generate_variants(name2)))
        
        sentence = random.choice(TEMPLATES_ORIGIN_DEST).format(a=v1, b=v2)
        
        start1 = sentence.find(v1)
        end1 = start1 + len(v1)
        start2 = sentence.find(v2, end1)  # Chercher après la première occurrence
        end2 = start2 + len(v2)
        
        samples.append((
            sentence,
            [
                (start1, end1, label1),
                (start2, end2, label2),
            ]
        ))
    
    # 25% ville + région
    n_city_region = int(n * 0.25)
    for _ in range(n_city_region):
        city_name, city_label = random.choice(cities)
        region_name, region_label = random.choice(regions)
        
        v_city = random.choice(list(generate_variants(city_name)))
        v_region = random.choice(list(generate_variants(region_name)))
        
        sentence = random.choice(TEMPLATES_CITY_REGION).format(city=v_city, region=v_region)
        
        start_city = sentence.find(v_city)
        end_city = start_city + len(v_city)
        start_region = sentence.find(v_region)
        end_region = start_region + len(v_region)
        
        entities = []
        if start_city != -1:
            entities.append((start_city, end_city, city_label))
        if start_region != -1:
            entities.append((start_region, end_region, region_label))
        
        if entities:
            samples.append((sentence, entities))
    
    # 10% ville + département
    n_city_dept = int(n * 0.10)
    for _ in range(n_city_dept):
        city_name, city_label = random.choice(cities)
        dept_name, dept_label = random.choice(departments)
        
        v_city = random.choice(list(generate_variants(city_name)))
        v_dept = random.choice(list(generate_variants(dept_name)))
        
        sentence = random.choice(TEMPLATES_CITY_DEPT).format(city=v_city, dept=v_dept)
        
        start_city = sentence.find(v_city)
        end_city = start_city + len(v_city)
        start_dept = sentence.find(v_dept)
        end_dept = start_dept + len(v_dept)
        
        entities = []
        if start_city != -1:
            entities.append((start_city, end_city, city_label))
        if start_dept != -1:
            entities.append((start_dept, end_dept, dept_label))
        
        if entities:
            samples.append((sentence, entities))
    
    # 5% origine + ville en région
    n_origin_city_region = n - n_origin_dest - n_city_region - n_city_dept
    for _ in range(n_origin_city_region):
        origin_name, origin_label = random.choice(locations)
        city_name, city_label = random.choice(cities)
        region_name, region_label = random.choice(regions)
        
        v_origin = random.choice(list(generate_variants(origin_name)))
        v_city = random.choice(list(generate_variants(city_name)))
        v_region = random.choice(list(generate_variants(region_name)))
        
        sentence = random.choice(TEMPLATES_ORIGIN_CITY_REGION).format(
            origin=v_origin, city=v_city, region=v_region
        )
        
        entities = []
        
        start_origin = sentence.find(v_origin)
        if start_origin != -1:
            end_origin = start_origin + len(v_origin)
            entities.append((start_origin, end_origin, origin_label))
        
        start_city = sentence.find(v_city)
        if start_city != -1:
            end_city = start_city + len(v_city)
            entities.append((start_city, end_city, city_label))
        
        start_region = sentence.find(v_region)
        if start_region != -1:
            end_region = start_region + len(v_region)
            entities.append((start_region, end_region, region_label))
        
        if entities:
            samples.append((sentence, entities))
    
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

# Générer 20000 exemples (plus d'exemples = meilleur modèle)
samples = generate_annotated_sentences(locations, cities, regions, departments, n=20000)

random.shuffle(samples)
split_train = int(len(samples) * 0.8)

train_data = samples[:split_train]
dev_data = samples[split_train:]

print(f"Generated {len(train_data)} training samples")
print(f"Generated {len(dev_data)} dev samples")

build_docbin(train_data, "train.spacy")
build_docbin(dev_data, "dev.spacy")

print("Training data saved to train.spacy")
print("Dev data saved to dev.spacy")