#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Génère un jeu de TEST pour évaluer le modèle RoBERTa NER après entraînement.
Utilise des formulations et un seed DIFFÉRENTS de generate_dataset.py
pour que le test soit sur des phrases jamais vues à l'entraînement.
Sortie : dataset_test_input.txt, dataset_test_output.txt
"""

import random
import itertools
from typing import List, Tuple

# Même liste de villes que le dataset d'entraînement (cohérence du domaine)
VILLES = [
    "Paris", "Lyon", "Marseille", "Toulouse", "Nice", "Nantes", "Strasbourg",
    "Montpellier", "Bordeaux", "Lille", "Rennes", "Reims", "Le Havre",
    "Saint-Étienne", "Toulon", "Grenoble", "Dijon", "Angers", "Nîmes",
    "Villeurbanne", "Saint-Denis", "Le Mans", "Aix-en-Provence", "Clermont-Ferrand",
    "Brest", "Tours", "Limoges", "Amiens", "Perpignan", "Metz", "Besançon",
    "Orléans", "Mulhouse", "Caen", "Boulogne-Billancourt", "Rouen", "Nancy",
    "Argenteuil", "Montreuil", "Saint-Denis", "Roubaix", "Avignon", "Tourcoing",
    "Fort-de-France", "Créteil", "Dunkirk", "Poitiers", "Asnières-sur-Seine",
    "Versailles", "Courbevoie", "Vitry-sur-Seine", "Colombes", "Aulnay-sous-Bois",
    "La Rochelle", "Antibes", "Mérignac", "Béziers", "Pessac", "Port-Boulet",
    "Albert", "Lourdes", "Florence", "Montaigu",
]

# Formulations de TEST différentes de generate_dataset.py (jamais vues à l'entraînement)
TEST_VALID_PATTERNS = [
    lambda d, a: f"J'ai besoin d'un trajet {d} {a}",
    lambda d, a: f"Réserver {d} - {a} pour demain",
    lambda d, a: f"Connexion ferroviaire {d} {a}",
    lambda d, a: f"Prochain train {d} vers {a}",
    lambda d, a: f"Billet SNCF {d} {a}",
    lambda d, a: f"Correspondance entre {d} et {a}",
    lambda d, a: f"TGV au départ de {d} à destination de {a}",
    lambda d, a: f"Liaison {d} {a} aujourd'hui",
    lambda d, a: f"Tarif {d} {a} s'il vous plaît",
    lambda d, a: f"Où prendre le train {d} {a}",
    lambda d, a: f"Réservation {d} {a} pour vendredi",
    lambda d, a: f"Trajet direct {d} {a}",
    lambda d, a: f"Des trains circulent-ils entre {d} et {a}",
    lambda d, a: f"Horaires {d} {a} ce week-end",
    lambda d, a: f"Voyage {d} {a} en première classe",
    lambda d, a: f"Peut-on aller de {d} à {a} en train",
    lambda d, a: f"Distance {d} {a} en train",
    lambda d, a: f"Arrêt à {a} en venant de {d}",
    lambda d, a: f"Train du matin {d} {a}",
    lambda d, a: f"Correspondance {d} puis {a}",
]

TEST_INVALID_PATTERNS = [
    "où est la gare la plus proche",
    "je voudrais un café",
    "quel temps fait-il",
    "réserver une table pour deux",
    "annuler ma commande",
    "où sont les toilettes",
    "combien coûte ce livre",
    "je cherche une pharmacie",
    "quelle heure est-il",
    "je me suis perdu",
    "pouvez-vous répéter",
    "je ne comprends pas",
    "où puis-je acheter un ticket",
]

# Pièges pour le test (voir / un / monaco boisson)
PRENOMS_VILLES = {
    "Albert": "albert",
    "Paris": "paris",
    "Florence": "florence",
    "Monaco": "monaco",
    "Nancy": "nancy",
    "Valérie": "valerie",
}

TEST_TRAP_PATTERNS_VALID = [
    lambda d, a, p: f"avec {p} je prends le train de {d} à {a}",
    lambda d, a, p: f"mon collègue {p} part de {d} pour {a}",
]

TEST_TRAP_PATTERNS_INVALID = [
    lambda d, a, p: f"je veux aller voir {p} et boire un monaco",
    lambda d, a, p: f"un billet pour voir {p} c'est quoi",
    lambda d, a, p: f"boire un monaco en allant à {a} depuis {d}",
]


def add_variations(phrase: str) -> List[str]:
    """Quelques variations (moins que l'entraînement pour garder le test lisible)."""
    variations = [phrase]
    no_accents = phrase.replace("é", "e").replace("è", "e").replace("ê", "e")
    no_accents = no_accents.replace("à", "a").replace("â", "a").replace("ù", "u")
    no_accents = no_accents.replace("ô", "o").replace("î", "i").replace("ç", "c")
    if no_accents != phrase:
        variations.append(no_accents)
    if phrase != phrase.lower():
        variations.append(phrase.lower())
    return list(set(variations))


def generate_valid(num: int) -> List[Tuple[str, str, str]]:
    sentences = []
    ville_pairs = list(itertools.permutations(VILLES, 2))
    for _ in range(num):
        pattern = random.choice(TEST_VALID_PATTERNS)
        depart, arrivee = random.choice(ville_pairs)
        base = pattern(depart.lower(), arrivee.lower())
        sentence = random.choice(add_variations(base))
        sentences.append((sentence, depart, arrivee))
    return sentences


def generate_invalid(num: int) -> List[Tuple[str, str, str]]:
    sentences = []
    for _ in range(num):
        sentence = random.choice(TEST_INVALID_PATTERNS)
        sentence = random.choice(add_variations(sentence))
        sentences.append((sentence, "INVALID", ""))
    return sentences


def generate_trap_valid(num: int) -> List[Tuple[str, str, str]]:
    sentences = []
    villes_normales = [v for v in VILLES if v not in PRENOMS_VILLES]
    ville_pairs = list(itertools.permutations(villes_normales, 2))
    prenoms = list(PRENOMS_VILLES.values())
    for _ in range(num):
        pattern = random.choice(TEST_TRAP_PATTERNS_VALID)
        depart, arrivee = random.choice(ville_pairs)
        prenom = random.choice(prenoms)
        base = pattern(depart.lower(), arrivee.lower(), prenom)
        sentence = random.choice(add_variations(base))
        sentences.append((sentence, depart, arrivee))
    return sentences


def generate_trap_invalid(num: int) -> List[Tuple[str, str, str]]:
    sentences = []
    prenoms = list(PRENOMS_VILLES.values())
    villes_normales = [v for v in VILLES if v not in PRENOMS_VILLES]
    for _ in range(num):
        pattern = random.choice(TEST_TRAP_PATTERNS_INVALID)
        prenom = random.choice(prenoms)
        depart = random.choice(villes_normales)
        arrivee = random.choice(villes_normales)
        if depart.lower() != arrivee.lower():
            base = pattern(depart.lower(), arrivee.lower(), prenom)
        else:
            base = f"je veux aller voir {prenom} et boire un monaco"
        sentence = random.choice(add_variations(base))
        sentences.append((sentence, "INVALID", ""))
    return sentences


def main():
    # Seed DIFFÉRENT de generate_dataset.py (42) pour des combinaisons jamais vues
    random.seed(123)

    num_valid = 1200
    num_invalid = 200
    num_trap_valid = 150
    num_trap_invalid = 250

    all_sentences = (
        generate_valid(num_valid)
        + generate_invalid(num_invalid)
        + generate_trap_valid(num_trap_valid)
        + generate_trap_invalid(num_trap_invalid)
    )
    random.shuffle(all_sentences)

    with open("dataset_test_input.txt", "w", encoding="utf-8") as f_in:
        with open("dataset_test_output.txt", "w", encoding="utf-8") as f_out:
            for sid, (sentence, depart, arrivee) in enumerate(all_sentences, 1):
                f_in.write(f"{sid},{sentence}\n")
                if depart == "INVALID":
                    f_out.write(f"{sid},INVALID,\n")
                else:
                    f_out.write(f"{sid},{depart},{arrivee}\n")

    print("Dataset de TEST généré (formulations différentes de l'entraînement).")
    print(f"  - Valides: {num_valid}, Invalides: {num_invalid}")
    print(f"  - Pièges valides: {num_trap_valid}, Pièges invalides: {num_trap_invalid}")
    print(f"  - Total: {len(all_sentences)}")
    print("  - Fichiers: dataset_test_input.txt, dataset_test_output.txt")


if __name__ == "__main__":
    main()
