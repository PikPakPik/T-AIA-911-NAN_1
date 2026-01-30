#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de génération de dataset pour le projet Travel Order Resolver
Génère des phrases variées avec différentes structures grammaticales
"""

import random
import itertools
from typing import List, Tuple

# Liste de villes/gares françaises courantes
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
    "Albert", "Lourdes", "Florence", "Montaigu"
]

# Structures de phrases valides
VALID_PATTERNS = [
    # Structure 1: "Je voudrais aller de X à Y"
    lambda d, a: f"je voudrais aller de {d} à {a}",
    lambda d, a: f"Je voudrais aller de {d} à {a}",
    lambda d, a: f"J'aimerais aller de {d} à {a}",
    lambda d, a: f"je souhaite me rendre de {d} à {a}",
    
    # Structure 2: "Je voudrais aller à Y depuis X"
    lambda d, a: f"je voudrais aller à {a} depuis {d}",
    lambda d, a: f"Je souhaite me rendre à {a} depuis {d}",
    lambda d, a: f"j'aimerais aller à {a} en partant de {d}",
    
    # Structure 3: "Comment aller de X à Y ?"
    lambda d, a: f"comment aller de {d} à {a}",
    lambda d, a: f"Comment me rendre de {d} à {a}",
    lambda d, a: f"comment puis-je aller de {d} à {a}",
    lambda d, a: f"Comment faire pour aller de {d} à {a}",
    
    # Structure 4: "Y depuis X"
    lambda d, a: f"je veux {a} depuis {d}",
    lambda d, a: f"Aller à {a} depuis {d}",
    lambda d, a: f"Un billet pour {a} depuis {d}",
    
    # Structure 5: "De X vers Y"
    lambda d, a: f"de {d} vers {a}",
    lambda d, a: f"Je voudrais de {d} vers {a}",
    lambda d, a: f"Trains de {d} vers {a}",
    
    # Structure 6: "Il y a-t-il des trains de X à Y"
    lambda d, a: f"il y a-t-il des trains de {d} à {a}",
    lambda d, a: f"Y a-t-il des trains de {d} à {a}",
    lambda d, a: f"Avez-vous des trains de {d} à {a}",
    lambda d, a: f"Est-ce qu'il y a des trains de {d} à {a}",
    
    # Structure 7: "A quelle heure pour X à Y"
    lambda d, a: f"à quelle heure y a-t-il des trains de {d} à {a}",
    lambda d, a: f"Quels horaires de {d} à {a}",
    lambda d, a: f"Horaires de {d} à {a}",
    
    # Structure 8: Avec contexte
    lambda d, a: f"je veux aller voir mon ami à {a} en partant de {d}",
    lambda d, a: f"Pour aller à {a} depuis {d}, quels sont les trains",
    lambda d, a: f"Je dois me rendre de {d} à {a} pour mon travail",
    lambda d, a: f"Un voyage de {d} à {a} s'il vous plaît",
    
    # Structure 9: Format simple "X Y" ou "X Paris Y"
    lambda d, a: f"je voudrais un billet {d} {a}",
    lambda d, a: f"Un trajet {d} {a}",
    lambda d, a: f"{d} {a}",
    lambda d, a: f"{d} vers {a}",
    
    # Structure 10: Avec "passer par" (bonus)
    lambda d, a: f"je voudrais aller de {d} à {a}",
    lambda d, a: f"De {d} jusqu'à {a}",
]

# Phrases invalides : rien à voir avec une demande de trajet
INVALID_PATTERNS = [
    # Courtoisie / général
    "bonjour comment allez-vous",
    "merci beaucoup",
    "à bientôt",
    "pouvez-vous m'aider",
    "je ne sais pas quoi faire",
    "c'est une belle journée",
    "une phrase sans origine ni destination",
    "je veux juste dire bonjour",
    "au revoir",
    "bonne journée",
    "excusez-moi",
    "s'il vous plaît",
    "pas de souci",
    "très bien",
    "d'accord",
    "compris",
    # Achats / réservations (non train)
    "je voudrais acheter un livre",
    "je voudrais réserver une chambre d'hôtel",
    "je cherche un restaurant",
    "réserver une table pour deux",
    "où est la pharmacie",
    "je voudrais un café",
    "l'addition s'il vous plaît",
    "c'est combien",
    "j'aimerais commander une pizza",
    "où puis-je acheter des billets de concert",
    # Météo / infos
    "quelle est la météo aujourd'hui",
    "il fait beau",
    "il pleut demain",
    "quelle heure est-il",
    "quelle date sommes-nous",
    "où sont les toilettes",
    "où est la gare la plus proche",
    "c'est ouvert à quelle heure",
    # Demandes non voyage
    "je cherche un emploi",
    "comment aller au musée",
    "où est le parc",
    "annuler ma commande",
    "je me suis perdu",
    "pouvez-vous répéter",
    "je ne comprends pas",
    "quel temps fait-il",
    "combien coûte ce livre",
    "où puis-je acheter un ticket de métro",
    "je voudrais un renseignement",
    "c'est où la sortie",
    "y a-t-il un distributeur",
    # Conversation générale
    "qu'est-ce que vous recommandez",
    "c'est délicieux",
    "je n'aime pas ça",
    "on fait quoi ce soir",
    "tu viens ce week-end",
    "raconte-moi ta journée",
    "comment ça va",
    "ça fait longtemps",
    "je suis fatigué",
    "j'ai faim",
    "il est tard",
    "je dois partir",
    "à tout à l'heure",
    "bonne nuit",
    "bon appétit",
    # Questions diverses
    "pourquoi pas",
    "comment ça marche",
    "c'est possible",
    "vous avez des infos",
    "quel est le numéro",
    "c'est gratuit",
    "il y a un wifi",
    "où est le parking",
    "c'est loin",
    "combien de temps ça prend",
    # Ordres / refus
    "stop",
    "attendez",
    "non merci",
    "pas maintenant",
    "plus tard",
    "je réfléchis",
    "laissez-moi tranquille",
    "je ne veux pas",
    "c'est tout",
    "rien d'autre",
    # Téléphone / tech
    "mon portable est déchargé",
    "vous avez du réseau",
    "je n'ai pas internet",
    "comment on fait une réservation en ligne",
    "l'application ne marche pas",
    # Santé / urgence
    "j'ai mal à la tête",
    "où est l'hôpital",
    "appelez une ambulance",
    "au secours",
    "aidez-moi",
    # Culture / loisirs (sans demande de trajet)
    "quel film à l'affiche",
    "c'est quoi ce spectacle",
    "je voudrais voir une exposition",
    "y a-t-il un concert ce soir",
    "où est le cinéma",
    "je cherche une librairie",
    "vous avez des places",
    # Travail / admin
    "je dois envoyer un mail",
    "où signer",
    "c'est pour un rendez-vous",
    "j'ai un entretien demain",
    "je cherche un stage",
    "vous avez des brochures",
    "je voudrais un reçu",
    # Nonsense / hors sujet
    "le chat est sur le tapis",
    "il pleut des cordes",
    "c'est la fin des haricots",
    "je m'en moque",
    "n'importe quoi",
    "bof",
    "peut-être",
    "on verra",
]

# Contexte recette / pâtisserie : Paris-Brest, religieuse = pâtisseries, pas des trajets → INVALID
RECIPE_INVALID_PATTERNS = [
    "Recette de cuisine pour un Paris-Brest",
    "Recette du Paris-Brest",
    "Je veux une religieuse à la pâtisserie",
    "Je commande un Paris-Brest",
    "Recette de la religieuse",
    "Un Paris-Brest et un café",
    "recette de cuisine pour un paris-brest",
    "Où trouver une bonne religieuse",
    "Ingrédients pour un Paris-Brest",
    "Je prépare un Paris-Brest pour le dessert",
    "La recette du Paris-Brest de ma grand-mère",
    "Différence entre Paris-Brest et religieuse",
]

# Prénoms qui sont aussi des noms de villes (pour les pièges)
PRENOMS_VILLES = {
    "Albert": "albert",
    "Paris": "paris",
    "Florence": "florence",
    "Monaco": "monaco",  # aussi une boisson
    "Nancy": "nancy",
    "Valérie": "valerie",  # proche de Valence
}

# Patterns de phrases pièges (avec prénoms ou mots ambigus)
TRAP_PATTERNS_VALID = [
    # Prénom mentionné mais ville réelle présente
    lambda d, a, prenom: f"avec mon ami {prenom} je voudrais aller de {d} à {a}",
    lambda d, a, prenom: f"je vais voir {prenom} à {a} en partant de {d}",
    lambda d, a, prenom: f"en compagnie de {prenom} je veux aller de {d} à {a}",
    lambda d, a, prenom: f"mon ami {prenom} et moi allons de {d} à {a}",
    # Cas spécial : prénoms qui sont aussi villes (exemple du PDF)
    lambda d, a, prenom: f"avec mes amis {prenom} et {a.lower()} je voudrais aller de {a.lower()} à {prenom}",
    lambda d, a, prenom: f"avec {prenom} je veux aller de {d} à {a}",
    lambda d, a, prenom: f"je veux aller voir mon ami {prenom} à {a} en partant de {d}",
]

TRAP_PATTERNS_INVALID = [
    # Prénom utilisé comme ville (avec vraie ville)
    lambda d, a, prenom: f"je voudrais aller de {prenom} à {a}",
    lambda d, a, prenom: f"aller de {d} à {prenom}",
    lambda d, a, prenom: f"trains de {prenom} vers {a}",
    # Exemple spécifique : boire un monaco (avec vraies villes)
    lambda d, a, prenom: f"je veux aller de {d} à {a} boire un monaco",
    lambda d, a, prenom: f"aller de {prenom} à {a} boire un monaco",
    lambda d, a, prenom: f"depuis {prenom} je veux aller à {a} pour boire un monaco",
    # Avec deux prénoms (tous les deux prénoms, pas villes)
    lambda d, a, prenom: f"je voudrais aller de {prenom} à albert",
    lambda d, a, prenom: f"trains de {prenom} vers nancy",
    lambda d, a, prenom: f"depuis {prenom} aller à florence",
    # Contexte avec prénom mais pas de ville claire
    lambda d, a, prenom: f"mon ami {prenom} veut aller quelque part",
    lambda d, a, prenom: f"je dois rejoindre {prenom} quelque part",
    lambda d, a, prenom: f"je veux aller voir {prenom} et boire un monaco",
    # Phrases avec contexte confus (prénom mélangé avec vraie ville mais mal utilisé)
    lambda d, a, prenom: f"avec mes amis {prenom} et {a} je voudrais aller voir paris",
    lambda d, a, prenom: f"mon ami {prenom} veut aller à {a} voir florence",
]

def add_variations(phrase: str) -> List[str]:
    """Ajoute des variations à une phrase (sans accents, fautes, etc.)"""
    variations = [phrase]
    
    # Version sans accents (approximative)
    no_accents = phrase.replace('é', 'e').replace('è', 'e').replace('ê', 'e')
    no_accents = no_accents.replace('à', 'a').replace('â', 'a')
    no_accents = no_accents.replace('ù', 'u').replace('û', 'u')
    no_accents = no_accents.replace('ô', 'o').replace('ö', 'o')
    no_accents = no_accents.replace('î', 'i').replace('ï', 'i')
    no_accents = no_accents.replace('ç', 'c')
    if no_accents != phrase:
        variations.append(no_accents)
    
    # Version tout en minuscules
    if phrase != phrase.lower():
        variations.append(phrase.lower())
    
    # Version avec quelques fautes courantes
    fautes = phrase.replace('je voudrais', 'je voudrai').replace('je veux', 'je veu')
    if fautes != phrase:
        variations.append(fautes)
    
    # Version sans majuscule au début
    if phrase and phrase[0].isupper():
        variations.append(phrase[0].lower() + phrase[1:])
    
    # Version avec tirets manquants/ajoutés
    if '-' in phrase:
        variations.append(phrase.replace('-', ' '))
    elif ' ' in phrase and random.random() < 0.1:
        parts = phrase.split()
        if len(parts) >= 2:
            variations.append('-'.join(parts[:2]) + ' ' + ' '.join(parts[2:]))
    
    return list(set(variations))

def generate_valid_sentences(num_sentences: int) -> List[Tuple[str, str, str]]:
    """Génère des phrases valides"""
    sentences = []
    
    # Générer toutes les combinaisons de villes
    ville_pairs = list(itertools.permutations(VILLES, 2))
    
    for _ in range(num_sentences):
        # Choisir une structure et une paire de villes
        pattern = random.choice(VALID_PATTERNS)
        depart, arrivee = random.choice(ville_pairs)
        
        # Générer la phrase de base
        base_phrase = pattern(depart.lower(), arrivee.lower())
        
        # Ajouter des variations
        variations = add_variations(base_phrase)
        sentence = random.choice(variations)
        
        sentences.append((sentence, depart, arrivee))
    
    return sentences

def generate_invalid_sentences(num_sentences: int) -> List[Tuple[str, str, str]]:
    """Génère des phrases invalides"""
    sentences = []

    for _ in range(num_sentences):
        # Parfois utiliser un pattern, parfois aléatoire
        if random.random() < 0.7:
            sentence = random.choice(INVALID_PATTERNS)
        else:
            # Phrase aléatoire qui pourrait ressembler à une demande de voyage
            mots_aleatoires = [
                "je voudrais", "aller", "trains", "billet", "voyage",
                "comment", "depuis", "vers", "à", "de"
            ]
            sentence = ' '.join(random.sample(mots_aleatoires, random.randint(3, 6)))

        # Ajouter des variations
        variations = add_variations(sentence)
        sentence = random.choice(variations)

        sentences.append((sentence, "INVALID", ""))

    return sentences


def generate_recipe_invalid_sentences(num_sentences: int) -> List[Tuple[str, str, str]]:
    """Phrases recette / pâtisserie (Paris-Brest, religieuse) = INVALID, pas des trajets."""
    sentences = []
    for _ in range(num_sentences):
        sentence = random.choice(RECIPE_INVALID_PATTERNS)
        variations = add_variations(sentence)
        sentence = random.choice(variations)
        sentences.append((sentence, "INVALID", ""))
    return sentences


def generate_city_context_invalid_sentences(num_sentences: int) -> List[Tuple[str, str, str]]:
    """Phrases avec noms de villes mais pas une demande de trajet (sport, culture, visite, etc.) → INVALID."""
    sentences = []
    # Templates à 1 ville
    one_city = [
        lambda v: f"J'ai visité {v} l'année dernière",
        lambda v: f"J'ai été à {v} en vacances",
        lambda v: f"Mon ami habite à {v}",
        lambda v: f"Je suis né à {v}",
        lambda v: f"Le musée de {v} est magnifique",
        lambda v: f"J'adore {v}",
        lambda v: f"On a passé le week-end à {v}",
        lambda v: f"La météo à {v} est bonne",
        lambda v: f"Tu connais {v}",
        lambda v: f"L'équipe de {v} a gagné",
        lambda v: f"Le match à {v} était nul",
        lambda v: f"Le concert à {v} est annulé",
        lambda v: f"Le festival de {v} a lieu en juillet",
        lambda v: f"La gare de {v} est en travaux",
        lambda v: f"Combien d'habitants à {v}",
        lambda v: f"Le maire de {v}",
        lambda v: f"L'aéroport de {v}",
    ]
    # Templates à 2 villes
    two_cities = [
        lambda v1, v2: f"Le match {v1} {v2} c'est quand",
        lambda v1, v2: f"Résultat du match {v1} contre {v2}",
        lambda v1, v2: f"Finale {v1} {v2}",
        lambda v1, v2: f"La distance {v1} {v2} en voiture",
        lambda v1, v2: f"Différence entre {v1} et {v2}",
    ]
    ville_pairs = list(itertools.permutations(VILLES, 2))
    for _ in range(num_sentences):
        if random.random() < 0.6:
            template = random.choice(one_city)
            ville = random.choice(VILLES)
            base = template(ville.lower())
        else:
            template = random.choice(two_cities)
            v1, v2 = random.choice(ville_pairs)
            base = template(v1.lower(), v2.lower())
        variations = add_variations(base)
        sentence = random.choice(variations)
        sentences.append((sentence, "INVALID", ""))
    return sentences

def generate_trap_sentences(num_valid_traps: int, num_invalid_traps: int) -> List[Tuple[str, str, str]]:
    """Génère des phrases pièges avec prénoms/ambiguités"""
    sentences = []
    
    # Villes normales (sans prénoms)
    villes_normales = [v for v in VILLES if v not in PRENOMS_VILLES]
    ville_pairs = list(itertools.permutations(villes_normales, 2))
    
    prenoms = list(PRENOMS_VILLES.values())
    
    # Phrases pièges VALIDES : prénom mentionné mais vraie ville présente
    for _ in range(num_valid_traps):
        pattern = random.choice(TRAP_PATTERNS_VALID)
        depart, arrivee = random.choice(ville_pairs)
        prenom = random.choice(prenoms)
        
        base_phrase = pattern(depart.lower(), arrivee.lower(), prenom)
        variations = add_variations(base_phrase)
        sentence = random.choice(variations)
        
        sentences.append((sentence, depart, arrivee))
    
    # Phrases pièges INVALIDES : prénom utilisé comme ville
    for _ in range(num_invalid_traps):
        pattern = random.choice(TRAP_PATTERNS_INVALID)
        prenom = random.choice(prenoms)
        
        # Cas spécial : exemple "aller de albert à paris boire un monaco"
        # Si prenom est "albert" ou "monaco", utiliser des patterns spécifiques
        if prenom == "albert" and random.random() < 0.15:
            # Albert comme prénom (pas ville) - exemple donné par l'utilisateur
            arrivee = random.choice([v for v in villes_normales if v.lower() != "albert"])
            base_phrase = f"aller de {prenom} à {arrivee.lower()} boire un monaco"
        elif prenom == "monaco" and random.random() < 0.15:
            # Monaco comme boisson (pas ville)
            depart = random.choice([v for v in villes_normales if v.lower() != "monaco"])
            base_phrase = f"aller de {depart.lower()} à paris boire un {prenom}"
        else:
            # La plupart du temps, utiliser une vraie ville avec un prénom piège
            if random.random() < 0.7:
                # Avec une vraie ville + prénom piège utilisé comme ville
                if random.random() < 0.5:
                    # Prénom en départ (le pattern le remplacera)
                    arrivee = random.choice(villes_normales)
                    # Passer prenom comme départ pour que le pattern l'utilise
                    base_phrase = pattern(prenom, arrivee.lower(), prenom)
                else:
                    # Prénom en arrivée
                    depart = random.choice(villes_normales)
                    # Passer prenom comme arrivée pour que le pattern l'utilise
                    base_phrase = pattern(depart.lower(), prenom, prenom)
            else:
                # Deux vraies villes mais pattern utilise un prénom à la place
                depart, arrivee = random.choice(ville_pairs)
                base_phrase = pattern(depart.lower(), arrivee.lower(), prenom)
        
        variations = add_variations(base_phrase)
        sentence = random.choice(variations)
        
        sentences.append((sentence, "INVALID", ""))
    
    return sentences

def main():
    """Génère le dataset complet (~100k phrases)"""
    print("Génération du dataset (~100k phrases)...")

    # Répartition pour ~100k : valides, invalides générales, villes en contexte (INVALID), pièges, recette
    num_valid = 51_000
    num_invalid = 25_000
    num_city_context_invalid = 12_000
    num_valid_traps = 5_000
    num_invalid_traps = 5_000
    num_recipe_invalid = 2_000

    valid_sentences = generate_valid_sentences(num_valid)
    invalid_sentences = generate_invalid_sentences(num_invalid)
    city_context_invalid = generate_city_context_invalid_sentences(num_city_context_invalid)
    trap_sentences_valid = generate_trap_sentences(num_valid_traps, 0)
    trap_sentences_invalid = generate_trap_sentences(0, num_invalid_traps)
    recipe_invalid_sentences = generate_recipe_invalid_sentences(num_recipe_invalid)

    # Mélanger toutes les phrases
    all_sentences = (
        valid_sentences
        + invalid_sentences
        + city_context_invalid
        + trap_sentences_valid
        + trap_sentences_invalid
        + recipe_invalid_sentences
    )
    random.shuffle(all_sentences)
    
    # Écrire le fichier d'entrée
    with open('dataset_input.txt', 'w', encoding='utf-8') as f_input:
        with open('dataset_output.txt', 'w', encoding='utf-8') as f_output:
            for sentence_id, (sentence, depart, arrivee) in enumerate(all_sentences, 1):
                # Format d'entrée: sentenceID,phrase
                f_input.write(f"{sentence_id},{sentence}\n")
                
                # Format de sortie: sentenceID,Departure,Destination
                if depart == "INVALID":
                    f_output.write(f"{sentence_id},INVALID,\n")
                else:
                    f_output.write(f"{sentence_id},{depart},{arrivee}\n")
    
    print(f"✅ Dataset généré avec succès!")
    print(f"   - Phrases valides (trajet): {num_valid}")
    print(f"   - Phrases valides pièges: {num_valid_traps}")
    print(f"   - Phrases invalides (hors voyage): {num_invalid}")
    print(f"   - Villes en contexte non-trajet (INVALID): {num_city_context_invalid}")
    print(f"   - Phrases invalides pièges: {num_invalid_traps}")
    print(f"   - Phrases recette/pâtisserie (INVALID): {num_recipe_invalid}")
    print(f"   - Total: {len(all_sentences)} phrases")
    print(f"   - Fichiers créés: dataset_input.txt, dataset_output.txt")

if __name__ == "__main__":
    random.seed(42)  # Pour la reproductibilité
    main()

