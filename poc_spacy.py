import spacy

PHRASES = [
    "Je voudrais organiser un voyage en train de Toulouse à Paris avec arrêts à Limoges et Poitiers, départ vendredi matin, retour dimanche soir.",
    "Réserver un aller-retour Lyon → Nice en s'arrêtant à Valence et Avignon, partir jeudi après-midi, billet flexible.",
    "Partir de Bordeaux vers Strasbourg via Tours et Orléans, départ samedi matin, retour lundi soir.",
    "Aller de Marseille à Lille avec étapes à Nîmes et Montpellier, partir mercredi matin, options bagages incluses.",
    "Organiser un trajet Nantes → Grenoble en s'arrêtant à Angers et Le Mans, partir dimanche soir, billet échangeable.",
    "Voyage Paris à Bruxelles avec escales à Lille et Amiens, partir vendredi, revenir dimanche soir.",

    "Je veux un voyage en train de Toulouse à Paris vendredi matin, aller-retour flexible, sans étapes.",
    "Réserver un billet aller simple Lyon → Nice samedi matin, trajet direct sans arrêts.",
    "Aller-retour Bordeaux → Marseille, départ lundi, trajet direct, siège préférentiel.",
    "Voyage direct Nantes → Rennes dimanche, bagages et restauration inclus.",
    "Partir de Strasbourg pour Lille jeudi soir, trajet sans correspondance.",
    "Organiser un trajet Paris → Lyon dimanche matin, billet flexible, direct.",

    "Je cherche un voyage partant de Toulouse la semaine prochaine, destination ouverte selon budget et durée.",
    "Départ de Lyon vendredi matin, destination à définir, préférence pour arrêts courts.",
    "Je voudrais partir de Bordeaux samedi, retour dimanche, destination ouverte selon le prix.",
    "Partant de Marseille jeudi, sans destination précise, recherche trajets confortables.",
    "Départ Nantes vers une ville au choix, dates flexibles, escales acceptées.",
    "Quitter Paris vendredi matin sans destination fixée, préférence pour trajets rapides.",

    "Arriver à Paris la semaine prochaine, dates flexibles, préférence pour correspondances courtes.",
    "Je souhaite être à Nice samedi soir, origine indifférente, billet flexible.",
    "Besoin d'arriver à Strasbourg dimanche matin, départ non précisé, éviter longues correspondances.",
    "Arrivée à Lille vendredi, options de départ à définir, bagages et restauration requis.",
    "Vouloir être à Grenoble mardi, départ ouvert, accepter correspondances.",
    "Arriver à Bruxelles dimanche soir, origines flexibles, préférence pour trajets rapides."
]

for phrase in PHRASES:
    nlp = spacy.load("fr_core_news_md")
    doc = nlp(phrase)
    print(f"Phrase: {phrase}")
    lieux = [ent.text for ent in doc.ents if ent.label_ == "LOC"]
    print(f"Lieux détectés : {lieux}\n")