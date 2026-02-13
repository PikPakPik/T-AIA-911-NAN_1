# VALENTIN - SpaCy et PoC Pathfinding

Ce dossier contient deux pistes exploratoires:

- `spacy/`: PoC NER avec SpaCy (entraînement + inférence).
- `pathfinding.py`: PoC de calcul d'itinéraire par distance géographique (Haversine).

## Statut

Le pipeline SpaCy est conservé pour référence, mais n'est pas la solution principale utilisée dans l'app finale.

## Lancer le PoC pathfinding

```bash
python3 model/VALENTIN/pathfinding.py
```

Le script expose une classe `PathFinder` qui:
- charge les lieux depuis un CSV géographique;
- calcule des distances entre villes;
- estime une route avec points intermédiaires;
- propose un ordre de visite "plus proche voisin" pour plusieurs lieux.

## Lancer le PoC SpaCy

Voir `model/VALENTIN/spacy/readme.md`.