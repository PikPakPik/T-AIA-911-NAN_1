# Pathfinding SNCF + Chat UI

Ce dossier contient:
- `sncf_pathfinder.py`: construction du graphe SNCF GTFS + recherche de trajet;
- `chat_pathfinding_app.py`: app Streamlit (chat + audio + carte);
- `pathfinding_sncf.ipynb`: notebook d'exploration/validation.

## Lancer l'app Streamlit

Depuis la racine du repo:

```bash
pip install streamlit pydeck pandas torch transformers
streamlit run model/ALEXANDRE/pathfinding/chat_pathfinding_app.py
```

## Fonctionnement de l'app

1. Charge un modèle NER (local ou Hugging Face).
2. Extrait les villes depuis la phrase utilisateur.
3. Calcule le trajet sur le graphe SNCF.
4. Affiche un recap des stations et une carte.
5. Option audio: upload WAV ou micro via `model/LEO/stt`.

## Données nécessaires

- GTFS SNCF dans `dataset/sncf/` (`stops.txt`, `routes.txt`, `trips.txt`, `stop_times.txt`).

## Lancer le notebook

```bash
jupyter notebook model/ALEXANDRE/pathfinding/pathfinding_sncf.ipynb
```

## Point d'entrée alternatif

```bash
python3 T-AIA-911.py --ui --model Jean-Baptiste/camembert-ner
```
