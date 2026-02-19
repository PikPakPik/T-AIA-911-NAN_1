# T-AIA-911 — Travel Order Resolver

Programme de traitement de commandes texte pour générer des itinéraires de train SNCF.

## Architecture

```
final_project/
├── __init__.py        # Description du package
├── __main__.py        # python -m final_project
├── cli.py             # Point d'entrée CLI (--file, --url, --text, --chat, --ui)
├── app.py             # Interface Streamlit (chat + carte + horaires)
├── ner.py             # Chargement NER et extraction d'entités
├── pathfinder.py      # Graphe GTFS, Dijkstra, horaires de trains
├── stt.py             # Speech-to-text (optionnel)
├── utils.py           # Fonctions utilitaires partagées
└── requirements.txt   # Dépendances Python
```

## Installation

```bash
pip install -r final_project/requirements.txt
```

## Utilisation

### UI Streamlit (recommandé)

```bash
# Depuis la racine du projet
streamlit run final_project/app.py

# Ou via le CLI
python final_project/cli.py --ui
```

### CLI — Extraction NER depuis un fichier

Format d'entrée : une phrase par ligne, `sentenceID,sentence`

```bash
python final_project/cli.py --file input.txt --output results.csv
```

Sortie : `sentenceID,Departure,[Steps...],Destination`

### CLI — Extraction NER depuis une URL

```bash
python final_project/cli.py --url https://example.com/sentences.txt
```

### CLI — Texte direct

```bash
python final_project/cli.py --text "Je veux aller de Nantes à Paris"
```

### CLI — Mode chat interactif

```bash
python final_project/cli.py --chat
```

### Options

| Option     | Description                                          | Défaut                      |
|------------|------------------------------------------------------|-----------------------------|
| `--model`  | Modèle NER (nom HF ou chemin local)                 | `PikPak44/yann_best_model`  |
| `--output` | Chemin du CSV de sortie (avec `--file` ou `--url`)   | `dataset/dataset_output.csv`|

## Modèle NER

Par défaut, le modèle est téléchargé automatiquement depuis Hugging Face Hub :
- **`PikPak44/yann_best_model`** — modèle custom entraîné avec labels ORIGIN/DEST

En mode UI (Streamlit), un sélecteur permet de choisir entre :
- Hugging Face (téléchargement automatique avec cache)
- Chemin local vers un modèle

## Données GTFS

Les fichiers SNCF GTFS sont attendus dans `dataset/sncf/` :
- `stops.txt` — liste des gares
- `stop_times.txt` — horaires d'arrêt
- `trips.txt` — liste des trajets
- `routes.txt` — lignes

## Composants

### NER (`ner.py`)
- Supporte CamemBERT, RoBERTa, BERT avec poids TF
- Extraction d'entités LOC ou DEP/ARR/ORIGIN/DEST
- Filtrage des faux positifs (recettes, noms de personnes)
- Téléchargement depuis Hugging Face Hub

### Pathfinding (`pathfinder.py`)
- Construction du graphe pondéré à partir des fichiers GTFS
- Dijkstra pour le plus court chemin
- Recherche de gares par nom (fuzzy matching)
- Fallback géolocalisation via `geo.csv` pour villes sans gare
- Itinéraires multi-étapes (waypoints)
- Horaires concrets avec correspondances et temps d'attente

### Interface (`app.py`)
- Chat Streamlit avec historique
- Carte interactive (pydeck)
- Horaires détaillés avec correspondances
- Support audio (STT optionnel)

### STT (`stt.py`)
- Bridge vers le module STT de Leo
- Transcription depuis fichier WAV ou microphone
