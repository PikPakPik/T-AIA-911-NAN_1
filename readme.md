# T-AIA-911 — Travel Order Resolver

Projet NLP + pathfinding SNCF : extraction de villes de départ/arrivée depuis du texte français, puis calcul d'itinéraire avec horaires concrets.

## Démarrage rapide

```bash
pyenv install 3.12.8
pyenv local 3.12.8
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r final_project/requirements.txt
```

## Utilisation

Tout se lance depuis le dossier `final_project/` :

### Interface Streamlit (recommandé)

```bash
streamlit run final_project/app.py
```

Chat IA avec extraction NER automatique, carte du trajet, horaires détaillés et correspondances.

### CLI — Extraction NER depuis un fichier (format sujet)

Format d'entrée : `sentenceID,sentence` (une par ligne, UTF-8).

```bash
python final_project/cli.py --file input.txt --output results.csv
```

Sortie : `sentenceID,Departure,[Steps...],Destination` ou `sentenceID,INVALID`

### CLI — Extraction NER depuis une URL

```bash
python final_project/cli.py --url https://example.com/sentences.txt
```

### CLI — Texte direct

```bash
python final_project/cli.py --text "Je veux aller de Nantes à Paris"
```

### CLI — Mode chat terminal

```bash
python final_project/cli.py --chat
```

### Options

| Option     | Description                                   | Défaut                     |
|------------|-----------------------------------------------|----------------------------|
| `--model`  | Modèle NER (nom HF ou chemin local)          | `PikPak44/yann_best_model` |
| `--output` | Chemin du CSV de sortie (`--file` / `--url`)  | `dataset/dataset_output.csv` |

Le modèle par défaut (`PikPak44/yann_best_model`) est téléchargé automatiquement depuis Hugging Face Hub au premier lancement.

## Architecture du projet final

```
final_project/
├── cli.py             # Point d'entrée CLI (--file, --url, --text, --chat, --ui)
├── app.py             # Interface Streamlit (chat + carte + horaires)
├── ner.py             # Chargement NER et extraction d'entités (LOC / ORIGIN / DEST)
├── pathfinder.py      # Graphe GTFS, Dijkstra, recherche de gares, horaires
├── stt.py             # Speech-to-text optionnel (bridge vers model/LEO/stt)
├── utils.py           # Fonctions utilitaires partagées
├── requirements.txt   # Dépendances Python
└── README.md          # Documentation détaillée du package
```

## Dossiers de travail (recherche & expérimentations)

Le dossier `model/` contient les travaux individuels de chaque membre :

- `model/ALEXANDRE/` — tokenisation, entraînement RoBERTa/BERT, pathfinding SNCF, app Streamlit
- `model/BAPTISTE/` — pipeline CamemBERT
- `model/LEO/` — STT (speech-to-text), baseline FlauBERT
- `model/VALENTIN/` — PoC SpaCy, pathfinding géographique
- `model/YANN/` — checkpoints CamemBERT ORIGIN/DEST

## Données

- `dataset/sncf/` — fichiers GTFS SNCF (stops, stop_times, trips, routes)
- `dataset/geo.csv` — 37 598 communes françaises avec coordonnées GPS
- `dataset/` — datasets NER (texte d'entrée/sortie, générés et manuels)

## Notes

- Plusieurs scripts utilisent des chemins relatifs: lance-les depuis leur dossier ou adapte les chemins CLI.
- Sur Apple Silicon, PyTorch peut utiliser MPS automatiquement quand disponible.
