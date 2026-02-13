# T-AIA-911

Projet NLP + pathfinding SNCF autour de la détection des villes de départ/arrivée.

## Démarrage rapide

```bash
pyenv install 3.12.8
pyenv local 3.12.8
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
```

Le projet n'a pas un seul `requirements.txt` global. Installe les dépendances selon le module que tu utilises (voir README par dossier ci-dessous).

## Point d'entrée principal

Le script `T-AIA-911.py` permet:
- extraction NER depuis un fichier/URL/texte;
- mode chat terminal;
- lancement de l'UI Streamlit de pathfinding.

Exemples:

```bash
python3 T-AIA-911.py --chat --model Jean-Baptiste/camembert-ner
python3 T-AIA-911.py --text "Je veux aller de Paris a Lyon" --model Jean-Baptiste/camembert-ner
python3 T-AIA-911.py --ui --model Jean-Baptiste/camembert-ner
```

## Arborescence docs

- `model/ALEXANDRE/README.md`: index des travaux Alexandre
- `model/ALEXANDRE/training/README.md`: entraînement et évaluation RoBERTa/BERT
- `model/ALEXANDRE/pathfinding/README.md`: moteur SNCF + app chat/audio Streamlit
- `model/BAPTISTE/camembert/README.md`: pipeline CamemBERT (data, train, test, eval)
- `model/LEO/stt/README.md`: transcription audio (WAV + micro)
- `model/LEO/flaubert/README.md`: baseline FlauBERT (classification VALID/INVALID)
- `model/VALENTIN/readme.md` et `model/VALENTIN/spacy/readme.md`: PoC SpaCy + pathfinding géographique
- `model/YANN/README.md`: checkpoints CamemBERT ORIGIN/DEST

## Données

- Données GTFS SNCF: `dataset/sncf/`
- Données textuelles (générées ou historiques): `dataset/`
- Générateurs historiques: `dataset/old/`

## Notes

- Plusieurs scripts utilisent des chemins relatifs: lance-les depuis leur dossier ou adapte les chemins CLI.
- Sur Apple Silicon, PyTorch peut utiliser MPS automatiquement quand disponible.
