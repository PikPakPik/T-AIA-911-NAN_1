# FlauBERT baseline (LEO)

Ce dossier contient un pipeline baseline autour de FlauBERT:
- `train.py`: entraînement d'un classifieur binaire (VALID/INVALID);
- `inference.py`: inférence + extraction règle/dictionnaire;
- `evaluate.py`: évaluation complète (métriques + confusion matrix).

## Installation

```bash
pip install -r model/LEO/flaubert/requirements.txt
```

## Entraîner

Le script utilise des chemins relatifs vers `../dataset/...`.
Lance-le depuis `model/LEO/flaubert` ou adapte les chemins.

```bash
cd model/LEO/flaubert
python3 train.py
```

## Inférence

```bash
cd model/LEO/flaubert
python3 inference.py
```

## Évaluation

```bash
cd model/LEO/flaubert
python3 evaluate.py
```

Des fichiers de sortie sont générés (matrices de confusion, rapport texte).
