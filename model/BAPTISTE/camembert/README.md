# CamemBERT NER (BAPTISTE)

Pipeline CamemBERT pour NER de localisation.

## Scripts

- `generate_training_data.py`: génération de données d'entraînement SpaCy.
- `train_camembert.py`: entraînement du modèle token-classification.
- `test_model.py`: test sur une phrase.
- `evaluate_camembert.py`: évaluation complète + exports.

## Génération des données

Le script `generate_training_data.py` attend un fichier `geo.csv` dans le dossier courant.
Si besoin, partir de `dataset/old/geo.csv`.

```bash
cd model/BAPTISTE/camembert
python3 generate_training_data.py
```

## Entraînement

```bash
python3 model/BAPTISTE/camembert/train_camembert.py \
  --train path/to/train.spacy \
  --dev path/to/dev.spacy \
  --output model/BAPTISTE/camembert/camembert_output
```

## Test rapide

```bash
python3 model/BAPTISTE/camembert/test_model.py \
  --model model/BAPTISTE/camembert/camembert_output \
  --text "Je veux aller de Paris a Lyon"
```

## Évaluation

```bash
python3 model/BAPTISTE/camembert/evaluate_camembert.py \
  --model model/BAPTISTE/camembert/camembert_output \
  --test path/to/test.spacy \
  --output model/BAPTISTE/camembert/results
```
