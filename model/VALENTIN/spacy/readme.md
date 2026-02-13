# SpaCy PoC (VALENTIN)

PoC NER en français avec SpaCy.

## Contenu

- `poc_spacy.py`: inférence NER sur fichier ou URL.
- `poc_spacy_training.py`: entraînement/fine-tuning d'un modèle SpaCy.
- `finetune_config_*.cfg`: configurations d'entraînement.
- `dataset_output_*.csv`: sorties d'expérimentation.

Note: certains dossiers de modèles fine-tunés ne sont pas versionnés (fichiers trop volumineux).

## Lancer l'inférence

Depuis `model/VALENTIN/spacy`:

```bash
python3 poc_spacy.py --file path/to/input.txt
```

Paramètres utiles:
- `--file`: fichier texte à analyser.
- `--url`: URL d'un texte à analyser.
- `--model`: modèle SpaCy (défaut `fr_core_news_lg`).
- `--output`: CSV de sortie.

## Lancer l'entraînement

```bash
python3 poc_spacy_training.py
```

Le script accepte un modèle de base (ex: `fr_core_news_md` ou `fr_core_news_lg`) et entraîne un modèle NER custom.