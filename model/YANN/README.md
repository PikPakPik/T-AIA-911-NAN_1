# YANN - CamemBERT checkpoints

Ce dossier contient des artefacts de modèle CamemBERT orientés NER départ/arrivée.

## Contenu

- `camembert-v2/`
  - `camembert-v2.ipynb`: notebook d'expérimentation/entraînement.
  - `config.json`, `tokenizer.json`, `tokenizer_config.json`: configuration tokenizer/modèle.
- `best_model/`
  - checkpoint sauvegardé (config + tokenizer).

Les labels configurés sont:
- `O`
- `B-ORIGIN`, `I-ORIGIN`
- `B-DEST`, `I-DEST`

## Utilisation (inférence)

Exemple de chargement via Hugging Face Transformers:

```python
from transformers import pipeline

ner = pipeline(
    "ner",
    model="model/YANN/camembert-v2",
    tokenizer="model/YANN/camembert-v2",
    aggregation_strategy="simple",
)
print(ner("Je veux aller de Paris à Lyon"))
```

## Notes

- Ce dossier contient surtout des artefacts; le pipeline d'app principal utilise actuellement les composants documentés dans `model/ALEXANDRE/pathfinding`.
- Si tu veux une intégration directe dans l'app, il faut aussi versionner les poids complets du modèle (`pytorch_model.bin` ou `model.safetensors`) si absents.
