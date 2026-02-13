# Training NER (ALEXANDRE)

Scripts d'entraînement, d'inférence CLI et d'évaluation pour les modèles NER.

## Scripts principaux

- `poc_roberta_training.py`: fine-tuning token-classification (LOC/BIO) depuis les datasets texte.
- `poc_roberta.py`: inférence batch (fichier/URL) vers CSV.
- `chat_roberta_loc.py`: chat CLI phrase par phrase.

## Entraîner un modèle

Depuis la racine du repo:

```bash
python3 model/ALEXANDRE/training/poc_roberta_training.py \
  --dataset-dir ./dataset \
  --output-dir ./roberta_ner_model \
  --seed 42
```

## Tester en batch

```bash
python3 model/ALEXANDRE/training/poc_roberta.py \
  --file dataset/generated/dataset_input.txt \
  --model ./roberta_ner_model \
  --output dataset_output_roberta.csv
```

## Tester en mode chat CLI

```bash
python3 model/ALEXANDRE/training/chat_roberta_loc.py --model ./roberta_ner_model
```

## Notebooks utiles

- `model/ALEXANDRE/evaluate_roberta_model.ipynb`
- `model/ALEXANDRE/finetune_bert_europeana.ipynb`
- `model/ALEXANDRE/test_bert_model.py` (CLI de test ponctuel)

## Notes

- Le modèle local est attendu dans `./roberta_ner_model`.
- Les scripts essaient d'utiliser CUDA/MPS quand disponible.
