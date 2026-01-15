# T-AIA-911

## Initialize the environment
```bash
pyenv install 3.12.8
```

## Spacy
### Train the model
```bash
python3 poc_spacy_training.py
python3 -m spacy train config.cfg --output output --paths.train train.spacy --paths.dev dev.spacy --paths.base_model fr_core_news_lg
```

## Test the model
### Command to run the PoC
```bash
python3 poc_spacy.py
```
### CLI Parameters
- `--file`: Path to a text file to analyze.
- `--url`: URL of a text document to analyze.
- `--model`: Path to the trained Spacy model (default: `fr_core_news_lg`).
- `--output`: Path to save the output results (default: `output/dataset_output.csv`).

## RoBERTa
### Train the model
```bash
python3 poc_roberta_training.py
```
This will fine-tune `benjamin/roberta-base-wechsel-french` for NER tasks. The trained model will be saved in `./roberta_ner_model/`.

### Test the model
### Command to run the PoC
```bash
python3 poc_roberta.py --file dataset/dataset_input.txt
```
### CLI Parameters
- `--file`: Path to a text file to analyze.
- `--url`: URL of a text document to analyze.
- `--model`: Model to use (default: `Jean-Baptiste/camembert-ner`). Options:
  - `Jean-Baptiste/camembert-ner`: Pre-trained NER model
  - `benjamin/roberta-base-wechsel-french`: Base model (needs fine-tuning)
  - `./roberta_ner_model`: Fine-tuned model (after training)
- `--output`: Path to save the output results (default: `dataset_output_roberta.csv`).

### Evaluate the model
To evaluate the fine-tuned model with metrics, visualizations, and detailed analysis:
```bash
jupyter notebook evaluate_roberta_model.ipynb
```
or
```bash
jupyter lab evaluate_roberta_model.ipynb
```

The notebook includes:
- Model loading and testing on specific examples
- Full dataset evaluation
- Performance metrics (accuracy, precision, recall, F1)
- Error analysis
- Visualizations (bar charts, confusion matrix, pie charts, top cities)
- Detailed evaluation report
- Export of results to CSV

### Analyze the dataset
To explore and understand the dataset structure, patterns, and statistics:
```bash
jupyter notebook analyze_dataset.ipynb
```

The notebook includes:
- General statistics and data overview
- Distribution analysis (VALID vs INVALID)
- City frequency analysis
- Phrase pattern detection
- Trap analysis (monaco, albert, etc.)
- Word frequency analysis
- City pair analysis
- Length distribution visualizations
- Export of statistics to JSON
