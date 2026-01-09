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
