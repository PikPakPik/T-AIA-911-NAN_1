# T-AIA-911 - SpaCy

Note: The finetuned_output_en_core_news_lg and finetuned_output_en_core_news_md folders are not included in this repository because they contain files that are too large.

## Description
This directory contains a proof of concept (PoC) for using SpaCy, a popular natural language processing library, to perform named entity recognition (NER) on French text. The PoC includes a script that loads a specified SpaCy model and processes a sample French text to extract and display named entities.

## How to run the PoC with trained SpaCy models

### CLI Parameters
- `--file`: Path to a text file to analyze.
- `--url`: URL of a text document to analyze.
- `--model`: Path to the trained Spacy model (default: `fr_core_news_lg`) available (`fr_core_news_md`, `fr_core_news_lg`, `./finetuned_output_fr_core_news_lg/model-best`, `./finetuned_output_fr_core_news_lg/model-last`, `./finetuned_output_fr_core_news_md/model-best`, `./finetuned_output_fr_core_news_md/model-last`).
- `--output`: Path to save the output results (default: `./output/dataset_output.csv`).

### Command to run the PoC
```bash
python3 poc_spacy.py {--file OR --url}
```

## How to run the PoC with custom trained SpaCy models

### CLI Parameters
- `--model`: Train a custom model based on an existing model (default: `fr_core_news_lg`) available (`fr_core_news_md`, `fr_core_news_lg`).

### Command to run the PoC
```bash
python3 poc_spacy_training.py
```