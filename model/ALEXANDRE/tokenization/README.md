# Tokenizers pour la détection Départ/Arrivée (NER)

Ce dossier contient plusieurs approches de tokenisation pour annoter des phrases en format BIO.

## Scripts

### 1) NLTK - `script1_nltk_basic.py`
- Méthode: `word_tokenize` (français).
- Exemple de sortie: `script1_nltk_output.json`.

### 2) Regex v1 - `script2_regexp_basic.py` / `script2_regex_v1.py`
- Contractions souvent collées au mot.
- Exemple de sortie: `script2_output.json`.

### 3) Regex v2 améliorée - `script3_regexp_improved.py`
- Contractions séparées du mot.
- Utilisée comme base d'alignement dans `finetune_bert_europeana.py` (sauf `--no-script3`).
- Exemple de sortie: `script3_output.json`.

### 4) Tokenizers pré-entraînés - `script4_pretrained_tokenizers.py`
- Compare CamemBERT/FlauBERT/BERT sur le découpage en sous-mots.
- Sortie console (comparatif), pas de fichier final dédié.

## Comparatifs

```bash
cd model/ALEXANDRE/tokenization
python3 compare_all_tokenizers.py
```

Notebook d'analyse:

```bash
jupyter notebook model/ALEXANDRE/compare_tokenizers_confusion_matrices.ipynb
```
