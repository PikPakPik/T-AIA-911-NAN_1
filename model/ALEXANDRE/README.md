# Tokenizers pour la détection Départ / Arrivée (NER)

Ce dossier contient **4 approches** de tokenisation pour annoter des phrases (format BIO : Départ, Arrivée). Les données viennent de `dataset_input.txt` et `dataset_output.txt`.

---

## Script 1 : NLTK (`script1_nltk_basic.py`)

- **Méthode** : `word_tokenize` de NLTK (langue française).
- **Comportement** : Sépare les contractions ("j'" → "j'", "aimerais"). Garde souvent les noms composés avec tirets en un seul token (ex. `nogent-l'artaud`).
- **Atouts** : Simple, bon taux de détection (~82–84 % sur le dataset).
- **Limites** : Dépend du découpage NLTK ; certains noms ou formulations peuvent être mal reconnus.
- **Lancer** : `python script1_nltk_basic.py`
- **Sortie** : `script1_nltk_output.json`

---

## Script 2 : Regex v1 (`script2_regexp_basic.py` / `script2_regex_v1.py`)

- **Méthode** : Expression régulière  
  `r"\b(?:d'|l'|j'|qu'|n'|s'|t'|m'|c')?\w+\b"`  
  (contractions optionnelles + mot).
- **Comportement** : Les contractions restent **collées** au mot (ex. `d'Alençon` = 1 token). Les tirets **coupent** les mots (ex. `Ailly-sur-Somme` → `ailly`, `sur`, `somme`).
- **Atouts** : Bonne détection sur beaucoup de phrases (~92–93 %).
- **Limites** : Si la ville est écrite avec une contraction (ex. "d'Alençon"), le départ n’est pas trouvé car "Alençon" n’est pas un token seul.
- **Lancer** : `python script2_regexp_basic.py` ou `python script2_regex_v1.py`
- **Sortie** : `script2_output.json` / `script2_regex_v1_output.json`

---

## Script 3 : Regex v2 améliorée (`script3_regexp_improved.py`)

- **Méthode** : Expression régulière  
  `r"\b(?:d'|l'|j'|qu'|n'|s'|t'|m'|c')\b|\b\w+\b"`  
  (contractions **séparées** OU mot).
- **Comportement** : Les contractions sont des tokens à part (ex. `d'` + `Alençon`). Les mots avec tirets sont aussi découpés.
- **Atouts** : Meilleure gestion des contractions ; très bon taux de détection (~92–93 %) et plus robuste que la v1 sur des formulations comme "d'Alençon".
- **Limites** : Plus de tokens par phrase (découpage plus fin).
- **Lancer** : `python script3_regexp_improved.py`
- **Sortie** : `script3_output.json`

---

## Script 4 : Tokenizers pré-entraînés (`script4_pretrained_tokenizers.py`)

- **Méthode** : Tokenizers Hugging Face (CamemBERT, FlauBERT, BERT français, BERT multilingue).
- **Comportement** : Découpage en **sous-mots** (ex. "Alençon" → "Al", "en", "çon"). Indicateurs : `▁` (CamemBERT), `</w>` (FlauBERT), `##` (BERT).
- **Atouts** : Adaptés au français ou multilingues ; utilisables pour fine-tuner un modèle de NER.
- **Limites** : Ne font **pas** la détection Départ/Arrivée tout seuls ; il faut entraîner un modèle (ex. avec `poc_roberta_training.py`) sur des données annotées.
- **Lancer** : `python script4_pretrained_tokenizers.py`
- **Sortie** : Affichage console uniquement (comparaison des segmentations).

---

## En résumé

| Script | Type        | Contractions      | Noms composés (tirets) | Détection DEP/ARR      |
|--------|-------------|-------------------|------------------------|-------------------------|
| 1      | NLTK        | Séparées          | Souvent 1 token        | ~82–84 %               |
| 2      | Regex v1    | Collées au mot    | Découpés               | ~92–93 % (échec si "d'Alençon") |
| 3      | Regex v2    | Séparées          | Découpés               | ~92–93 %, plus robuste |
| 4      | Hugging Face| Sous-mots         | Sous-mots              | Nécessite entraînement |

Pour **comparer** les 3 premières approches sur des exemples et des stats :  
`python compare_all_tokenizers.py`

Pour **matrices de confusion** et métriques détaillées : ouvrir le notebook  
`compare_tokenizers_confusion_matrices.ipynb`.
