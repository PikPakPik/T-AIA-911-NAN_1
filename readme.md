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
L’entraînement utilise **uniquement** les fichiers **dataset_input.txt** et **dataset_output.txt** (aucune phrase n’est définie dans le script d’entraînement).

1. Générer le dataset (~100k phrases : valides, invalides, villes en contexte non-trajet, pièges, recette) :
```bash
cd dataset
python3 generate_dataset.py
cd ..
```
(Peut prendre quelques minutes.)

2. Lancer l’entraînement :
```bash
python3 poc_roberta_training.py
```

Options : `--dataset-dir ./dataset`, `--output-dir ./roberta_ner_model`, `--seed 42`.

Le modèle fine-tuné est enregistré dans `./roberta_ner_model/`.

**GPU Apple Silicon (M1/M2/M3)** : MPS est utilisé automatiquement par le Trainer et les pipelines (`poc_roberta.py`, `chat_roberta_loc.py`) si PyTorch est installé avec support MPS. PyTorch ≥ 1.13 recommandé.

### Générer le jeu de test (formulations différentes)
Pour évaluer le modèle sur des phrases **jamais vues** à l’entraînement :

```bash
cd dataset
python3 generate_test_dataset.py
cd ..
```

Cela crée **dataset_test_input.txt** et **dataset_test_output.txt** (formulations et seed différents de `generate_dataset.py`). Ensuite, évaluer le modèle sur ce fichier :

```bash
python3 poc_roberta.py --file dataset/dataset_test_input.txt --output dataset_test_output_roberta.csv
```
Puis comparer avec `dataset/dataset_test_output.txt` (ex. dans le notebook d’évaluation en pointant vers ces fichiers).

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

---

## Pathfinding SNCF (`pathfinding_sncf.ipynb`)

### But
Trouver le **trajet le plus rapide** entre deux villes françaises en utilisant le **réseau ferroviaire réel** de la SNCF. Le notebook permet aussi de comparer les trajets en **TGV**, **TER**, ou **tous trains confondus**.

### Source des données : GTFS

On utilise les données **GTFS** (General Transit Feed Specification) de la SNCF. C'est le format standard mondial pour décrire les réseaux de transport en commun. Les fichiers sont dans `dataset/sncf/` :

| Fichier | Ce qu'il contient | Exemple |
|---------|-------------------|---------|
| `stops.txt` | Les gares avec leurs coordonnées GPS | "Lyon Part Dieu", lat=45.76, lon=4.86 |
| `routes.txt` | Les lignes ferroviaires | "Paris - Lyon TGV" |
| `trips.txt` | Les trains individuels sur chaque ligne | "TGV 6201 du 15 mars" |
| `stop_times.txt` | Les horaires d'arrêt de chaque train à chaque gare | "TGV 6201 arrive à Lyon à 10:00" |

**Relation entre les fichiers :**
```
routes --> trips --> stop_times --> stops
(lignes)  (trains)  (horaires)    (gares)
```

### Comment le graphe est construit

#### 1. Les nœuds = les gares

Dans `stops.txt`, il y a deux types d'entrées :
- **StopArea** (location_type=1) : la gare elle-même (ex: "Lyon Part Dieu") — **3 498 gares**
- **StopPoint** (location_type=0) : un quai spécifique dans la gare (ex: "voie 3 de Lyon Part Dieu") — **5 401 quais**

On travaille au niveau **StopArea** (la gare), pas au niveau des quais. Chaque StopPoint est rattaché à un StopArea parent.

#### 2. Les arêtes = les liaisons directes entre gares

On parcourt `stop_times.txt` qui contient tous les horaires de tous les trains. Pour chaque train, les arrêts sont numérotés dans l'ordre (stop_sequence). Deux arrêts **consécutifs** d'un même train = une liaison directe.

**Exemple** : le TGV 6201 fait Paris (08:00) → Mâcon (09:30) → Lyon (10:00)
- Ça crée l'arête Paris → Mâcon (90 min)
- Et l'arête Mâcon → Lyon (30 min)

#### 3. Le poids des arêtes = le temps de trajet minimum

Pour chaque paire de gares (A, B), il peut y avoir des centaines de trains différents avec des temps différents. On garde le **temps de trajet minimum** parmi tous les trains.

Par exemple, entre Paris et Lyon :
- TGV : 1h46
- TER : pourrait être 4h+

Le poids de l'arête Paris → Lyon dans le graphe global sera **1h46** (le min).

#### 4. Le type de train

Le type de train (TGV INOUI, TER, Intercités, etc.) est extrait depuis le `stop_id` :
- `StopPoint:OCETGV INOUI-87686006` → **TGV INOUI**
- `StopPoint:OCETrain TER-87342014` → **TER**
- `StopPoint:OCENavette-87571000` → **Navette**

On construit un **graphe par type de train** pour pouvoir comparer.

#### Résultat

| | Valeur |
|---|---|
| Nombre de gares (nœuds) | ~3 500 |
| Nombre de liaisons directes (arêtes) | ~10 800 |
| Gare la plus connectée | Lyon Part Dieu (42 connexions) |

### Algorithme de Dijkstra

**Dijkstra** est un algorithme classique pour trouver le **plus court chemin** dans un graphe pondéré (ici, pondéré par le temps de trajet).

#### Principe simplifié :

1. On part de la gare de départ avec un temps = 0
2. On regarde toutes les gares voisines et on note le temps pour y aller
3. On choisit la gare non-visitée la plus proche (la plus rapide à atteindre)
4. Depuis cette gare, on regarde ses voisins et on met à jour les temps si on trouve mieux
5. On répète jusqu'à arriver à la destination

C'est comme une vague qui se propage depuis le départ : elle atteint d'abord les gares proches, puis les gares de plus en plus lointaines, toujours en prenant le chemin le plus rapide.

#### Implémentation technique :

- **Min-heap (file de priorité)** : c'est une structure de données qui garde toujours l'élément le plus petit en haut de la pile. Ici, on y met les gares avec leur temps d'accès. À chaque étape, on retire la gare la plus rapide à atteindre (celle en haut du tas) au lieu de parcourir toute la liste — c'est ce qui rend Dijkstra efficace.

- **Complexité : O((V + E) log V)** — ça décrit la "vitesse" de l'algorithme en fonction de la taille du graphe :
  - **V** = nombre de nœuds (gares) = ~3 500
  - **E** = nombre d'arêtes (liaisons) = ~10 800
  - **log V** = le coût d'insertion/extraction dans le min-heap (~12 opérations pour 3 500 gares)
  - En gros : on visite chaque gare et chaque liaison une fois, et à chaque fois on fait ~12 opérations pour maintenir le tas trié. C'est très rapide comparé à une approche naïve en O(V²) qui reviendrait à comparer toutes les gares entre elles à chaque étape.

- En pratique : le calcul prend **50-100 ms** pour n'importe quel trajet en France

### Recherche ville → gare

L'utilisateur tape un nom de ville (ex: "Lyon"). Le système doit trouver la bonne gare dans le graphe.

**Stratégie de matching :**
1. Correspondance exacte → score 1.0
2. Le nom de la gare commence par la ville → score 0.93 (ex: "Lyon" → "Lyon Part Dieu")
3. Le nom contient la ville → score 0.70
4. Similarité approximative (fuzzy matching) → score variable

En cas d'égalité, on préfère les gares **les plus connectées** (les gares principales). Ainsi "Lyon" → "Lyon Part Dieu" (42 connexions) plutôt que "Lyon-Vaise" (4 connexions).

### Distances et vitesse

- La **distance** entre deux gares est calculée à vol d'oiseau avec la **formule de Haversine** (distance sur une sphère à partir des coordonnées GPS)
- La **vitesse moyenne** = distance totale / temps total
- Note : la distance réelle par les rails est plus longue (courbes, détours), donc la vitesse affichée est une estimation basse

### Comparaison TGV vs TER

Le notebook permet de comparer le même trajet avec différents types de trains :

| | TGV INOUI | TER |
|---|---|---|
| **Réseau** | Grandes lignes, liaisons directes longue distance | Réseau régional, dessert toutes les petites gares |
| **Paris → Lyon** | ~1h46, direct | Beaucoup plus long, nombreux arrêts |
| **Avantage** | Rapide | Dessert les petites villes |

**Pourquoi le TGV "saute" des gares :** Le TGV crée des arêtes directes longue distance (ex: Angers → Saint-Pierre-des-Corps en 22 min) alors que le TER s'arrête partout (Angers → Saumur → Saint-Pierre-des-Corps en ~55 min). Dijkstra choisit naturellement le chemin le plus rapide.

### Limites actuelles

1. **Pas de temps d'arrêt en gare** : le temps total ne compte pas les minutes passées à l'arrêt dans les gares intermédiaires ni le temps de correspondance
2. **Distance à vol d'oiseau** : pas la distance réelle par les rails
3. **Mélange de trains** : le trajet optimal peut combiner des segments de trains différents (un TGV + un TER) sans vérifier que la correspondance est possible dans les horaires réels
4. **Temps minimum** : on prend le meilleur temps observé par liaison, qui ne correspond pas forcément à un train circulant tous les jours

### Lancer le notebook

```bash
jupyter notebook pathfinding_sncf.ipynb
```

Exécuter toutes les cellules dans l'ordre. Les tests de trajet s'affichent automatiquement à la fin.
