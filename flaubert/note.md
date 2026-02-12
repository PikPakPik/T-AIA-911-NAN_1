# Résumé Flaubert 

Voici les 3 grandes étapes du projet :


### 2. Entraînement d'un Modèle de Classification (`train.py`)

Le but de cette étape est d'apprendre à un modèle à reconnaître si une phrase est une demande de voyage pertinente ou non.

*   **Quoi ?** On a entraîné un modèle **Flaubert** (`flaubert/flaubert_base_cased`), un modèle de langue spécialisé pour le français.
*   **Tâche :** Il s'agit d'une **classification binaire**. Le modèle ne fait qu'une seule chose : prédire si une phrase est `VALID` (1) ou `INVALID` (0). Il n'extrait pas les villes lui-même.
*   **Comment ?**
    *   On charge les données de tests.
    *   On utilise une boucle d'entraînement standard avec PyTorch et la bibliothèque `transformers`.
    *   Le modèle apprend sur 80% des données et est validé sur 20%.
*   **Résultat :** Le meilleur modèle est sauvegardé dans `best_model.pt`.

### 3. Extraction des Villes (Inférence) (`inference.py`)

C'est le cœur du système, celui qui donne le résultat final. L'approche est **basée sur des règles** et non sur le modèle Flaubert directement.

*   **Quoi ?** Extraire les villes de départ et d'arrivée d'une phrase donnée.
*   **Comment ?** La classe `TravelOrderExtractor` procède en plusieurs étapes :
    1.  **Pré-filtrage (`is_valid_travel_query`) :** Des règles simples éliminent les requêtes clairement non pertinentes (contenant "hôtel", "restaurant", etc.).
    2.  **Extraction par Règles (`extract_locations_rule_based`) :** C'est ici que la magie opère. Des **expressions régulières (regex)** recherchent des mots-clés ("de", "vers", "à", "depuis") et capturent les mots qui suivent.
    3.  **Validation par Dictionnaire :** Les villes candidates trouvées par les regex sont comparées à une liste de villes connues (chargée depuis `geo.csv`). Si un mot n'est pas dans le dictionnaire, il n'est pas considéré comme une ville. Cela évite de confondre "Albert" (prénom) avec la ville d'Albert.
*   **Point crucial pour la présentation :** Le script `inference.py` charge bien le modèle Flaubert (`best_model.pt`), mais **il ne l'utilise pas pour faire la prédiction finale**. L'extraction est 100% basée sur les règles et le dictionnaire. Le modèle est là, mais sa prédiction n'est pas exploitée dans la logique d'extraction.

### 4. Évaluation des Performances (`evaluate.py`)

Cette étape sert à mesurer objectivement la qualité de notre système d'extraction.

*   **Quoi ?** Évaluer la performance de `TravelOrderExtractor` sur le jeu de test.
*   **Comment ?**
    *   Le script compare les prédictions de notre extracteur (basé sur les règles) avec les résultats attendus du fichier `dataset_output.txt`.
    *   Il calcule plusieurs métriques :
        *   **Accuracy** : Pourcentage de prédictions parfaites (départ ET arrivée corrects).
        *   **Précision, Rappel, F1-Score** : Pour évaluer la capacité à classifier correctement les requêtes en `VALID` ou `INVALID`.
*   **Résultat :** Le script génère un **rapport textuel complet** (`evaluation_results.txt`) avec toutes les métriques et des exemples d'erreurs, ainsi qu'une **matrice de confusion** visuelle (`confusion_matrix.png`).

---

**En résumé pour l'oral :**

Notre projet "Flaubert" vise à extraire les villes de départ et d'arrivée. Pour cela, nous avons :
1.  **Créé notre propre dataset** avec des cas simples, complexes et des pièges.
2.  **Entraîné un modèle Flaubert** pour la classification binaire (requête valide/invalide).
3.  Développé un **extracteur basé sur des règles (regex) et un dictionnaire de villes**, qui est la méthode actuellement utilisée pour la prédiction. Le modèle Flaubert est chargé mais pas encore intégré dans la décision finale.
4.  Mis en place un **script d'évaluation robuste** pour mesurer la performance de notre extracteur avec des métriques claires et des rapports détaillés.

C'est un système hybride où l'IA (Flaubert) est pour l'instant utilisée pour une tâche de classification, tandis que l'extraction finale reste déterministe grâce à des règles précises.
