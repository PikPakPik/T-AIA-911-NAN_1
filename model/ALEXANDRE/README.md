# ALEXANDRE - NER et Pathfinding

Ce dossier contient le coeur du pipeline utilisé dans l'app:
- entraînement et tests NER (RoBERTa/BERT);
- pathfinding SNCF (graphe GTFS + Dijkstra);
- tokenisation et comparatifs;
- notebooks d'évaluation/expérimentation.

## Sous-dossiers

- `training/`: scripts d'entraînement et de test NER
- `pathfinding/`: moteur SNCF + app Streamlit chat/audio
- `tokenization/`: comparatifs de tokenizers et analyses

## Démarrage rapide

Depuis la racine du repo:

```bash
python3 T-AIA-911.py --ui --model Jean-Baptiste/camembert-ner
```

Ou directement:

```bash
streamlit run model/ALEXANDRE/pathfinding/chat_pathfinding_app.py
```

## Docs détaillées

- `model/ALEXANDRE/training/README.md`
- `model/ALEXANDRE/pathfinding/README.md`
- `model/ALEXANDRE/tokenization/README.md`
