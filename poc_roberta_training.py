import re
import random
import json
import argparse
import os
import csv
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import Dataset
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import unicodedata


def remove_accents(text: str) -> str:
    """Remove accents from text (used for matching in load_dataset_input_output)."""
    nfkd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def align_labels_with_char_positions(text, entities, offset_mapping):
    """Aligne les labels avec les tokens en utilisant les positions de caractères.

    `entities` est une liste de tuples (start, end) en indices de caractères dans `text`.
    Toutes les entités sont de type LOC, donc on génère des labels B-LOC / I-LOC.
    """
    # Crée un mapping caractère -> label
    char_labels = ["O"] * len(text)

    # Marque les entités dans le texte
    for start, end in entities:
        if start < len(text) and end <= len(text):
            char_labels[start] = "B-LOC"
            for i in range(start + 1, end):
                if i < len(char_labels):
                    char_labels[i] = "I-LOC"
    
    # Aligne avec les tokens via offset_mapping
    aligned_labels = []
    for start_char, end_char in offset_mapping:
        if start_char is None or end_char is None:
            # Token spécial ([CLS], [SEP], padding)
            aligned_labels.append(-100)
        else:
            # Prend le label du premier caractère du token
            label = char_labels[start_char] if start_char < len(char_labels) else "O"
            # Convertit en ID
            label_to_id = {"O": 0, "B-LOC": 1, "I-LOC": 2}
            aligned_labels.append(label_to_id.get(label, 0))
    
    return aligned_labels

def tokenize_and_align_labels(examples, tokenizer):
    """Tokenise et aligne les labels avec les tokens"""
    # Utilise le texte original (pas celui reconstruit depuis tokens)
    texts = examples["text"]
    entities_list = examples["entities"]
    
    # Tokenise le texte complet
    tokenized_inputs = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_offsets_mapping=True
    )
    
    all_labels = []
    offset_mappings = tokenized_inputs.pop("offset_mapping")
    
    for i, (text, entities) in enumerate(zip(texts, entities_list)):
        aligned_labels = align_labels_with_char_positions(
            text, entities, offset_mappings[i]
        )
        all_labels.append(aligned_labels)
    
    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs

def create_iob_labels(text, entities):
    """Crée des labels IOB (Inside-Outside-Beginning) pour une phrase"""
    # Initialise tous les tokens comme "O" (Outside)
    tokens = text.split()
    labels = ["O"] * len(tokens)
    
    # Trie les entités par position de début
    sorted_entities = sorted(entities, key=lambda x: x[0])
    
    # Crée un mapping position -> token index
    char_to_token = {}
    char_pos = 0
    for token_idx, token in enumerate(tokens):
        for _ in range(len(token)):
            char_to_token[char_pos] = token_idx
            char_pos += 1
        char_pos += 1  # Espace
    
    # Marque les entités
    for start, end, label in sorted_entities:
        # Trouve les tokens correspondants
        start_token = char_to_token.get(start, None)
        end_token = char_to_token.get(end - 1, None)
        
        if start_token is not None and end_token is not None:
            # Marque le premier token comme B-LOC (Beginning)
            labels[start_token] = "B-LOC"
            # Marque les tokens suivants comme I-LOC (Inside)
            for i in range(start_token + 1, end_token + 1):
                if i < len(labels):
                    labels[i] = "I-LOC"
    
    return tokens, labels

def _normalize_for_match(s: str) -> str:
    """Normalise pour recherche insensible à la casse et aux accents."""
    s = remove_accents(s).lower().replace("-", " ")
    return s

def _find_all_occurrences(phrase: str, name: str) -> list[tuple[int, int]]:
    """Trouve toutes les occurrences de name dans phrase (insensible casse/accents/tirets)."""
    if not name or not name.strip():
        return []
    phrase_norm = _normalize_for_match(phrase)
    name_norm = _normalize_for_match(name)
    if not name_norm:
        return []
    # Longueur dans la phrase normalisée (espaces = 1 char) pour garder les indices
    result = []
    pos = 0
    while True:
        i = phrase_norm.find(name_norm, pos)
        if i == -1:
            break
        # Span dans la phrase originale (même indices car replace préserve la longueur)
        result.append((i, i + len(name_norm)))
        pos = i + 1
    return result

def _load_known_cities(geo_csv_path: str = "./dataset/geo.csv") -> set[str]:
    """Charge la liste des villes connues depuis geo.csv (colonne 'nom')."""
    cities = set()
    try:
        with open(geo_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                city_name = row.get("nom", "").strip()
                if city_name:
                    cities.add(city_name)
    except FileNotFoundError:
        print(f"⚠️  Fichier {geo_csv_path} introuvable. Aucune ville chargée pour INVALID.")
    return cities

def _find_cities_in_phrase(phrase: str, known_cities: set[str]) -> list[tuple[int, int, str]]:
    """
    Détecte automatiquement les villes présentes dans une phrase.
    Retourne une liste d'entités (start, end, "LOC").
    """
    entities = []
    # Trie les villes par longueur décroissante pour détecter les plus longues d'abord
    # (évite que "Paris" match dans "Paris-Brest" si "Paris-Brest" est une ville)
    sorted_cities = sorted(known_cities, key=len, reverse=True)
    
    # Pour éviter les doublons, on garde trace des zones déjà couvertes
    covered_ranges = []
    
    for city in sorted_cities:
        occurrences = _find_all_occurrences(phrase, city)
        for start, end in occurrences:
            # Vérifie qu'on ne chevauche pas une zone déjà couverte
            overlaps = any(
                (start < c_end and end > c_start)
                for c_start, c_end in covered_ranges
            )
            if not overlaps:
                entities.append((start, end, "LOC"))
                covered_ranges.append((start, end))
    
    # Trie par position pour cohérence
    entities.sort(key=lambda x: x[0])
    return entities

def load_dataset_input_output(
    input_path: str = "./dataset/dataset_input.txt",
    output_path: str = "./dataset/dataset_output.txt",
    geo_csv_path: str = "./dataset/geo.csv",
) -> list[tuple[str, list[tuple[int, int, str]]]]:
    """
    Charge dataset_input.txt et dataset_output.txt et les convertit en samples NER.
    Chaque sample est (phrase, [(start, end, "LOC"), ...]).
    
    IMPORTANT: Pour les phrases INVALID, on détecte automatiquement les villes présentes
    dans la phrase (depuis geo.csv) et on les annote comme LOC. Cela permet au modèle
    d'apprendre à reconnaître les noms de villes même dans des contextes non-voyage.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        input_lines = [line.rstrip("\n") for line in f]
    with open(output_path, "r", encoding="utf-8") as f:
        output_lines = [line.rstrip("\n") for line in f]

    # Charge la liste des villes connues pour détecter les LOC dans les phrases INVALID
    print(f"  Chargement des villes depuis {geo_csv_path}...")
    known_cities = _load_known_cities(geo_csv_path)
    print(f"  → {len(known_cities)} villes chargées")

    samples = []
    for inp, out in zip(input_lines, output_lines):
        # Input: "id,phrase" (la phrase peut contenir des virgules)
        parts_in = inp.split(",", 1)
        if len(parts_in) < 2:
            continue
        phrase = parts_in[1].strip()

        # Output: "id,dep,dest" ou "id,INVALID,"
        parts_out = out.split(",")
        if len(parts_out) < 2:
            continue
        dep = parts_out[1].strip() if len(parts_out) > 1 else ""
        dest = parts_out[2].strip() if len(parts_out) > 2 else ""

        if dep == "INVALID" or not dep:
            # Détecte automatiquement les villes dans la phrase
            entities = _find_cities_in_phrase(phrase, known_cities)
            samples.append((phrase, entities))
            continue

        occ_dep = _find_all_occurrences(phrase, dep)
        occ_dest = _find_all_occurrences(phrase, dest)

        entities = []
        if dep == dest:
            # Même ville (ex: Paris → Paris) : prendre les 2 premières occurrences
            all_occ = occ_dep[:2]
            for (start, end) in all_occ:
                entities.append((start, end, "LOC"))
        else:
            # Deux villes différentes : première occurrence de chaque
            if occ_dep:
                start, end = occ_dep[0]
                entities.append((start, end, "LOC"))
            if occ_dest:
                start, end = occ_dest[0]
                entities.append((start, end, "LOC"))

        entities.sort(key=lambda x: x[0])
        samples.append((phrase, entities))

    return samples

def prepare_dataset(samples, tokenizer):
    """Prépare le dataset au format Hugging Face"""
    texts = []
    all_tokens = []
    all_entities_list = []
    
    label_to_id = {"O": 0, "B-LOC": 1, "I-LOC": 2}
    
    for text, entities in samples:
        # On garde le texte original et on ne conserve pour les entités
        # que les positions (start, end), le label étant toujours LOC.
        tokens, _ = create_iob_labels(text, entities)
        texts.append(text)  # Garde le texte original
        all_tokens.append(tokens)
        all_entities_list.append([(start, end) for (start, end, _label) in entities])
    
    # Crée le dataset avec les entités et le texte original
    dataset_with_entities = Dataset.from_dict(
        {
            "tokens": all_tokens,
            "text": texts,  # Texte original pour les positions de caractères
            "entities": all_entities_list,
        }
    )
    
    # Tokenise et aligne les labels
    tokenized_dataset = dataset_with_entities.map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True,
        remove_columns=dataset_with_entities.column_names
    )
    
    return tokenized_dataset, label_to_id

def compute_metrics(eval_pred):
    """Calcule les métriques d'évaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    # Enlève les labels -100 (ignorés)
    true_predictions = [
        [p for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [l for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Aplatit les listes
    true_predictions = [item for sublist in true_predictions for item in sublist]
    true_labels = [item for sublist in true_labels for item in sublist]

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, true_predictions, average='weighted', zero_division=0
    )
    accuracy = accuracy_score(true_labels, true_predictions)

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

def main():
    parser = argparse.ArgumentParser(
        description="Entraînement NER RoBERTa pour entités LOC (données = dataset_input.txt + dataset_output.txt)"
    )
    parser.add_argument("--dataset-dir", default="./dataset", help="Dossier contenant dataset_input.txt et dataset_output.txt")
    parser.add_argument("--output-dir", default="./roberta_ner_model", help="Dossier de sortie du modèle")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)

    model_name = "benjamin/roberta-base-wechsel-french"
    output_dir = args.output_dir

    # Chargement du dataset (dataset_input.txt + dataset_output.txt)
    input_path = os.path.join(args.dataset_dir, "dataset_input.txt")
    output_path = os.path.join(args.dataset_dir, "dataset_output.txt")
    geo_csv_path = os.path.join(args.dataset_dir, "geo.csv")
    if not os.path.isfile(input_path) or not os.path.isfile(output_path):
        raise FileNotFoundError(
            f"Fichiers dataset introuvables: {input_path}, {output_path}. "
            "Générez-les avec: cd dataset && python3 generate_dataset.py"
        )
    print(f"Chargement du dataset: {input_path} / {output_path}...")
    samples = load_dataset_input_output(input_path, output_path, geo_csv_path)
    print(f"  → {len(samples)} phrases chargées")


    # Division train / dev / test (80 / 10 / 10)
    n = len(samples)
    split_train = int(n * 0.8)
    split_dev = int(n * 0.9)
    train_samples = samples[:split_train]
    dev_samples = samples[split_train:split_dev]
    test_samples = samples[split_dev:]

    print(f"Train: {len(train_samples)}, Dev: {len(dev_samples)}, Test: {len(test_samples)}")
    
    # Charge le tokenizer
    print(f"Chargement du tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Prépare les datasets
    print("Préparation des datasets...")
    train_dataset, label_to_id = prepare_dataset(train_samples, tokenizer)
    dev_dataset, _ = prepare_dataset(dev_samples, tokenizer)
    test_dataset, _ = prepare_dataset(test_samples, tokenizer)
    
    # Charge le modèle
    print(f"Chargement du modèle: {model_name}...")
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label_to_id),
        id2label={v: k for k, v in label_to_id.items()},
        label2id=label_to_id
    )
    
    # Arguments d'entraînement
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        logging_dir=f"{output_dir}/logs",
    )
    
    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    # Affiche le device (MPS sur Mac ARM64, CUDA, ou CPU)
    _d = training_args.device
    if _d.type == "mps":
        print("Device: MPS (Apple Silicon GPU)")
    elif _d.type == "cuda":
        print("Device: CUDA")
    else:
        print("Device: CPU")

    # Trainer (MPS sur Mac ARM64 utilisé automatiquement si disponible)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Entraînement
    print("Démarrage de l'entraînement...")
    trainer.train()
    
    # Évaluation sur le test set
    print("Évaluation sur le test set...")
    test_results = trainer.evaluate(test_dataset)
    print(f"Résultats sur le test set: {test_results}")
    
    # Sauvegarde du modèle
    print(f"Sauvegarde du modèle dans {output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Sauvegarde du mapping des labels
    with open(f"{output_dir}/label_mapping.json", "w") as f:
        json.dump(label_to_id, f, indent=2)
    
    print("Entraînement terminé!")

if __name__ == "__main__":
    main()
