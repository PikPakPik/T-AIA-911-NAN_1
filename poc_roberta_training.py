import re
import csv
import random
import json
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import Dataset
import torch
from torch import nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import unicodedata

# Charge les données de géolocalisation
with open("./dataset/geo.csv", "r", encoding="utf-8") as file:
    reader = csv.reader(file)
    locations = list(reader)

def remove_accents(text: str) -> str:
    """Remove accents from text."""
    nfkd = unicodedata.normalize('NFD', text)
    return ''.join([c for c in nfkd if not unicodedata.combining(c)])

def generate_variants(name: str) -> set[str]:
    variants = set()
    base = name.strip()

    saint_rules = [
        (r"\bSaint\b", ["Saint", "St"]),
        (r"\bSainte\b", ["Sainte", "Ste"]),
    ]

    def apply_rules(text):
        results = {text}
        for pattern, repls in saint_rules:
            new = set()
            for r in results:
                if re.search(pattern, r, flags=re.IGNORECASE):
                    for repl in repls:
                        new.add(re.sub(pattern, repl, r, flags=re.IGNORECASE))
                else:
                    new.add(r)
            results = new
        return results

    for v in apply_rules(base):
        variants.add(v)
        variants.add(v.replace("-", " "))
        variants.add(v.replace(" ", "-"))
        variants.add(v.replace("'", " "))
        variants.add(v.replace("'", ""))

        no_accent = remove_accents(v)
        variants.add(no_accent)
        variants.add(no_accent.replace("-", " "))
        variants.add(no_accent.replace(" ", "-"))
        variants.add(no_accent.replace("'", " "))
        variants.add(no_accent.replace("'", ""))

    final = set()
    for v in variants:
        final.add(v)
        final.add(v.lower())
        final.add(v.upper())
        final.add(v.capitalize())

    return final

TEMPLATES = [
    "Je veux aller de {a} à {b}",
    "Train entre {a} et {b}",
    "Voyage de {a} vers {b}",
    "Billet {a} {b}",
    "Comment aller de {a} à {b}?",
    "Quel train entre {a} et {b}?",
    "C'est combien pour aller de {a} à {b}?",
    "Je dois me rendre de {a} à {b}",
    "Je dois aller de {a} à {b} demain",
    "Un aller-retour {a}-{b} s'il vous plaît",
    "Un billet {a} {b} svp",
    "Départ de {a} vers {b} ce soir",
    "Je pars de {a} pour {b} lundi prochain",
    "Train de {a} à {b} demain matin",
    "Départ {a} arrivée {b} cet après-midi",
    "Trajet depuis {a} vers {b}",
    "Réserver un billet de {a} pour {b}",
    "Je souhaite me rendre à {b} depuis {a}",
    "Je veux aller à {b} en partant de {a}",
    "Aller à {b} au départ de {a}",
    "Direction {b} depuis {a}",
    "J'ai besoin d'aller de {a} à {b}",
    "Faut que j'aille de {a} à {b}",
    "Cherche train {a} {b}",
    "Dispo {a}-{b}?",
    "Réservation au départ de {a} direction {b}",
    "Billet de train de {a} à destination de {b}",
    "Partir de {a} pour rejoindre {b}",
    "Trajet en train de {a} jusqu'à {b}",
]

def generate_annotated_sentences(locations, n=15000):
    """Génère des phrases annotées pour l'entraînement NER"""
    samples = []

    for _ in range(n):
        (name1, label1), (name2, label2) = random.sample(locations, 2)

        v1 = random.choice(list(generate_variants(name1)))
        v2 = random.choice(list(generate_variants(name2)))

        sentence = random.choice(TEMPLATES).format(a=v1, b=v2)

        start1 = sentence.index(v1)
        end1 = start1 + len(v1)
        start2 = sentence.index(v2)
        end2 = start2 + len(v2)

        samples.append((
            sentence,
            [
                (start1, end1, "LOC"),
                (start2, end2, "LOC"),
            ]
        ))

    return samples

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
    # Configuration
    model_name = "benjamin/roberta-base-wechsel-french"
    output_dir = "./roberta_ner_model"
    num_samples = 15000
    
    print(f"Génération de {num_samples} échantillons d'entraînement...")
    samples = generate_annotated_sentences(locations, n=num_samples)
    random.shuffle(samples)
    
    # Division train/dev/test
    split_train = int(len(samples) * 0.8)
    split_dev = int(len(samples) * 0.9)
    
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
    
    # Trainer
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
    random.seed(42)
    main()
