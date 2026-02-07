import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    CamembertForTokenClassification,
    CamembertTokenizerFast,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import Dataset as HFDataset
import spacy
from spacy.tokens import DocBin
import numpy as np
from sklearn.metrics import classification_report
import argparse
import os

# Étiquettes pour les entités
LABEL_LIST = ["O", "B-LOC_CITY", "I-LOC_CITY", "B-LOC_DEPARTMENT", "I-LOC_DEPARTMENT", 
              "B-LOC_REGION", "I-LOC_REGION", "B-LOC_GARE", "I-LOC_GARE", 
              "B-LOC_AEROPORT", "I-LOC_AEROPORT"]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}

def load_spacy_data(spacy_file):
    """Charge les données depuis un fichier .spacy"""
    nlp = spacy.blank("fr")
    doc_bin = DocBin().from_disk(spacy_file)
    docs = list(doc_bin.get_docs(nlp.vocab))
    
    examples = []
    for doc in docs:
        tokens = [token.text for token in doc]
        # Initialiser tous les tokens comme "O" (Outside)
        ner_tags = ["O"] * len(tokens)
        
        # Marquer les entités avec le schéma BIO
        for ent in doc.ents:
            start_token = None
            end_token = None
            
            # Trouver les indices de tokens correspondant à l'entité
            for i, token in enumerate(doc):
                if token.idx >= ent.start_char and start_token is None:
                    start_token = i
                if token.idx + len(token.text) <= ent.end_char:
                    end_token = i
            
            if start_token is not None and end_token is not None:
                # Premier token : B-
                ner_tags[start_token] = f"B-{ent.label_}"
                # Tokens suivants : I-
                for i in range(start_token + 1, end_token + 1):
                    ner_tags[i] = f"I-{ent.label_}"
        
        examples.append({
            "tokens": tokens,
            "ner_tags": ner_tags
        })
    
    return examples

def tokenize_and_align_labels(examples, tokenizer):
    """Tokenize et aligne les labels avec les sous-tokens"""
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding=False,
        max_length=512
    )
    
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                # Tokens spéciaux (CLS, SEP, PAD)
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # Premier sous-token d'un mot
                label_ids.append(LABEL2ID[label[word_idx]])
            else:
                # Sous-tokens suivants : on met -100 ou on propage le I-
                current_label = label[word_idx]
                if current_label.startswith("B-"):
                    # Convertir B- en I- pour les sous-tokens
                    label_ids.append(LABEL2ID[current_label.replace("B-", "I-")])
                else:
                    label_ids.append(LABEL2ID[current_label])
            
            previous_word_idx = word_idx
        
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def compute_metrics(pred):
    """Calcule les métriques d'évaluation"""
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=2)
    
    # Retirer les labels ignorés (-100)
    true_predictions = [
        [LABEL_LIST[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [LABEL_LIST[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    # Calculer les métriques
    from seqeval.metrics import classification_report as seq_classification_report
    from seqeval.metrics import f1_score, precision_score, recall_score
    
    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }

def main():
    parser = argparse.ArgumentParser(description="Train CamemBERT for French location NER")
    parser.add_argument("--train", default="../train.spacy", help="Path to training .spacy file")
    parser.add_argument("--dev", default="../dev.spacy", help="Path to dev .spacy file")
    parser.add_argument("--output", default="./camembert_output", help="Output directory")
    parser.add_argument("--epochs", type=int, default=8, help="Number of epochs (augmenté à 8)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate (réduit pour plus de stabilité)")
    parser.add_argument("--model", default="camembert-base", help="Base model")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    
    args = parser.parse_args()
    
    # Charger les données
    print("Loading training data...")
    train_examples = load_spacy_data(args.train)
    print(f"Loaded {len(train_examples)} training examples")
    
    print("Loading dev data...")
    dev_examples = load_spacy_data(args.dev)
    print(f"Loaded {len(dev_examples)} dev examples")
    
    # Créer les datasets HuggingFace
    train_dataset = HFDataset.from_list(train_examples)
    dev_dataset = HFDataset.from_list(dev_examples)
    
    # Charger le tokenizer et le modèle
    print(f"Loading {args.model}...")
    tokenizer = CamembertTokenizerFast.from_pretrained(args.model)
    model = CamembertForTokenClassification.from_pretrained(
        args.model,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    
    # Tokenizer les données
    print("Tokenizing datasets...")
    train_tokenized = train_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    dev_tokenized = dev_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True,
        remove_columns=dev_dataset.column_names
    )
    
    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    # Arguments d'entraînement améliorés
    training_args = TrainingArguments(
        output_dir=args.output,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=args.warmup_ratio,  # Warmup pour stabiliser l'entraînement
        logging_dir=f"{args.output}/logs",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        push_to_hub=False,
        save_total_limit=3,
        fp16=torch.cuda.is_available(),  # Utiliser mixed precision si GPU disponible
        dataloader_num_workers=4,
        gradient_accumulation_steps=1,
    )
    
    # Créer le Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=dev_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Entraîner
    print("Starting training...")
    trainer.train()
    
    # Sauvegarder le modèle final
    print(f"Saving model to {args.output}/{args.version}...")
    trainer.save_model(f"{args.output}/{args.version}")
    tokenizer.save_pretrained(f"{args.output}/{args.version}")

    # Évaluation finale
    print("Final evaluation...")
    results = trainer.evaluate()
    print(results)
    
    print("Training completed!")

if __name__ == "__main__":
    main()