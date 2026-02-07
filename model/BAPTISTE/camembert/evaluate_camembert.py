import torch
import numpy as np
import argparse
from transformers import CamembertForTokenClassification, CamembertTokenizerFast
from torch.utils.data import DataLoader
from datasets import Dataset as HFDataset
import spacy
from spacy.tokens import DocBin
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd

# Étiquettes (doit correspondre à votre modèle)
LABEL_LIST = ["O", "B-LOC_CITY", "I-LOC_CITY", "B-LOC_DEPARTMENT", "I-LOC_DEPARTMENT", 
              "B-LOC_REGION", "I-LOC_REGION", "B-LOC_GARE", "I-LOC_GARE", 
              "B-LOC_AEROPORT", "I-LOC_AEROPORT"]

def load_spacy_data(spacy_file):
    """Charge les données depuis un fichier .spacy"""
    nlp = spacy.blank("fr")
    doc_bin = DocBin().from_disk(spacy_file)
    docs = list(doc_bin.get_docs(nlp.vocab))
    
    examples = []
    for doc in docs:
        tokens = [token.text for token in doc]
        ner_tags = ["O"] * len(tokens)
        
        for ent in doc.ents:
            start_token = None
            end_token = None
            
            for i, token in enumerate(doc):
                if token.idx >= ent.start_char and start_token is None:
                    start_token = i
                if token.idx + len(token.text) <= ent.end_char:
                    end_token = i
            
            if start_token is not None and end_token is not None:
                ner_tags[start_token] = f"B-{ent.label_}"
                for i in range(start_token + 1, end_token + 1):
                    ner_tags[i] = f"I-{ent.label_}"
        
        examples.append({
            "tokens": tokens,
            "ner_tags": ner_tags,
            "text": doc.text
        })
    
    return examples

def tokenize_and_align_labels(examples, tokenizer, label2id):
    """Tokenize et aligne les labels"""
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
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label[word_idx]])
            else:
                current_label = label[word_idx]
                if current_label.startswith("B-"):
                    label_ids.append(label2id[current_label.replace("B-", "I-")])
                else:
                    label_ids.append(label2id[current_label])
            previous_word_idx = word_idx
        
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def evaluate_model(model_path, test_file, output_dir="./evaluation_results"):
    """Évalue le modèle et génère tous les rapports"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading model from {model_path}...")
    tokenizer = CamembertTokenizerFast.from_pretrained(model_path)
    model = CamembertForTokenClassification.from_pretrained(model_path)
    model.eval()
    
    # Récupérer les mappings du modèle
    id2label = model.config.id2label
    label2id = model.config.label2id
    label_list = [id2label[i] for i in range(len(id2label))]
    
    print(f"Loading test data from {test_file}...")
    test_examples = load_spacy_data(test_file)
    print(f"Loaded {len(test_examples)} test examples")
    
    # Préparer les données
    test_dataset = HFDataset.from_list(test_examples)
    test_tokenized = test_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, label2id),
        batched=True,
        remove_columns=["tokens", "ner_tags", "text"]
    )
    
    # Prédictions
    print("Making predictions...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0
    
    for example in test_tokenized:
        input_ids = torch.tensor([example["input_ids"]]).to(device)
        attention_mask = torch.tensor([example["attention_mask"]]).to(device)
        labels = torch.tensor([example["labels"]]).to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item()
            num_batches += 1
            
            predictions = torch.argmax(logits, dim=2)
            
            # Extraire les prédictions et labels (ignorer -100)
            pred_labels = predictions[0].cpu().numpy()
            true_labels = labels[0].cpu().numpy()
            
            pred_tags = []
            true_tags = []
            for p, l in zip(pred_labels, true_labels):
                if l != -100:
                    pred_tags.append(label_list[p])
                    true_tags.append(label_list[l])
            
            all_predictions.append(pred_tags)
            all_labels.append(true_tags)
    
    avg_loss = total_loss / num_batches
    
    # Calculer les métriques avec seqeval
    print("\n" + "="*80)
    print("MÉTRIQUES GLOBALES")
    print("="*80)
    
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    
    print(f"\n{'Métrique':<20} {'Valeur':>10}")
    print("-" * 32)
    print(f"{'Loss':<20} {avg_loss:>10.4f}")
    print(f"{'Precision':<20} {precision:>10.4f}")
    print(f"{'Recall':<20} {recall:>10.4f}")
    print(f"{'F1-Score':<20} {f1:>10.4f}")
    
    # Rapport détaillé par classe
    print("\n" + "="*80)
    print("MÉTRIQUES PAR CLASSE")
    print("="*80)
    report = classification_report(all_labels, all_predictions, digits=4)
    print(report)
    
    # Sauvegarder le rapport dans un fichier
    with open(f"{output_dir}/classification_report.txt", "w") as f:
        f.write("MÉTRIQUES GLOBALES\n")
        f.write("="*80 + "\n")
        f.write(f"Loss: {avg_loss:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n\n")
        f.write("MÉTRIQUES PAR CLASSE\n")
        f.write("="*80 + "\n")
        f.write(report)
    
    # Créer un DataFrame pour les métriques globales
    metrics_df = pd.DataFrame({
        'Métrique': ['Loss', 'Precision', 'Recall', 'F1-Score'],
        'Valeur': [avg_loss, precision, recall, f1]
    })
    metrics_df.to_csv(f"{output_dir}/global_metrics.csv", index=False)
    
    # Matrice de confusion
    print("\n" + "="*80)
    print("MATRICE DE CONFUSION")
    print("="*80)
    
    # Aplatir les listes pour la matrice de confusion
    flat_predictions = [item for sublist in all_predictions for item in sublist]
    flat_labels = [item for sublist in all_labels for item in sublist]
    
    # Labels uniques présents dans les données
    unique_labels = sorted(list(set(flat_labels + flat_predictions)))
    
    # Calculer la matrice de confusion
    cm = confusion_matrix(flat_labels, flat_predictions, labels=unique_labels)
    
    # Afficher la matrice dans la console
    print(f"\n{'':>15}", end="")
    for label in unique_labels:
        print(f"{label:>15}", end="")
    print()
    print("-" * (15 * (len(unique_labels) + 1)))
    
    for i, label in enumerate(unique_labels):
        print(f"{label:>15}", end="")
        for j in range(len(unique_labels)):
            print(f"{cm[i][j]:>15}", end="")
        print()
    
    # Sauvegarder la matrice de confusion en CSV
    cm_df = pd.DataFrame(cm, index=unique_labels, columns=unique_labels)
    cm_df.to_csv(f"{output_dir}/confusion_matrix.csv")
    print(f"\n✅ Matrice de confusion sauvegardée dans {output_dir}/confusion_matrix.csv")
    
    # Visualiser la matrice de confusion avec matplotlib
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=unique_labels, yticklabels=unique_labels,
                cbar_kws={'label': 'Nombre de prédictions'})
    plt.title('Matrice de Confusion - Modèle CamemBERT NER', fontsize=16, pad=20)
    plt.xlabel('Prédictions', fontsize=12)
    plt.ylabel('Vraies étiquettes', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
    print(f"✅ Matrice de confusion visualisée dans {output_dir}/confusion_matrix.png")
    
    # Matrice de confusion normalisée (en pourcentage)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=unique_labels, yticklabels=unique_labels,
                cbar_kws={'label': 'Pourcentage (%)'})
    plt.title('Matrice de Confusion Normalisée (%) - Modèle CamemBERT NER', 
              fontsize=16, pad=20)
    plt.xlabel('Prédictions', fontsize=12)
    plt.ylabel('Vraies étiquettes', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix_normalized.png", dpi=300, bbox_inches='tight')
    print(f"✅ Matrice de confusion normalisée dans {output_dir}/confusion_matrix_normalized.png")
    
    # Analyse des erreurs les plus fréquentes
    print("\n" + "="*80)
    print("ERREURS LES PLUS FRÉQUENTES")
    print("="*80)
    
    errors = defaultdict(int)
    for true, pred in zip(flat_labels, flat_predictions):
        if true != pred:
            errors[(true, pred)] += 1
    
    sorted_errors = sorted(errors.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print(f"\n{'Vraie étiquette':<20} {'Prédiction':<20} {'Nombre':>10}")
    print("-" * 52)
    for (true_label, pred_label), count in sorted_errors:
        print(f"{true_label:<20} {pred_label:<20} {count:>10}")
    
    # Sauvegarder les erreurs
    errors_df = pd.DataFrame(sorted_errors, columns=['Erreur', 'Nombre'])
    errors_df[['Vraie_étiquette', 'Prédiction']] = pd.DataFrame(errors_df['Erreur'].tolist())
    errors_df = errors_df[['Vraie_étiquette', 'Prédiction', 'Nombre']]
    errors_df.to_csv(f"{output_dir}/top_errors.csv", index=False)
    
    print("\n" + "="*80)
    print("RÉSUMÉ DES FICHIERS GÉNÉRÉS")
    print("="*80)
    print(f"✅ {output_dir}/classification_report.txt")
    print(f"✅ {output_dir}/global_metrics.csv")
    print(f"✅ {output_dir}/confusion_matrix.csv")
    print(f"✅ {output_dir}/confusion_matrix.png")
    print(f"✅ {output_dir}/confusion_matrix_normalized.png")
    print(f"✅ {output_dir}/top_errors.csv")
    
    return {
        'loss': avg_loss,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'labels': unique_labels
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate CamemBERT NER model")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--test", required=True, help="Path to test .spacy file")
    parser.add_argument("--output", default="./results", 
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    evaluate_model(args.model, args.test, args.output)

if __name__ == "__main__":
    main()