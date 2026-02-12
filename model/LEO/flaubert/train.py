"""
FlauBERT Travel Order - Extraction d'entités nommées pour les commandes de voyage
Utilise FlauBERT pour identifier les villes de départ et d'arrivée
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import FlaubertTokenizer, FlaubertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

class TravelOrderDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenization
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data(input_file, output_file, geo_file):
    """Charge et prépare les données"""
    # Charger les données
    with open(input_file, 'r', encoding='utf-8') as f:
        inputs = [line.strip().split(',', 1)[1] for line in f if line.strip()]
    
    with open(output_file, 'r', encoding='utf-8') as f:
        outputs = []
        for line in f:
            if line.strip():
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    outputs.append((parts[1], parts[2] if len(parts) > 2 else ''))
    
    # Charger les entités géographiques
    geo_df = pd.read_csv(geo_file, header=None, names=['name', 'type'])
    locations = set(geo_df['name'].str.lower())
    
    return inputs, outputs, locations

def prepare_labels(inputs, outputs, locations):
    """Prépare les labels pour l'entraînement
    0: INVALID
    1: VALID (contient départ et arrivée)
    """
    labels = []
    
    for i, (text, (from_loc, to_loc)) in enumerate(zip(inputs, outputs)):
        if from_loc == 'INVALID':
            labels.append(0)  # INVALID
        else:
            labels.append(1)  # VALID
    
    return labels

def train_model(model, train_loader, val_loader, device, epochs=5):
    """Entraîne le modèle"""
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} - Training'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculer l'accuracy
            predictions = torch.argmax(outputs.logits, dim=1)
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
        
        train_accuracy = train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} - Validation'):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                val_loss += outputs.loss.item()
                
                predictions = torch.argmax(outputs.logits, dim=1)
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
        
        val_accuracy = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'\nEpoch {epoch+1}:')
        print(f'  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f}')
        
        # Sauvegarder le meilleur modèle
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pt')
            print('  → Modèle sauvegardé!')

def main():
    # Configuration
    MODEL_NAME = 'flaubert/flaubert_base_cased'
    BATCH_SIZE = 16
    EPOCHS = 5
    MAX_LENGTH = 128
    
    # Charger les données
    print("Chargement des données...")
    inputs, outputs, locations = load_data(
        '../dataset/dataset_input.txt',
        '../dataset/dataset_output.txt',
        '../dataset/geo.csv'
    )
    
    # Préparer les labels
    print("Préparation des labels...")
    labels = prepare_labels(inputs, outputs, locations)
    
    # Split train/val
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        inputs, labels, test_size=0.2, random_state=42
    )
    
    print(f"Données d'entraînement: {len(train_texts)}")
    print(f"Données de validation: {len(val_texts)}")
    print(f"Classes: 0=INVALID, 1=VALID")
    
    # Initialiser le tokenizer et le modèle
    print("Chargement du modèle FlauBERT...")
    tokenizer = FlaubertTokenizer.from_pretrained(MODEL_NAME)
    model = FlaubertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2  # INVALID ou VALID
    )
    
    # Préparer les datasets
    train_dataset = TravelOrderDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = TravelOrderDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation du device: {device}")
    model.to(device)
    
    # Entraînement
    print("\nDébut de l'entraînement...")
    best_val_loss = train_model(model, train_loader, val_loader, device, EPOCHS)
    
    print("\nEntraînement terminé!")
    if best_val_loss is not None:
        print(f"Le meilleur modèle a été sauvegardé (validation loss: {best_val_loss:.4f})")

if __name__ == '__main__':
    main()