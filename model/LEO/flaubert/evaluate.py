"""
Script d'évaluation complète avec métriques détaillées
"""

import torch
from transformers import FlaubertTokenizer, FlaubertForSequenceClassification
import pandas as pd
import re
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class TravelOrderExtractor:
    def __init__(self, model_path='best_model.pt', geo_file='../dataset/geo.csv'):
        self.model_name = 'flaubert/flaubert_base_cased'
        self.tokenizer = FlaubertTokenizer.from_pretrained(self.model_name)
        self.model = FlaubertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2
        )
        
        # Charger le modèle entraîné
        try:
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            print("✓ Modèle chargé avec succès")
        except FileNotFoundError:
            print("⚠ Modèle non trouvé, utilisation du modèle de base")
        
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Charger les entités géographiques
        self.load_locations(geo_file)
        
    def load_locations(self, geo_file):
        """Charge les entités géographiques"""
        geo_df = pd.read_csv(geo_file, header=None, names=['name', 'type'])
        self.locations = set(geo_df['name'].str.lower())
        self.location_dict = dict(zip(geo_df['name'].str.lower(), geo_df['name']))
        print(f"✓ {len(self.locations)} entités géographiques chargées")
    
    def normalize_location(self, location):
        """Normalise le nom d'une ville"""
        if not location:
            return None
            
        location_lower = location.lower().strip()
        
        # Recherche exacte
        if location_lower in self.location_dict:
            return self.location_dict[location_lower]
        
        # Retourner avec première lettre majuscule si non trouvé
        return location.capitalize()
    
    def extract_locations_rule_based(self, text):
        """Extraction basée sur des règles et le dictionnaire"""
        text_lower = text.lower()
        
        # Patterns pour identifier départ et arrivée
        from_patterns = [
            r'(?:de|depuis|partir de|venant de|partant de)\s+([a-zàâäéèêëïîôùûüÿçæœ-]+)',
            r'^([a-zàâäéèêëïîôùûüÿçæœ-]+)\s+(?:vers|à|pour)',
        ]
        
        to_patterns = [
            r'(?:vers|à|pour|jusqu\'à|sur)\s+([a-zàâäéèêëïîôùûüÿçæœ-]+)',
            r'(?:de|depuis)\s+[a-zàâäéèêëïîôùûüÿçæœ-]+\s+(?:à|vers|a)\s+([a-zàâäéèêëïîôùûüÿçæœ-]+)',
        ]
        
        from_city = None
        to_city = None
        
        # Chercher ville de départ
        for pattern in from_patterns:
            match = re.search(pattern, text_lower)
            if match:
                candidate = match.group(1)
                if candidate in self.locations:
                    from_city = self.normalize_location(candidate)
                    break
        
        # Chercher ville d'arrivée
        for pattern in to_patterns:
            match = re.search(pattern, text_lower)
            if match:
                candidate = match.group(1)
                if candidate in self.locations:
                    to_city = self.normalize_location(candidate)
                    break
        
        return from_city, to_city
    
    def is_valid_travel_query(self, text):
        """Vérifie si la requête est valide"""
        text_lower = text.lower()
        
        # Mots-clés invalides
        invalid_keywords = ['restaurant', 'hôtel', 'hotel', 'météo', 'meteo']
        if any(keyword in text_lower for keyword in invalid_keywords):
            return False
        
        # Doit contenir des mots-clés de voyage
        travel_keywords = ['train', 'voyage', 'aller', 'partir', 'de', 'vers', 'à']
        has_travel_keyword = any(keyword in text_lower for keyword in travel_keywords)
        
        return has_travel_keyword
    
    def predict(self, text):
        """Prédit les villes de départ et d'arrivée"""
        # Vérifier si la requête est valide
        if not self.is_valid_travel_query(text):
            return "INVALID", None
        
        # Extraction basée sur des règles
        from_city, to_city = self.extract_locations_rule_based(text)
        
        if from_city and to_city:
            return from_city, to_city
        elif from_city or to_city:
            return from_city if from_city else "INVALID", to_city
        else:
            return "INVALID", None

def plot_confusion_matrix(cm, filename='confusion_matrix.png'):
    """Génère et sauvegarde la matrice de confusion"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['INVALID', 'VALID'],
                yticklabels=['INVALID', 'VALID'],
                cbar_kws={'label': 'Nombre de prédictions'})
    
    plt.title('Matrice de Confusion - FlauBERT Travel Order', fontsize=16, fontweight='bold')
    plt.ylabel('Vraie classe', fontsize=12)
    plt.xlabel('Classe prédite', fontsize=12)
    
    # Ajouter des statistiques
    total = cm.sum()
    accuracy = (cm[0,0] + cm[1,1]) / total
    plt.text(0.5, -0.15, f'Accuracy globale: {accuracy:.2%}', 
             ha='center', transform=plt.gca().transAxes, fontsize=11)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Matrice de confusion sauvegardée dans: {filename}")

def save_results_to_file(results, filename='evaluation_results.txt'):
    """Sauvegarde tous les résultats dans un fichier texte"""
    with open(filename, 'w', encoding='utf-8') as f:
        # En-tête
        f.write("="*80 + "\n")
        f.write("RAPPORT D'ÉVALUATION - FlauBERT Travel Order\n")
        f.write("="*80 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total d'exemples: {results['total']}\n")
        f.write("\n")
        
        # Métriques principales
        f.write("-"*80 + "\n")
        f.write("MÉTRIQUES PRINCIPALES\n")
        f.write("-"*80 + "\n")
        f.write(f"Loss moyenne:                {results['avg_loss']:.4f}\n")
        f.write(f"Accuracy (ville de départ):  {results['accuracy_from']:.2%}\n")
        f.write(f"Accuracy (ville d'arrivée):  {results['accuracy_to']:.2%}\n")
        f.write(f"Accuracy (les deux villes):  {results['accuracy_both']:.2%}\n")
        f.write(f"\nPrecision:                   {results['precision']:.4f}\n")
        f.write(f"Recall:                      {results['recall']:.4f}\n")
        f.write(f"F1-Score:                    {results['f1']:.4f}\n")
        f.write("\n")
        
        # Matrice de confusion
        f.write("-"*80 + "\n")
        f.write("MATRICE DE CONFUSION\n")
        f.write("-"*80 + "\n")
        cm = results['confusion_matrix']
        f.write("                Prédit INVALID  Prédit VALID\n")
        f.write(f"Vrai INVALID    {cm[0][0]:6d}         {cm[0][1]:6d}\n")
        f.write(f"Vrai VALID      {cm[1][0]:6d}         {cm[1][1]:6d}\n")
        f.write("\n")
        
        # Erreurs détaillées
        if results['errors']:
            f.write("-"*80 + "\n")
            f.write(f"ERREURS DÉTAILLÉES ({len(results['errors'])} erreurs au total)\n")
            f.write("-"*80 + "\n")
            for i, error in enumerate(results['errors'][:50], 1):  # Max 50 erreurs
                f.write(f"\n[Erreur {i}] Ligne {error['index']}\n")
                f.write(f"Texte: {error['text']}\n")
                f.write(f"  Attendu:  {error['from_expected']} → {error['to_expected']}\n")
                f.write(f"  Prédit:   {error['from_predicted']} → {error['to_predicted']}\n")
            
            if len(results['errors']) > 50:
                f.write(f"\n... et {len(results['errors']) - 50} autres erreurs\n")
        
        # Résumé final
        f.write("\n" + "="*80 + "\n")
        f.write("RÉSUMÉ\n")
        f.write("="*80 + "\n")
        f.write(f"Score global: {results['accuracy_both']:.2%}\n")
        f.write(f"F1-Score: {results['f1']:.4f}\n")
        f.write(f"Nombre d'erreurs: {len(results['errors'])}/{results['total']}\n")
    
    print(f"✓ Résultats sauvegardés dans: {filename}")
    """Charge les données de test"""
    with open(input_file, 'r', encoding='utf-8') as f:
        inputs = [line.strip().split(',', 1)[1] for line in f if line.strip()]
    
    with open(output_file, 'r', encoding='utf-8') as f:
        outputs = []
        for line in f:
            if line.strip():
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    from_loc = parts[1] if len(parts) > 1 else 'INVALID'
                    to_loc = parts[2] if len(parts) > 2 else None
                    outputs.append((from_loc, to_loc))
    
    return inputs, outputs

def load_test_data(input_file, output_file):
    """Charge les données de test"""
    with open(input_file, 'r', encoding='utf-8') as f:
        inputs = [line.strip().split(',', 1)[1] for line in f if line.strip()]
    
    with open(output_file, 'r', encoding='utf-8') as f:
        outputs = []
        for line in f:
            if line.strip():
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    from_loc = parts[1] if len(parts) > 1 else 'INVALID'
                    to_loc = parts[2] if len(parts) > 2 else None
                    outputs.append((from_loc, to_loc))
    
    return inputs, outputs

def evaluate_model(extractor, inputs, expected_outputs):
    """Évalue le modèle et calcule les métriques"""
    print("\n" + "="*80)
    print("ÉVALUATION DU MODÈLE")
    print("="*80)
    
    # Prédictions et calcul de la loss
    predictions = []
    total_loss = 0.0
    loss_fn = torch.nn.CrossEntropyLoss()
    
    print("\nGénération des prédictions et calcul de la loss...")
    for text, (from_exp, to_exp) in tqdm(zip(inputs, expected_outputs), total=len(inputs)):
        from_pred, to_pred = extractor.predict(text)
        predictions.append((from_pred, to_pred))
        
        # Calcul de la loss
        # Tokenize
        encoding = extractor.tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(extractor.device)
        attention_mask = encoding['attention_mask'].to(extractor.device)
        
        # Label: 0=INVALID, 1=VALID
        label = torch.tensor([0 if from_exp == 'INVALID' else 1], dtype=torch.long).to(extractor.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = extractor.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=label
            )
            total_loss += outputs.loss.item()
    
    avg_loss = total_loss / len(inputs)
    
    # Métriques globales
    total = len(inputs)
    correct_from = 0
    correct_to = 0
    correct_both = 0
    correct_invalid = 0
    
    # Pour classification binaire (VALID/INVALID)
    y_true_binary = []
    y_pred_binary = []
    
    # Détails des erreurs
    errors = []
    
    for i, (text, (from_exp, to_exp), (from_pred, to_pred)) in enumerate(
        zip(inputs, expected_outputs, predictions), 1
    ):
        # Classification binaire
        is_valid_exp = (from_exp != 'INVALID')
        is_valid_pred = (from_pred != 'INVALID')
        
        y_true_binary.append(1 if is_valid_exp else 0)
        y_pred_binary.append(1 if is_valid_pred else 0)
        
        # Comptage des corrections
        if from_exp == from_pred:
            correct_from += 1
        
        if to_exp == to_pred:
            correct_to += 1
        
        if from_exp == from_pred and to_exp == to_pred:
            correct_both += 1
        
        if from_exp == 'INVALID' and from_pred == 'INVALID':
            correct_invalid += 1
        
        # Collecter les erreurs
        if from_exp != from_pred or to_exp != to_pred:
            errors.append({
                'index': i,
                'text': text,
                'from_expected': from_exp,
                'from_predicted': from_pred,
                'to_expected': to_exp,
                'to_predicted': to_pred
            })
    
    # Calcul des métriques
    accuracy_from = correct_from / total
    accuracy_to = correct_to / total
    accuracy_both = correct_both / total
    
    # Métriques binaires (VALID/INVALID)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_binary, y_pred_binary, average='binary'
    )
    
    # Matrice de confusion
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    
    # Affichage des résultats
    print("\n" + "-"*80)
    print("MÉTRIQUES GLOBALES")
    print("-"*80)
    print(f"Total d'exemples: {total}")
    print(f"Loss moyenne:                {avg_loss:.4f}")
    print(f"\nAccuracy (ville de départ):  {accuracy_from:.2%} ({correct_from}/{total})")
    print(f"Accuracy (ville d'arrivée):  {accuracy_to:.2%} ({correct_to}/{total})")
    print(f"Accuracy (les deux villes):  {accuracy_both:.2%} ({correct_both}/{total})")
    print(f"Détection INVALID correcte:  {correct_invalid}/{sum(1 for y in y_true_binary if y == 0)}")
    
    print("\n" + "-"*80)
    print("MÉTRIQUES CLASSIFICATION BINAIRE (VALID/INVALID)")
    print("-"*80)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    print("\n" + "-"*80)
    print("MATRICE DE CONFUSION")
    print("-"*80)
    print("                Prédit INVALID  Prédit VALID")
    print(f"Vrai INVALID    {cm[0][0]:6d}         {cm[0][1]:6d}")
    print(f"Vrai VALID      {cm[1][0]:6d}         {cm[1][1]:6d}")
    
    # Rapport de classification détaillé
    print("\n" + "-"*80)
    print("RAPPORT DE CLASSIFICATION")
    print("-"*80)
    print(classification_report(
        y_true_binary, y_pred_binary,
        target_names=['INVALID', 'VALID']
    ))
    
    # Afficher quelques erreurs
    if errors:
        print("\n" + "-"*80)
        print(f"EXEMPLES D'ERREURS (10 premiers sur {len(errors)})")
        print("-"*80)
        for error in errors[:10]:
            print(f"\n[{error['index']}] {error['text']}")
            print(f"  Attendu:  {error['from_expected']} → {error['to_expected']}")
            print(f"  Prédit:   {error['from_predicted']} → {error['to_predicted']}")
    
    # Sauvegarder les résultats
    results = {
        'total': total,
        'avg_loss': avg_loss,
        'accuracy_from': accuracy_from,
        'accuracy_to': accuracy_to,
        'accuracy_both': accuracy_both,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm.tolist(),
        'errors': errors
    }
    
    return results

def main():
    print("=== FlauBERT Travel Order - Évaluation ===\n")
    
    # Initialiser l'extracteur
    extractor = TravelOrderExtractor()
    
    # Charger les données de test
    print("\nChargement des données de test...")
    inputs, outputs = load_test_data(
        '../dataset/dataset_input.txt',
        '../dataset/dataset_output.txt'
    )
    print(f"✓ {len(inputs)} exemples chargés")
    
    # Évaluer
    results = evaluate_model(extractor, inputs, outputs)
    
    # Générer et sauvegarder la matrice de confusion
    cm = np.array(results['confusion_matrix'])
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_confusion_matrix(cm, f'confusion_matrix_{timestamp}.png')
    
    # Sauvegarder les résultats dans un fichier texte
    save_results_to_file(results, f'evaluation_results_{timestamp}.txt')
    
    print("\n" + "="*80)
    print("✓ ÉVALUATION TERMINÉE")
    print("="*80)
    print(f"\nLoss moyenne: {results['avg_loss']:.4f}")
    print(f"Score global: {results['accuracy_both']:.2%}")
    print(f"F1-Score: {results['f1']:.4f}")
    print(f"\n✓ Fichiers générés:")
    print(f"  - confusion_matrix_{timestamp}.png")
    print(f"  - evaluation_results_{timestamp}.txt")

if __name__ == '__main__':
    main()