"""
Script d'inférence pour extraire les villes de départ et d'arrivée
"""

import torch
from transformers import FlaubertTokenizer, FlaubertForSequenceClassification
import pandas as pd
import re

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
        location_lower = location.lower().strip()
        
        # Recherche exacte
        if location_lower in self.location_dict:
            return self.location_dict[location_lower]
        
        # Recherche avec majuscules
        for loc in self.location_dict.keys():
            if loc == location_lower:
                return self.location_dict[loc]
        
        # Retourner avec première lettre majuscule si non trouvé
        return location.capitalize()
    
    def extract_locations_rule_based(self, text):
        """Extraction basée sur des règles et le dictionnaire"""
        text_lower = text.lower()
        
        # Patterns pour identifier départ et arrivée
        from_patterns = [
            r'(?:de|depuis|partir de|venant de)\s+([a-zàâäéèêëïîôùûüÿçæœ-]+)',
            r'^([a-zàâäéèêëïîôùûüÿçæœ-]+)\s+(?:vers|à|pour)',
        ]
        
        to_patterns = [
            r'(?:vers|à|pour|jusqu\'à)\s+([a-zàâäéèêëïîôùûüÿçæœ-]+)',
            r'(?:de|depuis)\s+[a-zàâäéèêëïîôùûüÿçæœ-]+\s+(?:à|vers)\s+([a-zàâäéèêëïîôùûüÿçæœ-]+)',
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
            # Partiel, essayer de compléter
            return from_city if from_city else "INVALID", to_city
        else:
            return "INVALID", None
    
    def predict_batch(self, texts):
        """Prédit pour un batch de textes"""
        results = []
        for text in texts:
            from_city, to_city = self.predict(text)
            results.append((from_city, to_city))
        return results

def main():
    print("=== FlauBERT Travel Order - Inférence ===\n")
    
    # Initialiser l'extracteur
    extractor = TravelOrderExtractor()
    
    # Exemples de test
    test_examples = [
        "je cherche un restaurant",
        "y a-t-il des trains de limoges à villeurbanne",
        "Un voyage de amiens a vitry-sur-seine s'il vous plait",
        "je voudrai aller de orléans à villeurbanne",
        "de grenoble vers colombes",
        "trains de valerie vers colombes",
        "je voudrais aller de rennes a strasbourg",
    ]
    
    print("\nTests d'inférence:")
    print("-" * 80)
    
    for text in test_examples:
        from_city, to_city = extractor.predict(text)
        print(f"Texte: {text}")
        print(f"  → Départ: {from_city if from_city else 'N/A'}")
        print(f"  → Arrivée: {to_city if to_city else 'N/A'}")
        print()
    
    # Mode interactif
    print("\n" + "="*80)
    print("Mode interactif (tapez 'quit' pour quitter)")
    print("="*80)
    
    while True:
        try:
            user_input = input("\nEntrez une commande de voyage: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            from_city, to_city = extractor.predict(user_input)
            print(f"  ✓ Départ: {from_city if from_city else 'N/A'}")
            print(f"  ✓ Arrivée: {to_city if to_city else 'N/A'}")
            
        except KeyboardInterrupt:
            break
    
    print("\nAu revoir!")

if __name__ == '__main__':
    main()