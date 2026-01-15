import argparse
from transformers import CamembertForTokenClassification, CamembertTokenizerFast
import torch

def test_model(model_path, text):
    """Test le modèle sur un texte donné"""
    print(f"Loading model from {model_path}...")
    
    tokenizer = CamembertTokenizerFast.from_pretrained(model_path)
    model = CamembertForTokenClassification.from_pretrained(model_path)
    model.eval()
    
    print(f"\nTexte : {text}\n")
    
    # Tokenization
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # Prédiction
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)
    
    # Récupérer les tokens et les labels
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    predicted_labels = [model.config.id2label[pred.item()] for pred in predictions[0]]
    
    # Regrouper les entités
    entities = []
    current_entity = None
    current_label = None
    
    for token, label in zip(tokens, predicted_labels):
        # Ignorer les tokens spéciaux
        if token in ["<s>", "</s>", "<pad>"]:
            continue
        
        # Nettoyer le token (enlever le préfixe Ġ de CamemBERT)
        clean_token = token.replace("Ġ", " ").replace("▁", " ")
        
        if label.startswith("B-"):
            # Sauvegarder l'entité précédente si elle existe
            if current_entity:
                entities.append((current_entity.strip(), current_label))
            # Commencer une nouvelle entité
            current_entity = clean_token
            current_label = label[2:]  # Enlever le "B-"
        
        elif label.startswith("I-") and current_entity is not None:
            # Continuer l'entité en cours
            current_entity += clean_token
        
        else:  # label == "O"
            # Sauvegarder l'entité précédente si elle existe
            if current_entity:
                entities.append((current_entity.strip(), current_label))
                current_entity = None
                current_label = None
    
    # Ajouter la dernière entité si elle existe
    if current_entity:
        entities.append((current_entity.strip(), current_label))
    
    # Afficher les résultats
    if entities:
        print("📍 Entités détectées :")
        entity_counts = {}
        for entity, label in entities:
            print(f"  - {entity} [{label}]")
            entity_counts[label] = entity_counts.get(label, 0) + 1
        
        print(f"\n📊 Résumé :")
        for label, count in entity_counts.items():
            print(f"  - {label}: {count} entité(s)")
    else:
        print("❌ Aucune entité détectée")
    
    # Afficher les tokens et labels pour debug
    print("\n🔍 Détails (tokens et labels) :")
    for token, label in zip(tokens, predicted_labels):
        if token not in ["<s>", "</s>", "<pad>"]:
            clean_token = token.replace("Ġ", " ").replace("▁", " ")
            print(f"  {clean_token:20s} -> {label}")

def main():
    parser = argparse.ArgumentParser(description="Test CamemBERT NER model")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--text", required=True, help="Text to test")
    
    args = parser.parse_args()
    
    test_model(args.model, args.text)

if __name__ == "__main__":
    main()