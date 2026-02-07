"""
Script pour tester le modèle BERT fine-tuné sur la détection de Départ/Arrivée
"""

import os
import numpy as np
from transformers import BertTokenizerFast, TFBertForTokenClassification
from sklearn.preprocessing import LabelEncoder

# Configuration
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'bert_europeana_ner')
UNIQUE_LABELS = ['O', 'B-DEP', 'I-DEP', 'B-ARR', 'I-ARR']


def load_model():
    """Charge le modèle et le tokenizer"""
    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(f"Modèle non trouvé dans {MODEL_DIR}. Entraînez d'abord le modèle avec finetune_bert_europeana.py")
    
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)
    model = TFBertForTokenClassification.from_pretrained(MODEL_DIR)
    
    label_encoder = LabelEncoder()
    label_encoder.fit(UNIQUE_LABELS)
    
    return tokenizer, model, label_encoder


def extract_entities(tokens, labels):
    """Extrait les entités Départ et Arrivée depuis les tokens et labels"""
    departure = []
    arrival = []
    
    for token, label in zip(tokens, labels):
        # Ignorer les tokens spéciaux et padding
        if token in ['[CLS]', '[SEP]', '[PAD]']:
            continue
        
        # Nettoyer les wordpieces (##)
        clean_token = token.replace('##', '')
        
        if label in ['B-DEP', 'I-DEP']:
            departure.append(clean_token)
        elif label in ['B-ARR', 'I-ARR']:
            arrival.append(clean_token)
    
    return ' '.join(departure), ' '.join(arrival)


def predict(phrase, tokenizer, model, label_encoder, verbose=True):
    """Prédit les entités Départ/Arrivée pour une phrase"""
    # Encoder la phrase
    encoding = tokenizer(
        phrase,
        max_length=64,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )
    
    # Prédiction
    outputs = model(encoding)
    predictions = np.argmax(outputs.logits, axis=-1)[0]
    
    # Convertir les IDs en labels
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
    predicted_labels = [label_encoder.classes_[p] for p in predictions]
    
    # Extraire les entités
    departure, arrival = extract_entities(tokens, predicted_labels)
    
    if verbose:
        print(f"\n📝 Phrase: {phrase}")
        print(f"🚀 Départ:  {departure if departure else 'NON DÉTECTÉ'}")
        print(f"🎯 Arrivée: {arrival if arrival else 'NON DÉTECTÉ'}")
        
        # Afficher les tokens avec labels pour debug
        print(f"\n🔍 Détails (tokens + labels):")
        for i, (token, label) in enumerate(zip(tokens[:20], predicted_labels[:20])):
            if token not in ['[CLS]', '[SEP]', '[PAD]']:
                marker = "→" if label != 'O' else " "
                print(f"  {marker} {token:15s} {label}")
    
    return departure, arrival, tokens, predicted_labels


def test_examples(tokenizer, model, label_encoder):
    """Teste le modèle sur des exemples"""
    test_phrases = [
        "je voudrais aller de Paris à Lyon",
        "comment aller de Marseille à Nice",
        "un billet pour Toulouse depuis Bordeaux",
        "Je souhaite me rendre à sainte julie depuis fontenoy",
        "trains de Montpellier vers Nantes",
        "Horaires de Lille à Strasbourg",
        "avec mon ami Albert je veux aller de Rennes à Dijon",
        "bonjour comment allez-vous",  # Phrase invalide (pas de trajet)
        "je voudrais réserver une chambre d'hôtel",  # Phrase invalide
        "de ambérieu-en-bugey à neuville-les-dames",
        "de paris à lyon pour boire un monaco",
        "je veux voir paris à lyon depuis bordeaux",
    ]
    
    print("=" * 80)
    print("🧪 TEST DU MODÈLE BERT - Détection Départ/Arrivée")
    print("=" * 80)
    
    correct = 0
    total = 0
    
    for phrase in test_phrases:
        departure, arrival, tokens, labels = predict(phrase, tokenizer, model, label_encoder, verbose=True)
        print("-" * 80)
        total += 1
    
    print(f"\n✅ {total} phrases testées")


def evaluate_on_dataset(tokenizer, model, label_encoder, test_file=None):
    """Évalue le modèle sur un dataset de test"""
    if test_file is None:
        # Utiliser un subset du dataset pour test rapide
        dataset_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'dataset'))
        test_file = os.path.join(dataset_dir, 'dataset_test_input.txt')
        test_output = os.path.join(dataset_dir, 'dataset_test_output.txt')
    
    if not os.path.exists(test_file):
        print(f"⚠️  Fichier de test non trouvé: {test_file}")
        return
    
    print("\n" + "=" * 80)
    print(f"📊 ÉVALUATION SUR DATASET: {test_file}")
    print("=" * 80)
    
    correct_dep = 0
    correct_arr = 0
    total = 0
    
    with open(test_file, 'r', encoding='utf-8') as f_in, \
         open(test_output, 'r', encoding='utf-8') as f_out:
        
        for line_in, line_out in zip(f_in, f_out):
            parts_in = line_in.strip().split(',', 1)
            parts_out = line_out.strip().split(',')
            
            if len(parts_in) < 2 or len(parts_out) < 3:
                continue
            
            phrase = parts_in[1]
            expected_dep = parts_out[1].strip()
            expected_arr = parts_out[2].strip()
            
            if expected_dep == 'INVALID':
                continue
            
            # Prédiction
            pred_dep, pred_arr, _, _ = predict(phrase, tokenizer, model, label_encoder, verbose=False)
            
            # Normaliser pour comparaison (minuscules, espaces)
            pred_dep_norm = pred_dep.lower().replace(' ', '')
            pred_arr_norm = pred_arr.lower().replace(' ', '')
            exp_dep_norm = expected_dep.lower().replace('-', '').replace(' ', '')
            exp_arr_norm = expected_arr.lower().replace('-', '').replace(' ', '')
            
            if pred_dep_norm == exp_dep_norm:
                correct_dep += 1
            if pred_arr_norm == exp_arr_norm:
                correct_arr += 1
            
            total += 1
            
            if total <= 5:  # Afficher les 5 premiers exemples
                print(f"\n{total}. {phrase[:60]}...")
                print(f"   Attendu: {expected_dep} → {expected_arr}")
                print(f"   Prédit:  {pred_dep} → {pred_arr}")
                dep_ok = "✓" if pred_dep_norm == exp_dep_norm else "✗"
                arr_ok = "✓" if pred_arr_norm == exp_arr_norm else "✗"
                print(f"   Résultat: Départ {dep_ok} | Arrivée {arr_ok}")
            
            if total >= 100:  # Limiter à 100 phrases pour test rapide
                break
    
    print("\n" + "=" * 80)
    print(f"📈 RÉSULTATS ({total} phrases testées)")
    print(f"   Départs corrects:  {correct_dep}/{total} ({100*correct_dep/total:.1f}%)")
    print(f"   Arrivées correctes: {correct_arr}/{total} ({100*correct_arr/total:.1f}%)")
    print(f"   Accuracy globale:   {(correct_dep+correct_arr)/(2*total):.1%}")
    print("=" * 80)


def interactive_mode(tokenizer, model, label_encoder):
    """Mode interactif pour tester des phrases"""
    print("\n" + "=" * 80)
    print("🎮 MODE INTERACTIF")
    print("Entrez des phrases pour tester le modèle (tapez 'quit' pour sortir)")
    print("=" * 80)
    
    while True:
        try:
            phrase = input("\n📝 Phrase: ").strip()
            if phrase.lower() in ['quit', 'exit', 'q']:
                break
            if not phrase:
                continue
            
            predict(phrase, tokenizer, model, label_encoder, verbose=True)
            print("-" * 80)
        
        except KeyboardInterrupt:
            print("\n\nInterrompu par l'utilisateur")
            break
        except Exception as e:
            print(f"❌ Erreur: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Tester le modèle BERT fine-tuné")
    parser.add_argument('--interactive', '-i', action='store_true', help='Mode interactif')
    parser.add_argument('--evaluate', '-e', action='store_true', help='Évaluer sur dataset de test')
    parser.add_argument('--phrase', '-p', type=str, help='Tester une phrase spécifique')
    args = parser.parse_args()
    
    print("Chargement du modèle...")
    tokenizer, model, label_encoder = load_model()
    print(f"✓ Modèle chargé depuis {MODEL_DIR}\n")
    
    if args.phrase:
        # Tester une phrase spécifique
        predict(args.phrase, tokenizer, model, label_encoder)
    elif args.evaluate:
        # Évaluer sur dataset
        evaluate_on_dataset(tokenizer, model, label_encoder)
    elif args.interactive:
        # Mode interactif
        interactive_mode(tokenizer, model, label_encoder)
    else:
        # Par défaut: exemples de test
        test_examples(tokenizer, model, label_encoder)
        
        # Proposer le mode interactif
        print("\n💡 Utilisez --interactive pour tester vos propres phrases")
        print("   ou --evaluate pour évaluer sur le dataset de test")


if __name__ == '__main__':
    main()
