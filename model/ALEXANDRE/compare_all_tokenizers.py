"""
Script de comparaison : Compare tous les tokenizers sur le même dataset

Ce script charge les 4 approches et les compare sur un échantillon du dataset :
1. NLTK word_tokenize
2. RegexpTokenizer v1 (contractions non séparées)
3. RegexpTokenizer v2 amélioré ✅ (meilleure approche simple)
4. Tokenizers pré-entraînés (nécessitent fine-tuning)
"""

import sys
import os
from typing import List, Dict
import json

# Import des différentes méthodes de tokenisation
sys.path.append(os.path.dirname(__file__))

from script1_nltk_basic import tokenize_basic
from script2_regexp_basic import tokenize_regexp_basic
from script3_regexp_improved import tokenize_regexp_improved

# Essayer d'importer les tokenizers Hugging Face
try:
    from transformers import AutoTokenizer
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


def load_dataset_sample(input_path: str, output_path: str, limit: int = 10) -> List[Dict]:
    """
    Charge un échantillon du dataset
    
    Args:
        input_path: Chemin vers dataset_input.txt
        output_path: Chemin vers dataset_output.txt
        limit: Nombre de lignes à charger
    
    Returns:
        Liste de dictionnaires avec phrase, départ, arrivée
    """
    samples = []
    
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'r', encoding='utf-8') as f_out:
        
        count = 0
        for input_line, output_line in zip(f_in, f_out):
            if count >= limit:
                break
            
            # Parser les lignes
            input_parts = input_line.strip().split(',', 1)
            output_parts = output_line.strip().split(',')
            
            if len(input_parts) < 2 or len(output_parts) < 2:
                continue
            
            phrase = input_parts[1]
            departure = output_parts[1] if len(output_parts) > 1 else ""
            arrival = output_parts[2] if len(output_parts) > 2 else ""
            
            # Ignorer les phrases INVALID
            if departure == "INVALID" or not departure or not arrival:
                continue
            
            samples.append({
                "phrase": phrase,
                "departure": departure,
                "arrival": arrival
            })
            count += 1
    
    return samples


def align_tokens(tokens: List[str], departure: str, arrival: str, tokenize_func) -> List[str]:
    """
    Aligne les tokens avec les labels BIO
    """
    labels = ["O"] * len(tokens)
    
    # Tokeniser les entités avec la même fonction
    dep_tokens = tokenize_func(departure)
    arr_tokens = tokenize_func(arrival)
    
    # Normaliser pour comparaison
    tokens_lower = [t.lower() for t in tokens]
    dep_tokens_lower = [t.lower() for t in dep_tokens]
    arr_tokens_lower = [t.lower() for t in arr_tokens]
    
    # Chercher le départ
    for i in range(len(tokens_lower) - len(dep_tokens_lower) + 1):
        if tokens_lower[i:i+len(dep_tokens_lower)] == dep_tokens_lower:
            labels[i] = "B-DEP"
            for j in range(1, len(dep_tokens_lower)):
                labels[i+j] = "I-DEP"
            break
    
    # Chercher l'arrivée
    for i in range(len(tokens_lower) - len(arr_tokens_lower) + 1):
        if tokens_lower[i:i+len(arr_tokens_lower)] == arr_tokens_lower:
            labels[i] = "B-ARR"
            for j in range(1, len(arr_tokens_lower)):
                labels[i+j] = "I-ARR"
            break
    
    return labels


def compare_tokenizers(sample: Dict):
    """
    Compare tous les tokenizers sur un échantillon
    """
    phrase = sample["phrase"]
    departure = sample["departure"]
    arrival = sample["arrival"]
    
    print(f"\n{'='*80}")
    print(f"Phrase: \"{phrase}\"")
    print(f"Départ: {departure}")
    print(f"Arrivée: {arrival}")
    print('='*80)
    
    results = {}
    
    # Script 1 : Tokenisation basique
    tokens1 = tokenize_basic(phrase)
    labels1 = align_tokens(tokens1, departure, arrival, tokenize_basic)
    results["Script 1 (NLTK)"] = {
        "tokens": tokens1,
        "labels": labels1,
        "dep_detected": "B-DEP" in labels1,
        "arr_detected": "B-ARR" in labels1
    }
    
    # Script 2 : RegexpTokenizer basique
    tokens2 = tokenize_regexp_basic(phrase)
    labels2 = align_tokens(tokens2, departure, arrival, tokenize_regexp_basic)
    results["Script 2 (Regex v1)"] = {
        "tokens": tokens2,
        "labels": labels2,
        "dep_detected": "B-DEP" in labels2,
        "arr_detected": "B-ARR" in labels2
    }
    
    # Script 3 : RegexpTokenizer amélioré
    tokens3 = tokenize_regexp_improved(phrase)
    labels3 = align_tokens(tokens3, departure, arrival, tokenize_regexp_improved)
    results["Script 3 (Regex v2 ✅)"] = {
        "tokens": tokens3,
        "labels": labels3,
        "dep_detected": "B-DEP" in labels3,
        "arr_detected": "B-ARR" in labels3
    }
    
    # Afficher les résultats
    for method, data in results.items():
        print(f"\n{method}:")
        print(json.dumps({
            "tokens": data["tokens"],
            "labels": data["labels"]
        }, indent=2, ensure_ascii=False))
        
        dep_status = "✅" if data["dep_detected"] else "❌"
        arr_status = "✅" if data["arr_detected"] else "❌"
        print(f"{dep_status} Départ | {arr_status} Arrivée")
    
    # Script 4 : Tokenizers pré-entraînés (si disponible)
    if HF_AVAILABLE:
        print(f"\nScript 4 (Tokenizers pré-entraînés):")
        print("ℹ️  Ces tokenizers montrent uniquement la segmentation.")
        print("   Ils nécessitent un fine-tuning pour détecter départ/arrivée.")
        
        models = [
            "camembert-base",
            "flaubert/flaubert_base_cased",
            "dbmdz/bert-base-french-europeana-cased",
            "bert-base-multilingual-cased"
        ]
        
        for model_name in models:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                tokens = tokenizer.tokenize(phrase)
                print(f"\n  {model_name}:")
                print(f"    {tokens[:20]}{'...' if len(tokens) > 20 else ''}")
            except Exception as e:
                print(f"  ⚠️  {model_name}: Erreur ({str(e)[:50]})")


def compute_statistics(input_path: str, output_path: str, limit: int = 1000):
    """
    Calcule les statistiques de détection sur un échantillon plus large
    """
    print(f"\n{'='*80}")
    print(f"STATISTIQUES DE DÉTECTION (sur {limit} phrases)")
    print('='*80)
    
    stats = {
        "Script 1 (NLTK)": {"dep": 0, "arr": 0, "both": 0, "total": 0},
        "Script 2 (Regex v1)": {"dep": 0, "arr": 0, "both": 0, "total": 0},
        "Script 3 (Regex v2 ✅)": {"dep": 0, "arr": 0, "both": 0, "total": 0}
    }
    
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'r', encoding='utf-8') as f_out:
        
        count = 0
        for input_line, output_line in zip(f_in, f_out):
            if count >= limit:
                break
            
            # Parser les lignes
            input_parts = input_line.strip().split(',', 1)
            output_parts = output_line.strip().split(',')
            
            if len(input_parts) < 2 or len(output_parts) < 2:
                continue
            
            phrase = input_parts[1]
            departure = output_parts[1] if len(output_parts) > 1 else ""
            arrival = output_parts[2] if len(output_parts) > 2 else ""
            
            # Ignorer les phrases INVALID
            if departure == "INVALID" or not departure or not arrival:
                continue
            
            # Tester chaque méthode
            methods = [
                ("Script 1 (NLTK)", tokenize_basic),
                ("Script 2 (Regex v1)", tokenize_regexp_basic),
                ("Script 3 (Regex v2 ✅)", tokenize_regexp_improved)
            ]
            
            for method_name, tokenize_func in methods:
                tokens = tokenize_func(phrase)
                labels = align_tokens(tokens, departure, arrival, tokenize_func)
                
                stats[method_name]["total"] += 1
                if "B-DEP" in labels:
                    stats[method_name]["dep"] += 1
                if "B-ARR" in labels:
                    stats[method_name]["arr"] += 1
                if "B-DEP" in labels and "B-ARR" in labels:
                    stats[method_name]["both"] += 1
            
            count += 1
    
    # Afficher les statistiques
    print("\nMéthode                    | Départ  | Arrivée | Les deux | Taux global")
    print("-" * 80)
    
    for method_name, data in stats.items():
        total = data["total"]
        if total > 0:
            dep_pct = 100 * data["dep"] / total
            arr_pct = 100 * data["arr"] / total
            both_pct = 100 * data["both"] / total
            
            print(f"{method_name:26} | {dep_pct:5.1f}% | {arr_pct:6.1f}% | {both_pct:7.1f}% | {both_pct:5.1f}%")


def main():
    # Chemins vers le dataset
    dataset_dir = "../../dataset"
    input_path = os.path.join(dataset_dir, "dataset_input.txt")
    output_path = os.path.join(dataset_dir, "dataset_output.txt")
    
    # Vérifier que les fichiers existent
    if not os.path.exists(input_path) or not os.path.exists(output_path):
        print(f"❌ Fichiers de dataset non trouvés :")
        print(f"   {input_path}")
        print(f"   {output_path}")
        return
    
    print("=" * 80)
    print("COMPARAISON DE TOUS LES TOKENIZERS")
    print("=" * 80)
    print("\n📊 Ce script compare les 4 approches de tokenisation :")
    print("  1. Script 1 : NLTK word_tokenize")
    print("  2. Script 2 : RegexpTokenizer v1 (contractions non séparées)")
    print("  3. Script 3 : RegexpTokenizer v2 ✅ (meilleure approche simple)")
    print("  4. Script 4 : Tokenizers pré-entraînés (nécessitent fine-tuning)")
    
    if not HF_AVAILABLE:
        print("\n⚠️  transformers non installé - Script 4 désactivé")
        print("   Installation : pip install transformers")
    
    # Charger un échantillon du dataset
    print("\n📂 Chargement d'un échantillon du dataset...")
    samples = load_dataset_sample(input_path, output_path, limit=3)
    
    if not samples:
        print("❌ Aucun échantillon valide trouvé dans le dataset")
        return
    
    print(f"✅ {len(samples)} échantillons chargés\n")
    
    # Comparer les tokenizers sur chaque échantillon
    for i, sample in enumerate(samples, 1):
        print(f"\n{'#'*80}")
        print(f"# ÉCHANTILLON {i}/{len(samples)}")
        print(f"{'#'*80}")
        compare_tokenizers(sample)
    
    # Calculer les statistiques globales
    compute_statistics(input_path, output_path, limit=1000)
    
    # Conclusion
    print(f"\n{'='*80}")
    print("CONCLUSION")
    print('='*80)
    print("""
✅ MEILLEURE APPROCHE SIMPLE : Script 3 (Regex v2)
   • Regex : r"\\b(?:d'|l'|j'|qu'|n'|s'|t'|m'|c')\\b|\\b\\w+\\b"
   • Sépare correctement les contractions françaises
   • Gère bien les mots composés avec tirets
   • Excellente détection sans entraînement de modèle
   • Format BIO lisible et exploitable

⚠️  Script 1 (NLTK) et Script 2 (Regex v1) :
   • Problèmes avec les contractions et les noms composés
   • Détection moins fiable des entités

🤖 Script 4 (Tokenizers pré-entraînés) :
   • CamemBERT, FlauBERT, BERT français, BERT multilingue
   • Nécessitent un fine-tuning pour la détection d'entités
   • Voir poc_roberta_training.py pour l'entraînement d'un modèle
   • Plus performants après entraînement sur beaucoup de données

📈 RECOMMANDATIONS :
   1. Utiliser Script 3 pour une solution rapide et efficace
   2. Pour de meilleures performances, fine-tuner un modèle (Script 4)
   3. Augmenter le dataset avec des cas atypiques pour améliorer la robustesse
    """)


if __name__ == "__main__":
    main()
