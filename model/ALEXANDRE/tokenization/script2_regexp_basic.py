r"""
Script 2 : Tokenisation avec RegexpTokenizer personnalisé
Regex: r"\b(?:d'|l'|j'|qu'|n'|s'|t'|m'|c')?\w+\b"

Problème : 
- Les contractions ne sont pas séparées (ex: "d'Alençon" est un seul token)
- Les mots composés avec tirets sont tokenisés en plusieurs parties
- Erreur d'annotation car Ailly-sur-Somme devient ["Ailly", "sur", "Somme"]
"""

import re
from typing import List, Dict
import json


def tokenize_regexp_basic(text: str) -> List[str]:
    r"""
    Tokenisation avec regex qui capture les contractions mais ne les sépare pas
    Regex: r"\b(?:d'|l'|j'|qu'|n'|s'|t'|m'|c')?\w+\b"
    """
    pattern = r"\b(?:d'|l'|j'|qu'|n'|s'|t'|m'|c')?\w+\b"
    return re.findall(pattern, text.lower())


def align_tokens_with_entities(
    tokens: List[str],
    departure: str,
    arrival: str
) -> List[str]:
    """
    Aligne les tokens avec les labels BIO (Begin, Inside, Outside)
    
    Args:
        tokens: Liste des tokens de la phrase
        departure: Ville de départ
        arrival: Ville d'arrivée
    
    Returns:
        Liste des labels BIO pour chaque token
    """
    labels = ["O"] * len(tokens)
    
    # Tokeniser les entités avec la même méthode
    dep_tokens = tokenize_regexp_basic(departure)
    arr_tokens = tokenize_regexp_basic(arrival)
    
    # Chercher le départ
    for i in range(len(tokens) - len(dep_tokens) + 1):
        if tokens[i:i+len(dep_tokens)] == dep_tokens:
            labels[i] = "B-DEP"
            for j in range(1, len(dep_tokens)):
                labels[i+j] = "I-DEP"
            break
    
    # Chercher l'arrivée
    for i in range(len(tokens) - len(arr_tokens) + 1):
        if tokens[i:i+len(arr_tokens)] == arr_tokens:
            labels[i] = "B-ARR"
            for j in range(1, len(arr_tokens)):
                labels[i+j] = "I-ARR"
            break
    
    return labels


def process_file(input_path: str, output_path: str, limit: int = None) -> List[Dict]:
    """
    Traite les fichiers dataset_input.txt et dataset_output.txt
    
    Args:
        input_path: Chemin vers dataset_input.txt
        output_path: Chemin vers dataset_output.txt
        limit: Nombre maximum de lignes à traiter (None = toutes)
    
    Returns:
        Liste de dictionnaires avec tokens et labels
    """
    results = []
    
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'r', encoding='utf-8') as f_out:
        
        for line_num, (input_line, output_line) in enumerate(zip(f_in, f_out), 1):
            if limit and line_num > limit:
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
            
            # Tokenisation
            tokens = tokenize_regexp_basic(phrase)
            
            # Alignement avec les entités
            labels = align_tokens_with_entities(tokens, departure, arrival)
            
            results.append({
                "line": line_num,
                "phrase": phrase,
                "tokens": tokens,
                "labels": labels,
                "departure": departure,
                "arrival": arrival
            })
    
    return results


def export_to_json(annotated_data: List[Dict], output_path: str):
    """
    Exporte les résultats au format JSON
    
    Args:
        annotated_data: Liste des données annotées
        output_path: Chemin du fichier de sortie
    """
    json_data = []
    for data in annotated_data:
        json_data.append({
            'tokens': data['tokens'],
            'labels': data['labels']
        })
    
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(json_data, json_file, ensure_ascii=False, indent=2)
    
    print(f"\n💾 JSON exporté vers : {output_path}")
    print(f"   {len(json_data)} phrases sauvegardées")


def main():
    import os
    
    print("=" * 80)
    print("SCRIPT 2 : Tokenisation avec RegexpTokenizer")
    print("Regex: r\"\\b(?:d'|l'|j'|qu'|n'|s'|t'|m'|c')?\\w+\\b\"")
    print("=" * 80)
    print("\n⚠️  PROBLÈMES ATTENDUS :")
    print("- Les contractions ne sont pas séparées (d'Alençon reste un token)")
    print("- Les mots composés avec tirets sont découpés (Ailly-sur-Somme → 3 tokens)")
    print("- Erreur d'annotation pour les entités composées\n")
    
    # Exemples de test
    test_examples = [
        {
            "phrase": "j'aimerais voyager de Ailly-sur-Somme à Aix-les-Bains le Revard",
            "departure": "Ailly-sur-Somme",
            "arrival": "Aix-les-Bains le Revard"
        },
        {
            "phrase": "quels trains voyagent d'Alençon à Corbeil-Essonnes",
            "departure": "Alençon",
            "arrival": "Corbeil-Essonnes"
        }
    ]
    
    print("\n📝 EXEMPLES DE TEST:")
    print("-" * 80)
    
    for i, example in enumerate(test_examples, 1):
        tokens = tokenize_regexp_basic(example["phrase"])
        labels = align_tokens_with_entities(
            tokens,
            example["departure"],
            example["arrival"]
        )
        
        print(f"\nExemple {i}:")
        print(f"Phrase: {example['phrase']}")
        print(f"Départ attendu: {example['departure']}")
        print(f"Arrivée attendue: {example['arrival']}")
        print(json.dumps({
            "tokens": tokens,
            "labels": labels
        }, indent=2, ensure_ascii=False))
        
        # Vérifier si les entités ont été détectées
        if "B-DEP" not in labels:
            print("❌ DÉPART NON DÉTECTÉ ou MAL DÉTECTÉ")
        else:
            print("✓ Départ détecté")
        
        if "B-ARR" not in labels:
            print("❌ ARRIVÉE NON DÉTECTÉE ou MAL DÉTECTÉE")
        else:
            print("✓ Arrivée détectée")
    
    # Traiter le dataset
    print("\n" + "=" * 80)
    print("📂 TRAITEMENT DU DATASET")
    print("=" * 80)
    
    dataset_dir = "../../dataset"
    input_path = os.path.join(os.path.dirname(__file__), dataset_dir, "dataset_input.txt")
    output_path = os.path.join(os.path.dirname(__file__), dataset_dir, "dataset_output.txt")
    
    if not os.path.exists(input_path) or not os.path.exists(output_path):
        print(f"\n⚠️  Fichiers de dataset non trouvés :")
        print(f"   {input_path}")
        print(f"   {output_path}")
        print("\n💡 Assurez-vous d'être dans le bon répertoire ou que les fichiers existent.")
        return
    
    print(f"\n📊 Traitement de 1000 phrases du dataset...")
    results = process_file(input_path, output_path, limit=1000)
    
    print(f"\n✅ {len(results)} phrases traitées\n")
    
    print("📋 Exemples de phrases traitées:")
    print("-" * 80)
    
    # Afficher quelques exemples du dataset
    for i, result in enumerate(results[:5], 1):
        print(f"\nPhrase {result['line']}:")
        print(f"  Texte: {result['phrase']}")
        print(f"  Départ: {result['departure']}")
        print(f"  Arrivée: {result['arrival']}")
        print(f"  Tokens: {result['tokens'][:10]}{'...' if len(result['tokens']) > 10 else ''}")
        print(f"  Labels: {result['labels'][:10]}{'...' if len(result['labels']) > 10 else ''}")
        
        dep_detected = "B-DEP" in result['labels']
        arr_detected = "B-ARR" in result['labels']
        print(f"  {'✅' if dep_detected else '❌'} Départ | {'✅' if arr_detected else '❌'} Arrivée")
    
    # Statistiques globales
    correct_dep = sum(1 for r in results if "B-DEP" in r['labels'])
    correct_arr = sum(1 for r in results if "B-ARR" in r['labels'])
    total = len(results)
    
    print(f"\n{'='*80}")
    print(f"📊 STATISTIQUES (sur les {total} phrases traitées):")
    print(f"Départs détectés: {correct_dep}/{total} ({100*correct_dep/total:.1f}%)")
    print(f"Arrivées détectées: {correct_arr}/{total} ({100*correct_arr/total:.1f}%)")
    print("=" * 80)
    
    # Export en JSON
    import os
    json_output_path = os.path.join(os.path.dirname(__file__), "script2_output.json")
    export_to_json(results, json_output_path)
    
    # Afficher un extrait du JSON
    print("\n📄 EXTRAIT DU JSON GÉNÉRÉ:")
    print("-" * 80)
    for result in results[:5]:
        print(json.dumps({
            "tokens": result["tokens"],
            "labels": result["labels"]
        }, ensure_ascii=False) + ",")


if __name__ == "__main__":
    main()
