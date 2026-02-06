r"""
Script 2 : RegexpTokenizer personnalisé pour les contractions françaises
REGEX : r"\b(?:d'|l'|j'|qu'|n'|s'|t'|m'|c')?\w+\b"

Problèmes :
- Contractions non séparées (ex: "d'Alençon" est un seul token)
- Noms composés avec tirets sont découpés en plusieurs tokens
"""

import re
import os
from typing import List, Dict
import json


class RegexpTokenizerV1:
    r"""Tokenizer avec regex pour gérer les contractions françaises (version 1)"""
    
    def __init__(self):
        # Regex qui capture les contractions mais ne les sépare pas
        self.pattern = re.compile(r"\b(?:d'|l'|j'|qu'|n'|s'|t'|m'|c')?\w+\b")
    
    def tokenize(self, text: str) -> List[str]:
        r"""Tokenise le texte selon le pattern regex"""
        return self.pattern.findall(text.lower())


def align_tokens_with_entities(
    tokens: List[str],
    departure: str,
    arrival: str
) -> List[str]:
    """Aligne les tokens avec les labels BIO"""
    labels = ["O"] * len(tokens)
    
    # Normaliser et tokeniser les entités
    tokenizer = RegexpTokenizerV1()
    dep_tokens = tokenizer.tokenize(departure)
    arr_tokens = tokenizer.tokenize(arrival)
    
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


def process_dataset(input_path: str, output_path: str, limit: int = None) -> List[Dict]:
    """
    Traite les fichiers dataset_input.txt et dataset_output.txt
    
    Args:
        input_path: Chemin vers dataset_input.txt
        output_path: Chemin vers dataset_output.txt
        limit: Nombre maximum de lignes à traiter (None = toutes)
    
    Returns:
        Liste de dictionnaires avec tokens et labels
    """
    if not os.path.exists(input_path) or not os.path.exists(output_path):
        print(f"❌ Fichiers non trouvés:")
        print(f"   {input_path}")
        print(f"   {output_path}")
        return []
    
    tokenizer = RegexpTokenizerV1()
    annotated_data = []
    count = 0
    correct_dep = 0
    correct_arr = 0
    
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'r', encoding='utf-8') as f_out:
        
        for line_num, (input_line, output_line) in enumerate(zip(f_in, f_out), 1):
            if limit and count >= limit:
                break
            
            # Parser les lignes
            input_parts = input_line.strip().split(',', 1)
            output_parts = output_line.strip().split(',')
            
            if len(input_parts) < 2 or len(output_parts) < 2:
                continue
            
            phrase = input_parts[1]
            depart = output_parts[1] if len(output_parts) > 1 else ""
            arrivee = output_parts[2] if len(output_parts) > 2 else ""
            
            # Ignorer les phrases INVALID
            if depart == "INVALID" or not depart or not arrivee:
                continue
            
            # Tokenisation
            tokens = tokenizer.tokenize(phrase)
            
            # Annotation
            labels = align_tokens_with_entities(tokens, depart, arrivee)
            
            # Statistiques
            if "B-DEP" in labels:
                correct_dep += 1
            if "B-ARR" in labels:
                correct_arr += 1
            
            annotated_data.append({
                "line": line_num,
                "phrase": phrase,
                "tokens": tokens,
                "labels": labels,
                "depart": depart,
                "arrivee": arrivee
            })
            
            count += 1
    
    # Afficher les statistiques
    if count > 0:
        print(f"\n📊 STATISTIQUES (sur {count} phrases valides):")
        print(f"Départs détectés: {correct_dep}/{count} ({100*correct_dep/count:.1f}%)")
        print(f"Arrivées détectées: {correct_arr}/{count} ({100*correct_arr/count:.1f}%)")
    
    return annotated_data


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
    
    tokenizer = RegexpTokenizerV1()
    
    print("=" * 80)
    print("SCRIPT 2 : RegexpTokenizer v1")
    print("=" * 80)
    print(f"REGEX : {tokenizer.pattern.pattern}")
    print("\n⚠️  PROBLÈMES ATTENDUS :")
    print("- Contractions comme \"d'Alençon\" restent en un seul token")
    print("- Les noms composés sont sur-découpés (Ailly-sur-Somme -> 3 tokens)")
    print("- Erreurs d'annotation possibles\n")
    
    for i, example in enumerate(test_examples, 1):
        tokens = tokenizer.tokenize(example["phrase"])
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
        
        # Analyse des problèmes
        if "d'" in ' '.join(tokens):
            print("⚠️  Contraction non séparée détectée")
        if "B-DEP" not in labels:
            print("❌ DÉPART NON DÉTECTÉ")
        if "B-ARR" not in labels:
            print("❌ ARRIVÉE NON DÉTECTÉE")
    
    # Traiter le dataset
    print("\n" + "=" * 80)
    print("📂 TRAITEMENT DU DATASET")
    print("=" * 80)
    
    # Chemins vers les fichiers
    dataset_dir = "../../dataset"
    input_path = os.path.join(os.path.dirname(__file__), dataset_dir, "dataset_input.txt")
    output_path = os.path.join(os.path.dirname(__file__), dataset_dir, "dataset_output.txt")
    
    print(f"\n📊 Traitement de 1000 phrases du dataset...")
    annotated_dataset = process_dataset(input_path, output_path, limit=1000)
    
    if not annotated_dataset:
        return
    
    print(f"\n✅ {len(annotated_dataset)} phrases traitées")
    
    # Afficher quelques exemples
    print("\n📋 Exemples de phrases traitées:")
    print("-" * 80)
    for i, data in enumerate(annotated_dataset[:5], 1):
        print(f"\nPhrase {data['line']}:")
        print(f"  Texte: {data['phrase']}")
        print(f"  Tokens: {data['tokens'][:10]}{'...' if len(data['tokens']) > 10 else ''}")
        print(f"  Labels: {data['labels'][:10]}{'...' if len(data['labels']) > 10 else ''}")
        
        dep_detected = "B-DEP" in data['labels']
        arr_detected = "B-ARR" in data['labels']
        print(f"  {'✅' if dep_detected else '❌'} Départ | {'✅' if arr_detected else '❌'} Arrivée")
    
    # Export JSON
    json_file_path = os.path.join(os.path.dirname(__file__), 'script2_regex_v1_output.json')
    export_to_json(annotated_dataset, json_file_path)
    
    # Afficher un extrait du JSON
    print("\n📄 EXTRAIT DU JSON GÉNÉRÉ:")
    print("-" * 80)
    for data in annotated_dataset[:5]:
        print(json.dumps({
            "tokens": data["tokens"],
            "labels": data["labels"]
        }, ensure_ascii=False) + ",")


if __name__ == "__main__":
    main()
