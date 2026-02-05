r"""
Script 3 : Tokenisation avec Regex améliorée
Regex: r"\b(?:d'|l'|j'|qu'|n'|s'|t'|m'|c')\b|\b\w+\b"

✅ SOLUTION OPTIMALE :
- Sépare les contractions en deux tokens distincts ("d'" et "Alençon")
- Gère correctement les mots composés avec tirets
- Très bon résultat avec une approche simple sans entraînement
- Format BIO lisible pour l'évaluation humaine
"""

import re
from typing import List, Dict, Tuple
import json


def tokenize_regexp_improved(text: str) -> List[str]:
    r"""
    Tokenisation avec regex améliorée qui sépare les contractions
    Regex: r"\b(?:d'|l'|j'|qu'|n'|s'|t'|m'|c')\b|\b\w+\b"
    
    Cette regex capture :
    1. Les contractions françaises comme tokens séparés
    2. Les mots complets
    3. Gère les tirets correctement
    """
    pattern = r"\b(?:d'|l'|j'|qu'|n'|s'|t'|m'|c')\b|\b\w+\b"
    tokens = re.findall(pattern, text, re.IGNORECASE)
    return tokens


def tokenize_regexp_improved_with_spans(text: str) -> List[Tuple[str, int, int]]:
    """
    Tokenisation comme tokenize_regexp_improved, avec positions caractères (start, end).
    Retourne une liste de (token, start, end) pour aligner les labels sur BERT (offset_mapping).
    """
    pattern = r"\b(?:d'|l'|j'|qu'|n'|s'|t'|m'|c')\b|\b\w+\b"
    return [(m.group(0), m.start(), m.end()) for m in re.finditer(pattern, text, re.IGNORECASE)]


def get_entity_spans_from_script3(phrase: str, departure: str, arrival: str) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Utilise la tokenisation script 3 + alignement BIO pour obtenir les spans (début, fin)
    des entités Départ et Arrivée en indices caractères.
    Retourne ((dep_start, dep_end), (arr_start, arr_end)), ou (-1,-1) si non trouvé.
    """
    token_spans = tokenize_regexp_improved_with_spans(phrase)
    if not token_spans:
        return (-1, -1), (-1, -1)
    tokens = [t[0] for t in token_spans]
    labels = align_tokens_with_entities(tokens, departure, arrival)
    dep_start = dep_end = -1
    arr_start = arr_end = -1
    for (_, s, e), lab in zip(token_spans, labels):
        if lab in ("B-DEP", "I-DEP"):
            if dep_start < 0:
                dep_start = s
            dep_end = e
        elif lab in ("B-ARR", "I-ARR"):
            if arr_start < 0:
                arr_start = s
            arr_end = e
    return (dep_start, dep_end), (arr_start, arr_end)


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
    dep_tokens = tokenize_regexp_improved(departure)
    arr_tokens = tokenize_regexp_improved(arrival)
    
    # Normaliser pour la comparaison (insensible à la casse)
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
    correct_dep = 0
    correct_arr = 0
    total = 0
    
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
            tokens = tokenize_regexp_improved(phrase)
            
            # Alignement avec les entités
            labels = align_tokens_with_entities(tokens, departure, arrival)
            
            # Statistiques
            total += 1
            if "B-DEP" in labels:
                correct_dep += 1
            if "B-ARR" in labels:
                correct_arr += 1
            
            results.append({
                "line": line_num,
                "phrase": phrase,
                "tokens": tokens,
                "labels": labels,
                "departure": departure,
                "arrival": arrival
            })
    
    if total > 0:
        print(f"\n📊 STATISTIQUES (sur {total} phrases valides):")
        print(f"Départs détectés: {correct_dep}/{total} ({100*correct_dep/total:.1f}%)")
        print(f"Arrivées détectées: {correct_arr}/{total} ({100*correct_arr/total:.1f}%)")
    
    return results


def export_to_json(results: List[Dict], output_path: str = "output_tokens.json"):
    """
    Exporte les résultats au format JSON
    
    Args:
        results: Liste des résultats de tokenisation
        output_path: Chemin du fichier de sortie
    """
    json_output = []
    
    for result in results:
        json_output.append({
            "tokens": result["tokens"],
            "labels": result["labels"]
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Résultats exportés vers : {output_path}")
    print(f"   {len(json_output)} phrases sauvegardées")


def main():
    import os
    
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
    
    print("=" * 80)
    print("SCRIPT 3 : Tokenisation avec Regex AMÉLIORÉE ✅")
    print("Regex: r\"\\b(?:d'|l'|j'|qu'|n'|s'|t'|m'|c')\\b|\\b\\w+\\b\"")
    print("=" * 80)
    print("\n✅ AVANTAGES :")
    print("- Sépare les contractions en deux tokens distincts")
    print("- Gère correctement les mots composés avec tirets")
    print("- Excellente détection des entités")
    print("- Solution simple et performante sans entraînement de modèle\n")
    
    print("\n📝 EXEMPLES DE TEST:")
    print("-" * 80)
    
    for i, example in enumerate(test_examples, 1):
        tokens = tokenize_regexp_improved(example["phrase"])
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
        dep_detected = "B-DEP" in labels
        arr_detected = "B-ARR" in labels
        
        print(f"{'✅' if dep_detected else '❌'} Départ {'détecté' if dep_detected else 'NON détecté'}")
        print(f"{'✅' if arr_detected else '❌'} Arrivée {'détectée' if arr_detected else 'NON détectée'}")
    
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
    
    # Afficher les statistiques globales (déjà affiché par process_file)
    print("\n" + "=" * 80)
    
    # Export en JSON
    json_output_path = os.path.join(os.path.dirname(__file__), "script3_output.json")
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
