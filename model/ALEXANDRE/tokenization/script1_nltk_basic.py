"""
Script 1 : Tokenisation avec NLTK word_tokenize
Problème : NLTK word_tokenize ne gère pas bien les contractions et noms composés français
"""

import os
import nltk
from nltk.tokenize import word_tokenize
import json
from typing import List, Dict

# Télécharger les ressources NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("📥 Téléchargement des ressources NLTK...")
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)


def tokenize_basic(text: str) -> List[str]:
    """
    Tokenisation avec NLTK word_tokenize.
    Export pour compare_all_tokenizers.py et usage interne.
    """
    return word_tokenize(text, language='french')


def annotate_tokens(depart: str, arrivee: str, tokens: List[str]) -> List[str]:
    """
    Annote les tokens avec les labels BIO
    
    Args:
        depart: Ville de départ
        arrivee: Ville d'arrivée
        tokens: Liste des tokens
    
    Returns:
        Liste des labels BIO
    """
    labels = ['O'] * len(tokens)
    
    # Tokeniser les entités de la même manière
    depart_tokens = word_tokenize(depart.lower(), language='french') if depart else []
    arrivee_tokens = word_tokenize(arrivee.lower(), language='french') if arrivee else []
    
    # Normaliser les tokens pour la comparaison
    tokens_lower = [t.lower() for t in tokens]
    
    def annotate_entity(entity_tokens: List[str], tag_prefix: str):
        """Annote une entité dans la liste de tokens"""
        if not entity_tokens:
            return
        
        start_index = 0
        while start_index < len(tokens_lower):
            try:
                # Chercher le premier token de l'entité
                start_index = tokens_lower.index(entity_tokens[0], start_index)
                
                # Vérifier si toute la séquence correspond
                if tokens_lower[start_index:start_index + len(entity_tokens)] == entity_tokens:
                    labels[start_index] = f"B-{tag_prefix}"
                    for i in range(1, len(entity_tokens)):
                        labels[start_index + i] = f"I-{tag_prefix}"
                    start_index += len(entity_tokens)
                else:
                    start_index += 1
            except ValueError:
                break
    
    # Annoter départ et arrivée
    annotate_entity(depart_tokens, 'DEP')
    annotate_entity(arrivee_tokens, 'ARR')
    
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
            
            # Tokenisation avec NLTK
            tokens = word_tokenize(phrase, language='french')
            
            # Annotation
            labels = annotate_tokens(depart, arrivee, tokens)
            
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
    print("=" * 80)
    print("SCRIPT 1 : Tokenisation avec NLTK word_tokenize")
    print("=" * 80)
    print("\n⚠️  PROBLÈMES ATTENDUS :")
    print("- NLTK word_tokenize sépare les contractions (\"d'\" + \"Alençon\")")
    print("- Les noms composés avec tirets sont tokenisés en plusieurs parties")
    print("- Comparaison insensible à la casse mais tokenisation moins optimale\n")
    
    # Exemples de test
    test_examples = [
        {
            "phrase": "j'aimerais voyager de Ailly-sur-Somme à Aix-les-Bains le Revard",
            "depart": "Ailly-sur-Somme",
            "arrivee": "Aix-les-Bains le Revard"
        },
        {
            "phrase": "quels trains voyagent d'Alençon à Corbeil-Essonnes",
            "depart": "Alençon",
            "arrivee": "Corbeil-Essonnes"
        }
    ]
    
    print("📝 EXEMPLES DE TEST:")
    print("-" * 80)
    
    for i, example in enumerate(test_examples, 1):
        tokens = word_tokenize(example["phrase"], language='french')
        labels = annotate_tokens(example["depart"], example["arrivee"], tokens)
        
        print(f"\nExemple {i}:")
        print(f"Phrase: {example['phrase']}")
        print(f"Départ: {example['depart']}")
        print(f"Arrivée: {example['arrivee']}")
        print(json.dumps({
            "tokens": tokens,
            "labels": labels
        }, indent=2, ensure_ascii=False))
        
        dep_detected = "B-DEP" in labels
        arr_detected = "B-ARR" in labels
        print(f"{'✅' if dep_detected else '❌'} Départ | {'✅' if arr_detected else '❌'} Arrivée")
    
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
        print(f"  Départ: {data['depart']}")
        print(f"  Arrivée: {data['arrivee']}")
        print(f"  Tokens: {data['tokens'][:10]}{'...' if len(data['tokens']) > 10 else ''}")
        print(f"  Labels: {data['labels'][:10]}{'...' if len(data['labels']) > 10 else ''}")
        
        dep_detected = "B-DEP" in data['labels']
        arr_detected = "B-ARR" in data['labels']
        print(f"  {'✅' if dep_detected else '❌'} Départ | {'✅' if arr_detected else '❌'} Arrivée")
    
    # Export JSON
    json_file_path = os.path.join(os.path.dirname(__file__), 'script1_nltk_output.json')
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
