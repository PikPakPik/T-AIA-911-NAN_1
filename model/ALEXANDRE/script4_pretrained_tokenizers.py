"""
Script 4 : Test des tokenizers pré-entraînés de Hugging Face

Ce script compare différents tokenizers optimisés pour le français :
- CamemBERT (camembert-base) : optimisé pour le français
- FlauBERT (flaubert/flaubert_base_cased) : français avec casse
- BERT français (dbmdz/bert-base-french-europeana-cased) : européen français
- BERT multilingue (bert-base-multilingual-cased) : multilingue avec casse

Chaque tokenizer a sa propre stratégie de segmentation et indicateurs de sous-mots.

Note: Le pipeline NER ne peut pas deviner départ/arrivée sans être entraîné sur les données.
Ce script montre seulement comment chaque tokenizer découpe les mots.
"""

from typing import List, Dict
import json

try:
    from transformers import AutoTokenizer
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("⚠️  transformers n'est pas installé. Installez avec: pip install transformers")


# Liste des modèles à tester
MODELS = [
    "camembert-base",
    "flaubert/flaubert_base_cased",
    "dbmdz/bert-base-french-europeana-cased",
    "bert-base-multilingual-cased"
]


def tokenize_with_model(text: str, model_name: str) -> List[str]:
    """
    Tokenise le texte avec un tokenizer pré-entraîné
    
    Args:
        text: Texte à tokeniser
        model_name: Nom du modèle Hugging Face
    
    Returns:
        Liste des tokens
    """
    if not HF_AVAILABLE:
        return []
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokens = tokenizer.tokenize(text)
        return tokens
    except Exception as e:
        print(f"Erreur avec {model_name}: {e}")
        return []


def display_tokenizer_comparison(phrase: str, departure: str, arrival: str):
    """
    Compare les différents tokenizers sur une phrase
    
    Args:
        phrase: Phrase à analyser
        departure: Ville de départ
        arrival: Ville d'arrivée
    """
    print(f"\nPhrase: \"{phrase}\"")
    print(f"Départ: {departure}")
    print(f"Arrivée: {arrival}")
    print("-" * 80)
    
    results = {
        "Phrase": phrase,
        "Départ": departure,
        "Arrivée": arrival,
        "Tokens": {}
    }
    
    for model_name in MODELS:
        print(f"\n📦 {model_name}:")
        tokens = tokenize_with_model(phrase, model_name)
        
        if tokens:
            results["Tokens"][model_name] = tokens
            print(f"   {tokens}")
            
            # Afficher les indicateurs spéciaux
            if "camembert" in model_name.lower():
                print("   ℹ️  '▁' indique le début d'un nouveau mot")
            elif "flaubert" in model_name.lower():
                print("   ℹ️  '</w>' indique la fin d'un mot")
            elif "bert" in model_name.lower() and "camembert" not in model_name.lower():
                print("   ℹ️  '##' indique une continuation du mot précédent")
    
    print("\n" + "=" * 80)
    print("RÉSUMÉ JSON:")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print("=" * 80)


def analyze_tokenizer_strategies():
    """
    Analyse détaillée des stratégies de tokenisation
    """
    print("\n" + "=" * 80)
    print("ANALYSE DES STRATÉGIES DE TOKENISATION")
    print("=" * 80)
    
    strategies = {
        "CamemBERT": {
            "indicateur": "▁ (underscore)",
            "signification": "Début d'un nouveau mot",
            "traitement_contractions": "Sépare 'd' et l'apostrophe",
            "optimisation": "Français, conserve les mots entiers"
        },
        "FlauBERT": {
            "indicateur": "</w>",
            "signification": "Fin d'un mot",
            "traitement_contractions": "Garde 'd'' ensemble",
            "optimisation": "Français avec casse préservée"
        },
        "BERT français": {
            "indicateur": "##",
            "signification": "Continuation du mot précédent (sous-mot)",
            "traitement_contractions": "Sépare en plusieurs tokens",
            "optimisation": "Corpus européen français, segmentation fine"
        },
        "BERT multilingue": {
            "indicateur": "##",
            "signification": "Continuation du mot précédent (sous-mot)",
            "traitement_contractions": "Segmentation très fine",
            "optimisation": "Flexibilité linguistique multilingue"
        }
    }
    
    for model, info in strategies.items():
        print(f"\n{model}:")
        for key, value in info.items():
            print(f"  • {key.replace('_', ' ').title()}: {value}")


def main():
    if not HF_AVAILABLE:
        print("\n❌ Ce script nécessite la bibliothèque 'transformers'")
        print("Installation: pip install transformers")
        return
    
    print("=" * 80)
    print("SCRIPT 4 : Test des tokenizers pré-entraînés")
    print("=" * 80)
    print("\n📚 Modèles testés:")
    for i, model in enumerate(MODELS, 1):
        print(f"  {i}. {model}")
    
    print("\n⚠️  IMPORTANT :")
    print("Ces tokenizers ne peuvent pas identifier départ/arrivée sans entraînement.")
    print("Ce script montre uniquement comment ils découpent les mots en tokens.")
    print("Pour la détection d'entités, il faut fine-tuner un modèle sur vos données.\n")
    
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
    
    for i, example in enumerate(test_examples, 1):
        print(f"\n{'='*80}")
        print(f"EXEMPLE {i}")
        print('='*80)
        display_tokenizer_comparison(
            example["phrase"],
            example["departure"],
            example["arrival"]
        )
    
    # Analyse des stratégies
    analyze_tokenizer_strategies()
    
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("""
Stratégies de segmentation observées :

1. CamemBERT et FlauBERT :
   ✓ Optimisés pour le français
   ✓ Tendance à conserver les mots plus entiers
   ✓ Meilleur pour préserver le sens des mots français

2. BERT français :
   • Segmentation fine avec ##
   • Bon compromis entre granularité et sens

3. BERT multilingue :
   • Segmentation très fine
   • Flexibilité linguistique mais perd certaines spécificités françaises

4. Traitement des tirets :
   • Généralement tokenisés comme tokens séparés
   • Peut segmenter les noms composés (Ailly-sur-Somme)

Pour la détection de départ/arrivée :
→ Il faut fine-tuner un de ces modèles sur votre dataset
→ Le script 3 (regex améliorée) reste une excellente alternative simple
→ Pour un modèle entraîné, voir poc_roberta_training.py
    """)


if __name__ == "__main__":
    main()
