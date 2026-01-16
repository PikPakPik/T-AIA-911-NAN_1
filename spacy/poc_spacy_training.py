import csv
import re
import unicodedata
from pathlib import Path
import spacy
from spacy.tokens import DocBin
import random
import spacy.cli.train
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse

# Charge les données de géolocalisation
with open("../dataset/geo.csv", "r", encoding="utf-8") as file:
    reader = csv.reader(file)
    locations = list(reader)

def remove_accents(text: str) -> str:
    """Remove accents from text."""
    nfkd = unicodedata.normalize('NFD', text)
    return ''.join([c for c in nfkd if not unicodedata.combining(c)])

def generate_variants(name: str) -> set[str]:
    variants = set()

    base = name.strip()

    saint_rules = [
        (r"\bSaint\b", ["Saint", "St", "St."]),
        (r"\bSainte\b", ["Sainte", "Ste", "Ste."]),
    ]

    def apply_rules(text):
        results = {text}
        for pattern, repls in saint_rules:
            new = set()
            for r in results:
                if re.search(pattern, r, flags=re.IGNORECASE):
                    for repl in repls:
                        new.add(re.sub(pattern, repl, r, flags=re.IGNORECASE))
                else:
                    new.add(r)
            results = new
        return results

    for v in apply_rules(base):
        variants.add(v)
        variants.add(v.replace("-", " "))
        variants.add(v.replace(" ", "-"))
        variants.add(v.replace("'", " "))
        variants.add(v.replace("'", ""))

        no_accent = remove_accents(v)
        variants.add(no_accent)
        variants.add(no_accent.replace("-", " "))
        variants.add(no_accent.replace(" ", "-"))
        variants.add(no_accent.replace("'", " "))
        variants.add(no_accent.replace("'", ""))

    final = set()
    for v in variants:
        final.add(v)
        final.add(v.lower())
        final.add(v.upper())
        final.add(v.capitalize())

    return final

TEMPLATES = [
    "Je veux aller de {a} à {b}",
    "Train entre {a} et {b}",
    "Voyage de {a} vers {b}",
    "Billet {a} {b}",
    "Comment aller de {a} à {b}?",
    "Quel train entre {a} et {b}?",
    "C'est combien pour aller de {a} à {b}?",
    "Je dois me rendre de {a} à {b}",
    "Je dois aller de {a} à {b} demain",
    "Un aller-retour {a}-{b} s'il vous plaît",
    "Un billet {a} {b} svp",
    "Départ de {a} vers {b} ce soir",
    "Je pars de {a} pour {b} lundi prochain",
    "Train de {a} à {b} demain matin",
    "Départ {a} arrivée {b} cet après-midi",
    "Trajet depuis {a} vers {b}",
    "Réserver un billet de {a} pour {b}",
    "Je souhaite me rendre à {b} depuis {a}",
    "Je veux aller à {b} en partant de {a}",
    "Aller à {b} au départ de {a}",
    "Direction {b} depuis {a}",
    "J'ai besoin d'aller de {a} à {b}",
    "Faut que j'aille de {a} à {b}",
    "Cherche train {a} {b}",
    "Dispo {a}-{b}?",
    "Réservation au départ de {a} direction {b}",
    "Billet de train de {a} à destination de {b}",
    "Partir de {a} pour rejoindre {b}",
    "Trajet en train de {a} jusqu'à {b}",
]

def generate_annotated_sentences(locations, n=15000):
    """Génère des phrases annotées pour l'entraînement NER"""
    samples = []

    for _ in range(n):
        (name1, label1), (name2, label2) = random.sample(locations, 2)

        v1 = random.choice(list(generate_variants(name1)))
        v2 = random.choice(list(generate_variants(name2)))

        sentence = random.choice(TEMPLATES).format(a=v1, b=v2)

        start1 = sentence.index(v1)
        end1 = start1 + len(v1)
        start2 = sentence.index(v2)
        end2 = start2 + len(v2)

        samples.append((
            sentence,
            [
                (start1, end1, "LOC"),
                (start2, end2, "LOC"),
            ]
        ))

    return samples

def convert_to_spacy_format(samples, nlp, output_path):
    """Convertit les échantillons annotés au format SpaCy DocBin"""
    doc_bin = DocBin()
    
    for text, entities in samples:
        # Crée un Doc SpaCy à partir du texte
        doc = nlp.make_doc(text)
        
        # Crée les spans d'entités
        ents = []
        for start, end, label in entities:
            # Vérifie que les indices sont valides
            if start < len(text) and end <= len(text) and start < end:
                span = doc.char_span(start, end, label=label, alignment_mode="contract")
                if span is not None:
                    ents.append(span)
        
        # Assigne les entités au document
        doc.ents = ents
        doc_bin.add(doc)
    
    # Sauvegarde le DocBin
    doc_bin.to_disk(output_path)
    print(f"Sauvegardé {len(samples)} échantillons dans {output_path}")

def evaluate_model_and_confusion_matrix(model_path, test_data_path, output_dir):
    """Évalue le modèle et génère une matrice de confusion"""
    print(f"\n{'='*60}")
    print("Évaluation du modèle et génération de la matrice de confusion")
    print(f"{'='*60}\n")
    
    # Charge le modèle entraîné
    print(f"Chargement du modèle depuis {model_path}...")
    nlp = spacy.load(model_path)
    
    # Charge les données de test
    print(f"Chargement des données de test depuis {test_data_path}...")
    doc_bin = DocBin().from_disk(test_data_path)
    test_docs = list(doc_bin.get_docs(nlp.vocab))
    print(f"Nombre de documents de test: {len(test_docs)}\n")
    
    # Prépare les listes pour les labels réels et prédits (au niveau des tokens)
    true_labels = []
    pred_labels = []
    
    # Prépare aussi pour l'évaluation au niveau des entités
    true_entities = []
    pred_entities = []
    
    # Évalue chaque document
    for doc in test_docs:
        # Prédit avec le modèle
        pred_doc = nlp(doc.text)
        
        # Extrait les entités réelles et prédites
        true_ents = {(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents}
        pred_ents = {(ent.start_char, ent.end_char, ent.label_) for ent in pred_doc.ents}
        
        true_entities.append(true_ents)
        pred_entities.append(pred_ents)
        
        # Pour la matrice de confusion au niveau des tokens
        # Crée un mapping caractère -> label pour les entités réelles
        char_to_label = {}
        for ent in doc.ents:
            for i in range(ent.start_char, ent.end_char):
                char_to_label[i] = ent.label_
        
        # Parcourt chaque token dans le document original
        for token in doc:
            # Détermine le label réel
            if token.idx in char_to_label:
                true_label = char_to_label[token.idx]
            else:
                true_label = "O"
            
            # Détermine le label prédit
            pred_label = "O"
            for ent in pred_doc.ents:
                if ent.start_char <= token.idx < ent.end_char:
                    pred_label = ent.label_
                    break
            
            true_labels.append(true_label)
            pred_labels.append(pred_label)
    
    # Génère la matrice de confusion au niveau des tokens
    unique_labels = sorted(set(true_labels + pred_labels))
    cm = confusion_matrix(true_labels, pred_labels, labels=unique_labels)
    
    # Affiche les métriques
    print("📊 Métriques de classification:")
    print("-" * 60)
    report = classification_report(true_labels, pred_labels, labels=unique_labels, zero_division=0)
    print(report)
    
    # Calcule les métriques au niveau des entités
    print("\n📈 Métriques au niveau des entités:")
    print("-" * 60)
    correct = 0
    total_true = 0
    total_pred = 0
    
    for true_ents, pred_ents in zip(true_entities, pred_entities):
        total_true += len(true_ents)
        total_pred += len(pred_ents)
        correct += len(true_ents & pred_ents)
    
    precision = correct / total_pred if total_pred > 0 else 0
    recall = correct / total_true if total_true > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Précision: {precision:.4f} ({correct}/{total_pred})")
    print(f"Rappel: {recall:.4f} ({correct}/{total_true})")
    print(f"F1-score: {f1:.4f}")
    
    # Crée et sauvegarde la matrice de confusion
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=unique_labels, yticklabels=unique_labels,
                cbar_kws={'label': 'Nombre de tokens'})
    plt.title('Matrice de confusion - Modèle SpaCy NER', fontsize=16, fontweight='bold')
    plt.ylabel('Labels réels', fontsize=12)
    plt.xlabel('Labels prédits', fontsize=12)
    plt.tight_layout()
    
    confusion_matrix_path = output_dir / "confusion_matrix.png"
    plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
    print(f"\nMatrice de confusion sauvegardée dans: {confusion_matrix_path}")
    
    # Crée aussi une matrice de confusion normalisée (pourcentages)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # Remplace les NaN par 0
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=unique_labels, yticklabels=unique_labels,
                cbar_kws={'label': 'Pourcentage'})
    plt.title('Matrice de confusion normalisée - Modèle SpaCy NER', fontsize=16, fontweight='bold')
    plt.ylabel('Labels réels', fontsize=12)
    plt.xlabel('Labels prédits', fontsize=12)
    plt.tight_layout()
    
    confusion_matrix_norm_path = output_dir / "confusion_matrix_normalized.png"
    plt.savefig(confusion_matrix_norm_path, dpi=300, bbox_inches='tight')
    print(f"Matrice de confusion normalisée sauvegardée dans: {confusion_matrix_norm_path}")
    
    print(f"\n{'='*60}\n")

def parse_args():
    parser = argparse.ArgumentParser(description="Train SpaCy NER with optional base model")
    parser.add_argument(
        "--model",
        choices=[
            "fr_core_news_md",
            "fr_core_news_lg"
        ],
        default="fr_core_news_lg",
        help="SpaCy model to load",
    )
    return parser.parse_args()

def main():
    # Configuration
    args = parse_args()
    base_model = args.model
    output_dir = Path(f"./finetuned_output_{base_model}")
    output_dir.mkdir(exist_ok=True)
    num_samples = 15000
    
    print(f"Chargement du modèle de base: {base_model}...")
    nlp = spacy.load(base_model)
    
    print(f"Génération de {num_samples} échantillons d'entraînement...")
    samples = generate_annotated_sentences(locations, n=num_samples)
    random.shuffle(samples)
    
    # Division train/dev/test (même logique que Roberta)
    split_train = int(len(samples) * 0.8)
    split_dev = int(len(samples) * 0.9)
    
    train_samples = samples[:split_train]
    dev_samples = samples[split_train:split_dev]
    test_samples = samples[split_dev:]
    
    print(f"Train: {len(train_samples)}, Dev: {len(dev_samples)}, Test: {len(test_samples)}")
    
    # Conversion au format SpaCy
    print("Conversion des données au format SpaCy...")
    convert_to_spacy_format(train_samples, nlp, output_dir / "train.spacy")
    convert_to_spacy_format(dev_samples, nlp, output_dir / "dev.spacy")
    convert_to_spacy_format(test_samples, nlp, output_dir / "test.spacy")
    
    # Entraînement
    print("Démarrage de l'entraînement...")
    spacy.cli.train.train(
        config_path=f"finetune_config_{base_model}.cfg",
        output_path=str(output_dir),
        overrides={},
    )
    
    print("Entraînement terminé!")
    
    # Évaluation avec matrice de confusion
    model_best_path = output_dir / "model-best"
    test_data_path = output_dir / "test.spacy"
    
    if model_best_path.exists() and test_data_path.exists():
        evaluate_model_and_confusion_matrix(
            str(model_best_path),
            str(test_data_path),
            output_dir
        )
    else:
        print(f"\nImpossible de générer la matrice de confusion:")
        if not model_best_path.exists():
            print(f"   - Modèle non trouvé: {model_best_path}")
        if not test_data_path.exists():
            print(f"   - Données de test non trouvées: {test_data_path}")

if __name__ == "__main__":
    random.seed(42)
    main()