"""
Utilitaires pour le fine-tuning BERT NER (Départ / Arrivée).
- encode_data : encodage des phrases avec alignement BIO via offset_mapping
- get_metrics : matrice de confusion, rapport de classification, courbes loss/accuracy/F1
- get_ROC_curve : courbes ROC multiclasses (one-vs-rest)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import LabelEncoder, label_binarize

# Ordre des labels pour cohérence
UNIQUE_LABELS = ['O', 'B-DEP', 'I-DEP', 'B-ARR', 'I-ARR']


def _find_entity_spans(phrase: str, depart: str, arrivee: str):
    """Retourne (dep_start, dep_end), (arr_start, arr_end) en indices caractères (phrase en minuscules)."""
    phrase_lower = phrase.lower()
    dep_lower = (depart or "").strip().lower()
    arr_lower = (arrivee or "").strip().lower()
    dep_span = (-1, -1)
    arr_span = (-1, -1)
    if dep_lower:
        idx = phrase_lower.find(dep_lower)
        if idx != -1:
            dep_span = (idx, idx + len(dep_lower))
    if arr_lower:
        idx = phrase_lower.find(arr_lower)
        if idx != -1:
            arr_span = (idx, idx + len(arr_lower))
    return dep_span, arr_span


def _get_entity_spans_script3(phrase: str, depart: str, arrivee: str):
    """Retourne (dep_span, arr_span) en utilisant la tokenisation + alignement du script 3 (regex améliorée)."""
    try:
        from script3_regexp_improved import get_entity_spans_from_script3
        return get_entity_spans_from_script3(phrase, depart or "", arrivee or "")
    except ImportError:
        return _find_entity_spans(phrase, depart, arrivee)


def _span_overlap(token_start: int, token_end: int, entity_start: int, entity_end: int) -> bool:
    """True si le token [token_start, token_end) chevauche l'entité [entity_start, entity_end)."""
    if entity_start < 0:
        return False
    return not (token_end <= entity_start or token_start >= entity_end)


def _which_entity(token_start: int, token_end: int, dep_span, arr_span):
    """Retourne 'DEP' ou 'ARR' ou None selon le chevauchement (priorité au premier en cas de chevauchement)."""
    dep_start, dep_end = dep_span
    arr_start, arr_end = arr_span
    in_dep = _span_overlap(token_start, token_end, dep_start, dep_end)
    in_arr = _span_overlap(token_start, token_end, arr_start, arr_end)
    if in_dep and in_arr:
        return 'DEP' if dep_start <= arr_start else 'ARR'
    if in_dep:
        return 'DEP'
    if in_arr:
        return 'ARR'
    return None


def encode_data(data_list, tokenizer, label_encoder, max_length=64, verbose=False, use_script3=False):
    """
    Encode les données pour TFBertForTokenClassification.
    
    data_list: liste de dicts avec clés 'phrase', 'depart', 'arrivee'
    tokenizer: BertTokenizerFast (transformers)
    label_encoder: sklearn LabelEncoder fitté sur ['O','B-DEP','I-DEP','B-ARR','I-ARR']
    max_length: longueur max des séquences
    use_script3: si True, les spans Départ/Arrivée sont dérivés de la tokenisation script 3 (regex améliorée)
    
    Returns:
        encodings: dict avec input_ids, attention_mask, etc. (format attendu par .fit)
        labels: np.array (N, max_length) avec ids 0..4, padding = 0 (O)
    """
    label2id = {lab: i for i, lab in enumerate(label_encoder.classes_)}
    all_input_ids = []
    all_attention_mask = []
    all_labels = []
    span_fn = _get_entity_spans_script3 if use_script3 else _find_entity_spans

    for idx, row in enumerate(data_list):
        phrase = row['phrase']
        depart = row.get('depart', '')
        arrivee = row.get('arrivee', '')
        dep_span, arr_span = span_fn(phrase, depart, arrivee)
        
        enc = tokenizer(
            phrase,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )
        offset_mapping = enc['offset_mapping']
        special_tokens_mask = enc['special_tokens_mask']
        
        token_labels = []
        prev_entity = None
        for i, ((s, e), is_special) in enumerate(zip(offset_mapping, special_tokens_mask)):
            if is_special:
                token_labels.append(label2id['O'])
                continue
            entity = _which_entity(s, e, dep_span, arr_span)
            if entity is None:
                token_labels.append(label2id['O'])
                prev_entity = None
            else:
                if entity == prev_entity:
                    token_labels.append(label2id[f'I-{entity}'])
                else:
                    token_labels.append(label2id[f'B-{entity}'])
                prev_entity = entity
        
        all_input_ids.append(enc['input_ids'])
        all_attention_mask.append(enc['attention_mask'])
        all_labels.append(token_labels)
        
        if verbose and idx < 4:
            tokens = tokenizer.convert_ids_to_tokens(enc['input_ids'])
            label_names = [label_encoder.classes_[i] for i in token_labels]
            print(f"\n---> Ligne {idx + 1}")
            print(f"- Original sentence:  {phrase[:80]}{'...' if len(phrase) > 80 else ''}")
            print(f"- Encoded tokens:  {tokens[:20]}{'...' if len(tokens) > 20 else ''}")
            print(f"- Labels after encoding the entities:  {label_names[:20]}{'...' if len(label_names) > 20 else ''}")
            print(f"- Label IDs {np.array(token_labels)[:20]}{'...' if len(token_labels) > 20 else ''}")
    
    if verbose:
        print("\n// Encoding completed\n")
    
    return (
        {
            'input_ids': np.array(all_input_ids, dtype=np.int32),
            'attention_mask': np.array(all_attention_mask, dtype=np.int32),
        },
        np.array(all_labels, dtype=np.int32),
    )


def get_metrics(true_labels, predicted_labels, unique_labels, history=None, label_encoder=None):
    """
    Affiche matrice de confusion, rapport de classification, et courbes (loss, accuracy, F1 par epoch).
    true_labels: flatten array (ex. test_labels.flatten())
    predicted_labels: flatten array (ex. argmax des logits, flatten)
    unique_labels: liste des noms de classes
    history: objet history de model.fit (optionnel)
    label_encoder: optionnel, pour afficher les noms
    """
    if label_encoder is not None:
        unique_labels = list(label_encoder.classes_)
    
    # Masquer les positions padding si on utilise un id spécial (ici on utilise O=0 pour padding, on garde tout)
    cm = confusion_matrix(true_labels, predicted_labels, labels=list(range(len(unique_labels))))
    cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
    
    print("Confusion Matrix %:")
    print(cm_normalized.round(2))
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(
        true_labels, predicted_labels,
        labels=list(range(len(unique_labels))),
        target_names=unique_labels,
        zero_division=0,
    ))
    
    # Courbes: loss, accuracy, et F1 par epoch si history fourni
    if history is not None and hasattr(history, 'history'):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        epochs = range(1, len(history.history['loss']) + 1)
        
        axes[0].plot(epochs, history.history['loss'], 'b-', label='Train loss')
        axes[0].plot(epochs, history.history['val_loss'], 'r-', label='Val loss')
        axes[0].set_title('Loss')
        axes[0].legend()
        axes[0].set_xlabel('Epoch')
        
        axes[1].plot(epochs, history.history['accuracy'], 'b-', label='Train accuracy')
        axes[1].plot(epochs, history.history['val_accuracy'], 'r-', label='Val accuracy')
        axes[1].set_title('Accuracy')
        axes[1].legend()
        axes[1].set_xlabel('Epoch')
        
        # F1 par epoch (enregistré par F1Callback)
        if 'val_f1' in history.history:
            axes[2].plot(epochs, history.history['val_f1'], 'r-o', label='Val F1 (weighted)')
        else:
            axes[2].text(0.5, 0.5, 'F1 par epoch\n(voir callback F1)', ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title('F1 score (validation)')
        axes[2].set_xlabel('Epoch')
        axes[2].legend()
        axes[2].set_ylim(0, 1.05)
        
        plt.suptitle('Evolution over epochs')
        plt.tight_layout()
        plt.savefig('bert_ner_metrics_curves.png', dpi=120, bbox_inches='tight')
        plt.show()
    
    # Figure matrice de confusion
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_normalized, cmap='Blues')
    ax.set_xticks(range(len(unique_labels)))
    ax.set_yticks(range(len(unique_labels)))
    ax.set_xticklabels(unique_labels)
    ax.set_yticklabels(unique_labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    for i in range(len(unique_labels)):
        for j in range(len(unique_labels)):
            ax.text(j, i, f'{cm_normalized[i, j]:.2f}', ha='center', va='center', color='black' if cm_normalized[i, j] < 0.6 else 'white')
    plt.colorbar(im, ax=ax, label='Proportion')
    plt.title('Confusion Matrix (normalized)')
    plt.tight_layout()
    plt.savefig('bert_ner_confusion_matrix.png', dpi=120, bbox_inches='tight')
    plt.show()


def get_ROC_curve(true_labels, logits, unique_labels, label_encoder=None):
    """
    Courbes ROC multiclasses (one-vs-rest).
    true_labels: (N, seq_len) ou flatten
    logits: (N, seq_len, num_classes)
    unique_labels: noms des classes
    """
    if label_encoder is not None:
        unique_labels = list(label_encoder.classes_)
    
    y_true = np.array(true_labels).flatten()
    # logits: (batch, seq, num_classes) -> flatten to (total_tokens, num_classes)
    logits_flat = logits.reshape(-1, logits.shape[-1])
    
    n_classes = len(unique_labels)
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    
    # On garde les positions où le vrai label est valide (pas de padding hors 0)
    valid = (y_true >= 0) & (y_true < n_classes)
    y_true_bin = y_true_bin[valid]
    logits_flat = logits_flat[valid]
    
    # Probas par softmax
    exp_logits = np.exp(logits_flat - logits_flat.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2, label=f'{unique_labels[i]} (AUC = {roc_auc[i]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC multiclasses (one-vs-rest)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('bert_ner_roc_curve.png', dpi=120, bbox_inches='tight')
    plt.show()
