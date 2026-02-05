"""
Fine-tuning de dbmdz/bert-base-french-europeana-cased pour la NER Départ/Arrivée.
Dataset: dataset_input.txt + dataset_output.txt
- 10 epochs, courbe F1 par epoch, courbes ROC multiclasses.
"""

import os
import warnings
import numpy as np

# Réduire les logs TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore')

from transformers import BertTokenizerFast, TFBertForTokenClassification
from sklearn.preprocessing import LabelEncoder

from bert_ner_utils import (
    encode_data,
    get_metrics,
    get_ROC_curve,
    UNIQUE_LABELS,
)

# Chemins
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '..', '..', 'dataset'))
INPUT_PATH = os.path.join(DATASET_DIR, 'dataset_input.txt')
OUTPUT_PATH = os.path.join(DATASET_DIR, 'dataset_output.txt')
MODEL_NAME = 'dbmdz/bert-base-french-europeana-cased'
SAVE_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, 'bert_europeana_ner'))
MAX_LENGTH = 64
EPOCHS = 10
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.1
RANDOM_STATE = 42


def load_dataset(limit=None):
    """Charge (phrase, depart, arrivee) depuis dataset_input.txt et dataset_output.txt."""
    data = []
    with open(INPUT_PATH, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_PATH, 'r', encoding='utf-8') as f_out:
        count = 0
        for line_in, line_out in zip(f_in, f_out):
            if limit and count >= limit:
                break
            parts_in = line_in.strip().split(',', 1)
            parts_out = line_out.strip().split(',')
            if len(parts_in) < 2 or len(parts_out) < 2:
                continue
            phrase = parts_in[1]
            depart = parts_out[1].strip() if len(parts_out) > 1 else ''
            arrivee = parts_out[2].strip() if len(parts_out) > 2 else ''
            if depart == 'INVALID' or not depart or not arrivee:
                continue
            data.append({'phrase': phrase, 'depart': depart, 'arrivee': arrivee})
            count += 1
    return data


class F1Callback(tf.keras.callbacks.Callback):
    """Calcule F1 (weighted) sur le set de validation à chaque epoch."""
    def __init__(self, val_encodings, val_labels, label_encoder):
        super().__init__()
        self.val_encodings = val_encodings
        self.val_labels = val_labels
        self.label_encoder = label_encoder

    def on_epoch_end(self, epoch, logs=None):
        from sklearn.metrics import f1_score
        pred = self.model.predict(self.val_encodings, verbose=0)
        pred_ids = np.argmax(pred.logits, axis=-1).flatten()
        true_ids = self.val_labels.flatten()
        # Ignorer les positions padding (label 0 = O, on garde tout pour simplicité)
        n_classes = len(self.label_encoder.classes_)
        mask = (true_ids >= 0) & (true_ids < n_classes)
        f1 = f1_score(true_ids[mask], pred_ids[mask], average='weighted', zero_division=0)
        logs = logs or {}
        logs['val_f1'] = float(f1)


def main():
    print("// Preparation of labels ['O', 'B-DEP', 'I-DEP', 'B-ARR', 'I-ARR']")
    label_encoder = LabelEncoder()
    label_encoder.fit(UNIQUE_LABELS)
    print("-> mapping labels and ids:", {label: idx for idx, label in enumerate(label_encoder.classes_)})

    if not os.path.exists(INPUT_PATH) or not os.path.exists(OUTPUT_PATH):
        raise FileNotFoundError(f"Dataset not found: {INPUT_PATH} / {OUTPUT_PATH}")

    # Charger tout le dataset (ou une limite pour test rapide)
    data = load_dataset(limit=None)
    np.random.seed(RANDOM_STATE)
    np.random.shuffle(data)
    print(f"-> Loaded {len(data)} samples")

    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

    print("\n// Encoding data...")
    encodings, labels = encode_data(data, tokenizer, label_encoder, max_length=MAX_LENGTH, verbose=True)

    # Split train / val
    n = len(data)
    n_val = int(n * VALIDATION_SPLIT)
    n_train = n - n_val
    train_enc = {k: v[:n_train] for k, v in encodings.items()}
    train_lab = labels[:n_train]
    val_enc = {k: v[n_train:] for k, v in encodings.items()}
    val_lab = labels[n_train:]
    print(f"-> Train: {n_train}, Val: {n_val}")

    # Modèle
    model = TFBertForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_encoder.classes_),
    )
    print("-> model loaded with", len(label_encoder.classes_), "labels")

    # legacy.Adam recommandé sur M1/M2 Mac (v2.11+)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=5e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    f1_callback = F1Callback(val_enc, val_lab, label_encoder)

    print("\n// Starting model training (10 epochs)...")
    history = model.fit(
        train_enc,
        train_lab,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(val_enc, val_lab),
        callbacks=[f1_callback],
    )

    # Sauvegarde
    os.makedirs(SAVE_DIR, exist_ok=True)
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"\n// Model saved to {os.path.abspath(SAVE_DIR)}")

    # Évaluation sur le set de validation (comme "test" ici)
    print("\n// Evaluation on validation set...")
    eval_res = model.evaluate(val_enc, val_lab)
    print("Evaluation results (loss, accuracy):", eval_res)

    test_predictions = model.predict(val_enc)
    pred_labels = np.argmax(test_predictions.logits, axis=-1).flatten()
    true_labels_flat = val_lab.flatten()

    get_metrics(
        true_labels=true_labels_flat,
        predicted_labels=pred_labels,
        unique_labels=UNIQUE_LABELS,
        history=history,
        label_encoder=label_encoder,
    )
    get_ROC_curve(
        true_labels=val_lab,
        logits=test_predictions.logits,
        unique_labels=UNIQUE_LABELS,
        label_encoder=label_encoder,
    )

    # Courbe F1 seule (par epoch)
    if 'val_f1' in history.history:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, EPOCHS + 1), history.history['val_f1'], 'o-', color='C1', label='Val F1 (weighted)')
        plt.xlabel('Epoch')
        plt.ylabel('F1 score')
        plt.title('F1 score over epochs')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(SCRIPT_DIR, 'bert_ner_f1_epochs.png'), dpi=120, bbox_inches='tight')
        plt.show()
        print("\nF1 per epoch:", [round(x, 4) for x in history.history['val_f1']])


if __name__ == '__main__':
    main()
