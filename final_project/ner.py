"""
NER — Chargement de modèles et extraction d'entités de lieu.

Supporte :
  - Modèles CamemBERT / RoBERTa avec labels LOC
  - Modèles custom avec labels DEP/ARR ou ORIGIN/DEST
  - Modèles BERT avec poids TensorFlow (tf_model.h5)
  - Téléchargement depuis Hugging Face Hub
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple

from transformers import AutoConfig, AutoTokenizer, BertForTokenClassification, pipeline

try:
    from huggingface_hub import snapshot_download as _hf_snapshot_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

from utils import get_device

# ─── Constantes ───────────────────────────────────────────────────────────────

RECIPE_CUES = (
    "recette", "cuisine", "pâtisserie", "patisserie", "dessert",
    "gâteau", "gateau", "un paris-brest", "une religieuse",
    "un monaco", "boire un monaco",
)

_MIN_SCORE_DEP_ARR = 0.85
_MAX_GAP = 5

_LOC_LABELS = frozenset({
    "B-LOC", "I-LOC", "B-ORIGIN", "I-ORIGIN",
    "B-DEST", "I-DEST", "ORIGIN", "DEST",
})


# ─── Téléchargement HF ───────────────────────────────────────────────────────

def download_hf_model(repo_id: str) -> str:
    """
    Télécharge un modèle depuis Hugging Face Hub (avec cache).
    Retourne le chemin local vers le snapshot.
    """
    if not HF_HUB_AVAILABLE:
        raise RuntimeError(
            "huggingface-hub n'est pas installé. "
            "Installe-le avec : pip install huggingface-hub"
        )
    return _hf_snapshot_download(repo_id=repo_id)


# ─── Chargement du pipeline NER ──────────────────────────────────────────────

def load_ner(model_name: str) -> dict:
    """
    Charge un pipeline NER depuis un chemin local ou un nom Hugging Face.

    Retourne un dict {"ner": pipeline, "use_dep_arr": bool}.
    - use_dep_arr=True si le modèle utilise les labels DEP/ARR/ORIGIN/DEST.
    - use_dep_arr=False si le modèle utilise les labels LOC classiques.
    """
    device = get_device()
    model_path = str(Path(model_name).expanduser())
    tf_weights = str(Path(model_path) / "tf_model.h5")

    if Path(tf_weights).is_file():
        # Modèle BERT avec poids TF
        model = BertForTokenClassification.from_pretrained(
            model_path, from_tf=True, low_cpu_mem_usage=False
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        ner = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=device,
        )
        use_dep_arr = True
    else:
        # Pipeline classique (CamemBERT, RoBERTa, modèle HF, etc.)
        ner = pipeline(
            "ner",
            model=model_name,
            tokenizer=model_name,
            aggregation_strategy="simple",
            device=device,
        )
        use_dep_arr = False
        try:
            config = AutoConfig.from_pretrained(model_name)
            all_labels = " ".join(str(v).upper() for v in config.id2label.values())
            if any(tok in all_labels for tok in ("DEP", "ARR", "ORIGIN", "DEST")):
                use_dep_arr = True
        except Exception:
            pass

    return {"ner": ner, "use_dep_arr": use_dep_arr}


# ─── Extraction d'entités ────────────────────────────────────────────────────

def _merge_spans(spans: List[Tuple[int, int, str]], phrase: str) -> str:
    """Fusionne des spans proches et retourne la chaîne résultante."""
    if not spans:
        return ""
    groups = [[spans[0]]]
    for span in spans[1:]:
        prev_end = groups[-1][-1][1]
        if span[0] - prev_end <= _MAX_GAP:
            groups[-1].append(span)
        else:
            groups.append([span])
    best = max(groups, key=lambda g: g[-1][1] - g[0][0])
    if phrase:
        return phrase[best[0][0]: best[-1][1]].strip()
    return " ".join(part[2] for part in best if part[2]).strip()


def extract_locations_from_ner(entities: list, phrase: str) -> List[str]:
    """
    Extrait les entités LOC et fusionne les tokens consécutifs.
    Utilisé quand le modèle produit des labels B-LOC / I-LOC.
    """
    loc_entities: List[Tuple[int, int, str]] = []
    for ent in entities:
        label = ent.get("entity_group", ent.get("label", "")).upper()
        if "LOC" in label or label in {"B-LOC", "I-LOC"}:
            start = ent.get("start")
            end = ent.get("end")
            word = (ent.get("word", ent.get("text", "")) or "").strip()
            if isinstance(start, int) and isinstance(end, int) and end > start:
                loc_entities.append((start, end, word))
            elif word:
                loc_entities.append((-1, -1, word))

    if not loc_entities:
        return []

    spans = [item for item in loc_entities if item[0] >= 0 and item[1] > item[0]]
    if spans:
        spans.sort(key=lambda item: item[0])
        merged: List[str] = []
        cur_start, cur_end, cur_word = spans[0]
        for start, end, word in spans[1:]:
            if start <= cur_end + 5:
                cur_end = max(cur_end, end)
            else:
                if phrase and cur_start < len(phrase) and cur_end <= len(phrase):
                    merged.append(phrase[cur_start:cur_end].strip())
                else:
                    merged.append(cur_word)
                cur_start, cur_end, cur_word = start, end, word
        if phrase and cur_start < len(phrase) and cur_end <= len(phrase):
            merged.append(phrase[cur_start:cur_end].strip())
        else:
            merged.append(cur_word)
        return [m for m in merged if m]

    return [item[2] for item in loc_entities if item[2]]


def extract_dep_arr_from_ner(entities: list, phrase: str) -> List[str]:
    """
    Extrait départ et arrivée depuis les labels DEP/ARR ou ORIGIN/DEST.
    Retourne [départ, arrivée] ou liste partielle.
    """
    def _extract_with_min_score(min_score: float) -> List[str]:
        dep_spans: List[Tuple[int, int, str]] = []
        arr_spans: List[Tuple[int, int, str]] = []
        for ent in entities:
            score = ent.get("score", 1.0)
            if score < min_score:
                continue
            label = str(ent.get("entity_group", ent.get("label", ""))).upper()
            start = ent.get("start")
            end = ent.get("end")
            word = (ent.get("word", ent.get("text", "")) or "").strip()
            if not (isinstance(start, int) and isinstance(end, int) and end > start):
                continue
            if "DEP" in label or "ORIGIN" in label:
                dep_spans.append((start, end, word))
            elif "ARR" in label or "DEST" in label:
                arr_spans.append((start, end, word))

        dep_spans.sort(key=lambda x: x[0])
        arr_spans.sort(key=lambda x: x[0])
        dep = _merge_spans(dep_spans, phrase)
        arr = _merge_spans(arr_spans, phrase)

        out: List[str] = []
        if dep:
            out.append(dep)
        if arr:
            out.append(arr)
        return out

    strict = _extract_with_min_score(_MIN_SCORE_DEP_ARR)
    if len(strict) >= 2:
        return strict
    return _extract_with_min_score(0.0)


def extract_locations_from_any_location_labels(entities: list, phrase: str) -> List[str]:
    """
    Fallback : récupère tout label de lieu (LOC/DEP/ARR/ORIGIN/DEST)
    puis fusionne les spans proches.
    """
    spans: List[Tuple[int, int, str]] = []
    for ent in entities:
        label = str(ent.get("entity_group", ent.get("label", ""))).upper()
        if not any(tag in label for tag in ("LOC", "DEP", "ARR", "ORIGIN", "DEST")):
            continue
        start = ent.get("start")
        end = ent.get("end")
        word = (ent.get("word", ent.get("text", "")) or "").strip()
        if isinstance(start, int) and isinstance(end, int) and end > start:
            spans.append((start, end, word))
        elif word:
            spans.append((-1, -1, word))

    if not spans:
        return []

    with_offsets = [s for s in spans if s[0] >= 0 and s[1] > s[0]]
    if not with_offsets:
        return [s[2] for s in spans if s[2]]

    with_offsets.sort(key=lambda s: s[0])
    merged: List[str] = []
    cur_start, cur_end, cur_word = with_offsets[0]
    for start, end, word in with_offsets[1:]:
        if start <= cur_end + _MAX_GAP:
            cur_end = max(cur_end, end)
        else:
            if phrase and cur_start < len(phrase) and cur_end <= len(phrase):
                merged.append(phrase[cur_start:cur_end].strip())
            else:
                merged.append(cur_word)
            cur_start, cur_end, cur_word = start, end, word
    if phrase and cur_start < len(phrase) and cur_end <= len(phrase):
        merged.append(phrase[cur_start:cur_end].strip())
    else:
        merged.append(cur_word)
    return [m for m in merged if m]


# ─── Post-traitement des lieux ───────────────────────────────────────────────

def normalize_locations(locations: List[str]) -> List[str]:
    """Nettoie les noms de lieux : supprime doublons, ponctuation, trop courts."""
    cleaned: List[str] = []
    seen: set = set()
    for loc in locations:
        loc = re.sub(r"\s+", " ", loc).strip(" ,.;:!?")
        if len(loc) < 2:
            continue
        key = loc.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(loc)
    return cleaned


def filter_non_travel_locs(phrase: str, locs: List[str]) -> List[str]:
    """
    Filtre les LOC quand la phrase parle de recette/cuisine.
    Ex: Paris-Brest (pâtisserie), Monaco (cocktail).
    """
    if not locs:
        return locs
    phrase_lower = phrase.lower().strip()
    for cue in RECIPE_CUES:
        if cue in phrase_lower:
            if len(locs) >= 2 and "paris" in phrase_lower and "brest" in phrase_lower:
                return []
            if "monaco" in phrase_lower and any(l.lower() == "monaco" for l in locs):
                return [l for l in locs if l.lower() != "monaco"]
    return locs


# ─── Fonction principale d'extraction ────────────────────────────────────────

def extract_cities(user_text: str, ner_bundle: dict) -> List[str]:
    """
    Pipeline complet : NER → extraction → filtrage → normalisation.
    Retourne la liste ordonnée de villes détectées.
    """
    ner = ner_bundle["ner"]
    use_dep_arr = ner_bundle["use_dep_arr"]
    entities = ner(user_text)

    if use_dep_arr:
        locations = extract_dep_arr_from_ner(entities, user_text)
        locations = normalize_locations(locations)
    else:
        locations = extract_locations_from_ner(entities, user_text)
        locations = filter_non_travel_locs(user_text, locations)
        locations = normalize_locations(locations)

    if len(locations) < 2:
        # Fallback : prend tous les labels de lieu détectés
        locations = normalize_locations(
            extract_locations_from_any_location_labels(entities, user_text)
        )

    return locations
