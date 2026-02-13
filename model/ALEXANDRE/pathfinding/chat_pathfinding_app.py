import re
import importlib.util
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import pydeck as pdk
import streamlit as st
import torch
from transformers import AutoConfig, AutoTokenizer, BertForTokenClassification, pipeline

from sncf_pathfinder import RouteResult, SNCFPathFinder


RECIPE_CUES = (
    "recette",
    "cuisine",
    "pâtisserie",
    "patisserie",
    "dessert",
    "gâteau",
    "gateau",
    "un paris-brest",
    "une religieuse",
    "un monaco",
    "boire un monaco",
)

_MIN_SCORE_DEP_ARR = 0.85
_MAX_GAP = 5


def get_device():
    if torch.cuda.is_available():
        return 0
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return -1


def format_duration(minutes: float) -> str:
    h = int(minutes // 60)
    m = int(minutes % 60)
    if h > 0:
        return f"{h}h{m:02d}"
    return f"{m}min"


def extract_locations_from_ner(entities, phrase: str) -> List[str]:
    loc_entities = []
    for ent in entities:
        label = ent.get("entity_group", ent.get("label", "")).upper()
        if "LOC" in label or label in {"B-LOC", "I-LOC"}:
            start = ent.get("start")
            end = ent.get("end")
            word = (ent.get("word", ent.get("text", "")) or "").strip()
            if isinstance(start, int) and isinstance(end, int) and end > start:
                loc_entities.append((start, end, word))
            elif word:
                # Fallback si start/end ne sont pas exploitables
                loc_entities.append((-1, -1, word))

    if not loc_entities:
        return []

    spans = [item for item in loc_entities if item[0] >= 0 and item[1] > item[0]]
    if spans:
        spans.sort(key=lambda item: item[0])
        merged = []
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

    # Fallback final: utiliser les tokens words
    words = [item[2] for item in loc_entities if item[2]]
    return words


def _merge_spans(spans, phrase: str) -> str:
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
        return phrase[best[0][0] : best[-1][1]].strip()
    return " ".join(part[2] for part in best if part[2]).strip()


def extract_dep_arr_from_ner(entities, phrase: str) -> List[str]:
    """Supporte les labels DEP/ARR et ORIGIN/DEST."""
    def _extract_with_min_score(min_score: float) -> List[str]:
        dep_spans = []
        arr_spans = []
        for ent in entities:
            score = ent.get("score", 1.0)
            if score < min_score:
                continue
            label = ent.get("entity_group", ent.get("label", ""))
            label_u = str(label).upper()
            start = ent.get("start")
            end = ent.get("end")
            word = (ent.get("word", ent.get("text", "")) or "").strip()
            if not (isinstance(start, int) and isinstance(end, int) and end > start):
                continue
            if "DEP" in label_u or "ORIGIN" in label_u:
                dep_spans.append((start, end, word))
            elif "ARR" in label_u or "DEST" in label_u:
                arr_spans.append((start, end, word))

        dep_spans.sort(key=lambda x: x[0])
        arr_spans.sort(key=lambda x: x[0])
        dep = _merge_spans(dep_spans, phrase)
        arr = _merge_spans(arr_spans, phrase)

        out = []
        if dep:
            out.append(dep)
        if arr:
            out.append(arr)
        return out

    # Passage strict puis fallback permissif si une ville manque.
    strict = _extract_with_min_score(_MIN_SCORE_DEP_ARR)
    if len(strict) >= 2:
        return strict
    return _extract_with_min_score(0.0)


def extract_locations_from_any_location_labels(entities, phrase: str) -> List[str]:
    """
    Fallback NER-only:
    récupère les entités portant des labels de lieu (LOC/DEP/ARR/ORIGIN/DEST),
    puis fusionne les spans proches.
    """
    spans = []
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
    merged = []
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


def normalize_locations(locations: List[str]) -> List[str]:
    cleaned = []
    seen = set()
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
    """Même logique que chat_roberta_loc.py (mode LOC)."""
    if not locs:
        return locs
    phrase_lower = phrase.lower().strip()
    for cue in RECIPE_CUES:
        if cue in phrase_lower:
            # Paris-Brest (pâtisserie): deux LOC dans ce contexte -> pas un trajet.
            if len(locs) >= 2 and "paris" in phrase_lower and "brest" in phrase_lower:
                return []
            # "un monaco" / "boire un monaco": retirer Monaco si présent.
            if "monaco" in phrase_lower and "monaco" in [l.lower() for l in locs]:
                return [l for l in locs if l.lower() != "monaco"]
    return locs


@st.cache_resource(show_spinner=True)
def load_ner(model_name: str):
    device = get_device()
    model_path = str(Path(model_name).expanduser())
    tf_weights = str(Path(model_path) / "tf_model.h5")

    # Même logique que chat_roberta_loc.py:
    # - si tf_model.h5 existe: chargement BertForTokenClassification.from_pretrained(..., from_tf=True)
    # - sinon pipeline classique
    # - puis auto-détection des labels DEP/ARR/ORIGIN/DEST
    if Path(tf_weights).is_file():
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
            if any(token in all_labels for token in ("DEP", "ARR", "ORIGIN", "DEST")):
                use_dep_arr = True
        except Exception:
            pass
    return {"ner": ner, "use_dep_arr": use_dep_arr}


@st.cache_resource(show_spinner=True)
def load_pathfinder(gtfs_dir: str):
    pathfinder = SNCFPathFinder(gtfs_dir=gtfs_dir)
    pathfinder.build()
    return pathfinder


def extract_cities(user_text: str, ner_bundle) -> List[str]:
    ner = ner_bundle["ner"]
    use_dep_arr = ner_bundle["use_dep_arr"]
    entities = ner(user_text)
    if use_dep_arr:
        locations = extract_dep_arr_from_ner(entities, user_text)
        # En mode DEP/ARR, on suit chat_roberta_loc.py: pas de filtre recette.
        locations = normalize_locations(locations)
    else:
        locations = extract_locations_from_ner(entities, user_text)
        locations = filter_non_travel_locs(user_text, locations)
        locations = normalize_locations(locations)
    if len(locations) < 2:
        # Fallback sans patterns: on prend tous les labels de lieu détectés par le NER.
        locations = normalize_locations(extract_locations_from_any_location_labels(entities, user_text))
    return locations


def route_to_dataframe(route: RouteResult) -> pd.DataFrame:
    rows = []
    for idx, seg in enumerate(route.segments, start=1):
        rows.append(
            {
                "step": idx,
                "from": seg.from_station_name,
                "to": seg.to_station_name,
                "time_min": round(seg.time_min, 1),
                "train_count": seg.train_count,
            }
        )
    return pd.DataFrame(rows)


def render_map(route: RouteResult, pathfinder: SNCFPathFinder):
    points = []
    for station_id, station_name in zip(route.station_ids, route.station_names):
        coords = pathfinder.station_coordinates(station_id)
        if not coords:
            continue
        lat, lon = coords
        points.append({"station": station_name, "lat": lat, "lon": lon})

    if len(points) < 2:
        st.info("Coordonnees insuffisantes pour dessiner la carte.")
        return

    line_path = [[p["lon"], p["lat"]] for p in points]
    line_data = [{"path": line_path}]
    point_data = [{"position": [p["lon"], p["lat"]], "station": p["station"]} for p in points]

    mid_lat = sum(p["lat"] for p in points) / len(points)
    mid_lon = sum(p["lon"] for p in points) / len(points)

    path_layer = pdk.Layer(
        "PathLayer",
        line_data,
        get_path="path",
        get_color=[255, 99, 71],
        width_scale=8,
        width_min_pixels=3,
    )
    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        point_data,
        get_position="position",
        get_radius=6000,
        get_fill_color=[30, 144, 255, 180],
        pickable=True,
    )

    tooltip = {"text": "{station}"}
    st.pydeck_chart(
        pdk.Deck(
            map_provider="carto",
            map_style="light",
            initial_view_state=pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=5),
            layers=[path_layer, scatter_layer],
            tooltip=tooltip,
        )
    )


def answer_user(user_text: str, ner_bundle, pathfinder: SNCFPathFinder):
    cities = extract_cities(user_text, ner_bundle)

    if len(cities) < 2:
        return {
            "ok": False,
            "message": (
                "Je n'ai pas detecte 2 villes dans ta phrase. "
                "Exemple: 'Je veux aller de Paris a Lyon'."
            ),
        }

    try:
        route = pathfinder.route_with_waypoints(cities)
    except Exception as exc:
        return {"ok": False, "message": f"Impossible de calculer le trajet: {exc}"}

    recap = (
        f"Trajet trouve: **{route.matched_stations[0]}** -> "
        f"**{route.matched_stations[-1]}**\n\n"
        f"- Villes detectees: {', '.join(cities)}\n"
        f"- Temps total estime: **{format_duration(route.total_time_min)}**\n"
        f"- Nombre de gares: **{len(route.station_names)}**\n"
        f"- Correspondances: **{max(0, len(route.station_names) - 2)}**"
    )

    return {"ok": True, "message": recap, "route": route}


@st.cache_resource(show_spinner=False)
def load_stt_module(repo_root: Path):
    stt_path = repo_root / "model" / "LEO" / "stt" / "stt_bridge.py"
    if not stt_path.is_file():
        return None, f"Fichier STT introuvable: {stt_path}"
    try:
        spec = importlib.util.spec_from_file_location("leo_microphone_recognition", stt_path)
        if spec is None or spec.loader is None:
            return None, "Impossible de charger le module STT."
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, None
    except Exception as exc:
        return None, f"Erreur de chargement du module STT: {exc}"


def transcribe_wav_bytes_with_stt(
    audio_bytes: bytes,
    stt_module,
    language: str = "fr-FR",
) -> Tuple[Optional[str], Optional[str]]:
    if stt_module is None:
        return None, "Module STT indisponible."
    transcriber = getattr(stt_module, "transcribe_wav_bytes", None)
    if transcriber is None:
        return None, "Fonction `transcribe_wav_bytes` absente de stt_bridge.py."
    try:
        return transcriber(audio_bytes, language=language)
    except Exception as exc:
        return None, f"Erreur pendant la transcription WAV: {exc}"


def transcribe_server_microphone_with_stt(
    stt_module,
    language: str = "fr-FR",
) -> Tuple[Optional[str], Optional[str]]:
    if stt_module is None:
        return None, "Module STT indisponible."
    transcriber = getattr(stt_module, "transcribe_microphone", None)
    if transcriber is None:
        return None, "Fonction `transcribe_microphone` absente de stt_bridge.py."
    try:
        return transcriber(language=language)
    except Exception as exc:
        return None, f"Erreur pendant la transcription microphone: {exc}"


def process_prompt(prompt: str, ner_bundle, pathfinder: SNCFPathFinder):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    result = answer_user(prompt, ner_bundle, pathfinder)
    assistant_msg = {"role": "assistant", "content": result["message"]}
    if result.get("route") is not None:
        assistant_msg["route"] = result["route"]
    st.session_state.messages.append(assistant_msg)

    with st.chat_message("assistant"):
        st.markdown(result["message"])
        if result.get("route") is not None:
            route = result["route"]
            st.subheader("Recap des stations")
            st.dataframe(route_to_dataframe(route), use_container_width=True, hide_index=True)
            st.subheader("Carte du trajet")
            render_map(route, pathfinder)


def main():
    st.set_page_config(page_title="SNCF AI Chat Pathfinding", layout="wide")
    st.title("SNCF AI Chat Pathfinding")
    st.caption("Chat IA -> extraction des villes -> pathfinding GTFS -> recap + carte")

    repo_root = Path(__file__).resolve().parents[3]
    gtfs_dir = str(repo_root / "dataset" / "sncf")
    stt_module, stt_error = load_stt_module(repo_root)

    default_local_model = str((repo_root / "roberta_ner_model").resolve())
    if not Path(default_local_model).exists():
        default_local_model = "Jean-Baptiste/camembert-ner"

    with st.sidebar:
        st.subheader("Configuration")
        model_name = st.text_input("Modele NER", value=default_local_model)
        st.write(f"GTFS: `{gtfs_dir}`")
        st.write("Le graphe est calcule une fois puis mis en cache.")

    try:
        ner_bundle = load_ner(model_name)
        pathfinder = load_pathfinder(gtfs_dir)
    except Exception as exc:
        st.error(f"Erreur d'initialisation: {exc}")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("route") is not None:
                route: RouteResult = msg["route"]
                st.dataframe(route_to_dataframe(route), use_container_width=True, hide_index=True)
                render_map(route, pathfinder)

    st.divider()
    st.subheader("Entree audio")
    st.caption("Tu peux envoyer un .wav ou utiliser le micro, puis envoyer la transcription dans le chat.")
    if stt_error:
        st.warning(stt_error)
        st.info("Verifie aussi que `SpeechRecognition` est installe (`pip install SpeechRecognition`).")

    audio_mode = st.radio("Source audio", ["Fichier WAV", "Microphone"], horizontal=True)
    audio_prompt = None

    if audio_mode == "Fichier WAV":
        uploaded_wav = st.file_uploader("Importer un fichier .wav", type=["wav"])
        if uploaded_wav is not None:
            wav_bytes = uploaded_wav.getvalue()
            st.audio(wav_bytes, format="audio/wav")
            if st.button("Transcrire et envoyer", key="send_wav"):
                with st.spinner("Transcription en cours..."):
                    audio_prompt, error = transcribe_wav_bytes_with_stt(wav_bytes, stt_module)
                if error:
                    st.error(error)
                else:
                    st.success(f"Transcription: {audio_prompt}")
    else:
        if hasattr(st, "audio_input"):
            mic_audio = st.audio_input("Enregistrer depuis le microphone")
            if mic_audio is not None:
                mic_bytes = mic_audio.getvalue()
                st.audio(mic_bytes, format="audio/wav")
                if st.button("Transcrire et envoyer", key="send_mic"):
                    with st.spinner("Transcription en cours..."):
                        audio_prompt, error = transcribe_wav_bytes_with_stt(mic_bytes, stt_module)
                    if error:
                        st.error(error)
                    else:
                        st.success(f"Transcription: {audio_prompt}")
        else:
            st.info("Cette version de Streamlit ne supporte pas `st.audio_input`.")
            st.caption("Option de secours: utiliser le microphone local de la machine serveur.")
            if st.button("Ecouter le micro serveur et envoyer", key="send_server_mic"):
                with st.spinner("Ecoute du microphone serveur..."):
                    audio_prompt, error = transcribe_server_microphone_with_stt(stt_module)
                if error:
                    st.error(error)
                else:
                    st.success(f"Transcription: {audio_prompt}")

    if audio_prompt:
        process_prompt(audio_prompt, ner_bundle, pathfinder)
        return

    prompt = st.chat_input("Ex: Je veux aller de Nantes a Marseille en passant par Angers")
    if prompt:
        process_prompt(prompt, ner_bundle, pathfinder)


if __name__ == "__main__":
    main()
