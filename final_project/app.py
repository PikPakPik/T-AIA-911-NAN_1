"""
App — Interface Streamlit : chat + carte + horaires.

Lancement :
    streamlit run final_project/app.py
"""

from pathlib import Path
from typing import List

import pandas as pd
import pydeck as pdk
import streamlit as st

from ner import (
    HF_HUB_AVAILABLE,
    download_hf_model,
    extract_cities,
    load_ner,
)
from pathfinder import (
    ConnectionInfo,
    RouteResult,
    ScheduleResult,
    SNCFPathFinder,
    TrainLeg,
)
from stt import load_stt_module, transcribe_microphone, transcribe_wav_bytes
from utils import format_duration


# ─── Constantes ───────────────────────────────────────────────────────────────

HF_MODEL_ID = "PikPak44/yann_best_model"


# ─── Fonctions d'affichage ───────────────────────────────────────────────────

def route_to_dataframe(route: RouteResult) -> pd.DataFrame:
    """Convertit un RouteResult en DataFrame récapitulatif."""
    rows = []
    for idx, seg in enumerate(route.segments, start=1):
        rows.append({
            "Étape": idx,
            "De": seg.from_station_name,
            "À": seg.to_station_name,
            "Temps (min)": round(seg.time_min, 1),
            "Trains/jour": seg.train_count,
        })
    return pd.DataFrame(rows)


def render_map(route: RouteResult, pathfinder: SNCFPathFinder):
    """Affiche la carte du trajet avec pydeck."""
    points = []
    for station_id, station_name in zip(route.station_ids, route.station_names):
        coords = pathfinder.station_coordinates(station_id)
        if not coords:
            continue
        lat, lon = coords
        points.append({"station": station_name, "lat": lat, "lon": lon})

    if len(points) < 2:
        st.info("Coordonnées insuffisantes pour dessiner la carte.")
        return

    line_path = [[p["lon"], p["lat"]] for p in points]
    mid_lat = sum(p["lat"] for p in points) / len(points)
    mid_lon = sum(p["lon"] for p in points) / len(points)

    path_layer = pdk.Layer(
        "PathLayer",
        [{"path": line_path}],
        get_path="path",
        get_color=[255, 99, 71],
        width_scale=8,
        width_min_pixels=3,
    )
    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        [{"position": [p["lon"], p["lat"]], "station": p["station"]} for p in points],
        get_position="position",
        get_radius=6000,
        get_fill_color=[30, 144, 255, 180],
        pickable=True,
    )

    st.pydeck_chart(
        pdk.Deck(
            map_provider="carto",
            map_style="light",
            initial_view_state=pdk.ViewState(
                latitude=mid_lat, longitude=mid_lon, zoom=5
            ),
            layers=[path_layer, scatter_layer],
            tooltip={"text": "{station}"},
        )
    )


def format_schedule_markdown(schedule: ScheduleResult) -> str:
    """Génère le markdown des horaires détaillés."""
    text = "\n---\n\n🚆 **Horaires détaillés (exemple)**\n"
    for leg_idx, leg in enumerate(schedule.legs):
        if leg_idx > 0 and leg_idx - 1 < len(schedule.connections):
            conn = schedule.connections[leg_idx - 1]
            text += (
                f"\n🔄 **Correspondance à {conn.station_name}** "
                f"— **{format_duration(conn.wait_min)}** d'attente "
                f"({conn.arrive_time} → {conn.depart_time})\n"
            )
        type_str = leg.train_type if leg.train_type else "Train"
        text += f"\n**{type_str} n°{leg.train_number}**\n"
        for si, stop in enumerate(leg.stops):
            if si == 0:
                text += f"- {stop.departure_time} — **{stop.station_name}** *(départ)*\n"
            elif si == len(leg.stops) - 1:
                text += f"- {stop.arrival_time} — **{stop.station_name}** *(arrivée)*\n"
            else:
                text += f"- {stop.arrival_time} — {stop.station_name}\n"

    if schedule.total_wait_min > 0:
        text += (
            f"\n⏱️ Temps de correspondance total : "
            f"**{format_duration(schedule.total_wait_min)}**"
        )
    return text


# ─── Logique métier ──────────────────────────────────────────────────────────

def answer_user(user_text: str, ner_bundle, pathfinder: SNCFPathFinder) -> dict:
    """
    Pipeline complet : extraction NER → pathfinding → horaires.
    Retourne {"ok": bool, "message": str, "route": RouteResult | None}.
    """
    cities = extract_cities(user_text, ner_bundle)

    if len(cities) < 2:
        return {
            "ok": False,
            "message": (
                "Je n'ai pas détecté 2 villes dans ta phrase. "
                "Exemple : 'Je veux aller de Paris à Lyon'."
            ),
        }

    try:
        route = pathfinder.route_with_waypoints(cities)
    except Exception as exc:
        return {"ok": False, "message": f"Impossible de calculer le trajet : {exc}"}

    recap = (
        f"Trajet trouvé : **{route.matched_stations[0]}** → "
        f"**{route.matched_stations[-1]}**\n\n"
        f"- Villes détectées : {', '.join(cities)}\n"
        f"- Temps total estimé : **{format_duration(route.total_time_min)}**\n"
        f"- Nombre de gares : **{len(route.station_names)}**\n"
        f"- Correspondances : **{max(0, len(route.station_names) - 2)}**"
    )

    # Horaires concrets
    schedule = None
    try:
        schedule = pathfinder.find_schedule_for_route(route.station_ids)
    except Exception:
        pass

    if schedule and schedule.legs:
        recap += format_schedule_markdown(schedule)

    return {"ok": True, "message": recap, "route": route}


# ─── Affichage d'un message ──────────────────────────────────────────────────

def process_prompt(prompt: str, ner_bundle, pathfinder: SNCFPathFinder):
    """Traite un message utilisateur et affiche la réponse."""
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
            st.subheader("Récap des stations")
            st.dataframe(route_to_dataframe(route), use_container_width=True, hide_index=True)
            st.subheader("Carte du trajet")
            render_map(route, pathfinder)


# ─── Chargement (avec cache Streamlit) ───────────────────────────────────────

@st.cache_resource(show_spinner="Téléchargement du modèle depuis Hugging Face…")
def _cached_download_hf_model(repo_id: str) -> str:
    return download_hf_model(repo_id)


@st.cache_resource(show_spinner="Chargement du modèle NER…")
def _cached_load_ner(model_name: str):
    return load_ner(model_name)


@st.cache_resource(show_spinner="Construction du graphe GTFS…")
def _cached_load_pathfinder(gtfs_dir: str, geo_csv_path: str = ""):
    pf = SNCFPathFinder(gtfs_dir=gtfs_dir, geo_csv_path=geo_csv_path or None)
    pf.build()
    return pf


@st.cache_resource(show_spinner=False)
def _cached_load_stt(repo_root_str: str):
    return load_stt_module(Path(repo_root_str))


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="SNCF AI — Travel Order Resolver", layout="wide")
    st.title("🚄 SNCF AI — Travel Order Resolver")
    st.caption("Chat IA → extraction NER → pathfinding GTFS → horaires + carte")

    # Chemins
    repo_root = Path(__file__).resolve().parents[1]
    gtfs_dir = str(repo_root / "dataset" / "sncf")
    geo_csv_path = str(repo_root / "dataset" / "geo.csv")
    stt_module, stt_error = _cached_load_stt(str(repo_root))

    # ── Sidebar ──
    with st.sidebar:
        st.subheader("⚙️ Configuration")

        model_source = st.radio(
            "Source du modèle NER",
            [f"🤗 Hugging Face ({HF_MODEL_ID})", "📁 Chemin local"],
            index=0,
        )

        if model_source.startswith("🤗"):
            if not HF_HUB_AVAILABLE:
                st.error("`huggingface-hub` non installé. `pip install huggingface-hub`")
                st.stop()
            model_name = _cached_download_hf_model(HF_MODEL_ID)
            st.success("Modèle chargé depuis HF")
            st.caption(f"`{model_name}`")
        else:
            default_local = str((repo_root / "roberta_ner_model").resolve())
            if not Path(default_local).exists():
                default_local = "Jean-Baptiste/camembert-ner"
            model_name = st.text_input("Chemin du modèle local", value=default_local)

        st.divider()
        st.caption(f"📂 GTFS : `{gtfs_dir}`")
        st.caption("Le graphe est calculé une fois puis mis en cache.")

    # ── Chargement ──
    try:
        ner_bundle = _cached_load_ner(model_name)
        pathfinder = _cached_load_pathfinder(gtfs_dir, geo_csv_path)
    except Exception as exc:
        st.error(f"Erreur d'initialisation : {exc}")
        st.stop()

    # ── Historique du chat ──
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("route") is not None:
                route: RouteResult = msg["route"]
                st.dataframe(
                    route_to_dataframe(route), use_container_width=True, hide_index=True
                )
                render_map(route, pathfinder)

    # ── Audio (optionnel) ──
    st.divider()
    st.subheader("🎙️ Entrée audio")
    st.caption("Envoie un .wav ou utilise le micro, puis la transcription sera envoyée au chat.")

    if stt_error:
        st.warning(stt_error)
        st.info("Vérifie que `SpeechRecognition` est installé (`pip install SpeechRecognition`).")

    audio_mode = st.radio("Source audio", ["Fichier WAV", "Microphone"], horizontal=True)
    audio_prompt = None

    if audio_mode == "Fichier WAV":
        uploaded_wav = st.file_uploader("Importer un fichier .wav", type=["wav"])
        if uploaded_wav is not None:
            wav_bytes = uploaded_wav.getvalue()
            st.audio(wav_bytes, format="audio/wav")
            if st.button("Transcrire et envoyer", key="send_wav"):
                with st.spinner("Transcription en cours…"):
                    audio_prompt, error = transcribe_wav_bytes(wav_bytes, stt_module)
                if error:
                    st.error(error)
                else:
                    st.success(f"Transcription : {audio_prompt}")
    else:
        if hasattr(st, "audio_input"):
            mic_audio = st.audio_input("Enregistrer depuis le microphone")
            if mic_audio is not None:
                mic_bytes = mic_audio.getvalue()
                st.audio(mic_bytes, format="audio/wav")
                if st.button("Transcrire et envoyer", key="send_mic"):
                    with st.spinner("Transcription en cours…"):
                        audio_prompt, error = transcribe_wav_bytes(mic_bytes, stt_module)
                    if error:
                        st.error(error)
                    else:
                        st.success(f"Transcription : {audio_prompt}")
        else:
            st.info("Cette version de Streamlit ne supporte pas `st.audio_input`.")
            if st.button("Écouter le micro serveur", key="send_server_mic"):
                with st.spinner("Écoute du microphone serveur…"):
                    audio_prompt, error = transcribe_microphone(stt_module)
                if error:
                    st.error(error)
                else:
                    st.success(f"Transcription : {audio_prompt}")

    if audio_prompt:
        process_prompt(audio_prompt, ner_bundle, pathfinder)
        return

    # ── Chat input ──
    prompt = st.chat_input("Ex : Je veux aller de Nantes à Marseille en passant par Angers")
    if prompt:
        process_prompt(prompt, ner_bundle, pathfinder)


if __name__ == "__main__":
    main()
