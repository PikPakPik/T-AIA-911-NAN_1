"""
Pathfinder SNCF — Graphe GTFS, Dijkstra, horaires de trains.

Charge les fichiers GTFS (stops, stop_times, trips, routes), construit un
graphe pondéré par le temps de trajet minimum, et propose :
  - Recherche de gares par nom ou géolocalisation
  - Plus court chemin (Dijkstra) entre gares
  - Itinéraire multi-étapes (waypoints)
  - Horaires concrets avec correspondances
"""

import heapq
import math
import os
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ─── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class RouteSegment:
    """Un tronçon entre deux gares consécutives."""
    from_station_id: str
    to_station_id: str
    from_station_name: str
    to_station_name: str
    time_min: float
    train_count: int


@dataclass
class RouteResult:
    """Résultat complet d'un itinéraire."""
    station_ids: List[str]
    station_names: List[str]
    segments: List[RouteSegment]
    total_time_min: float
    requested_cities: List[str]
    matched_stations: List[str]


@dataclass
class ScheduleStop:
    """Un arrêt dans un horaire concret."""
    station_id: str
    station_name: str
    arrival_time: str
    departure_time: str


@dataclass
class TrainLeg:
    """Un trajet continu sur un même train."""
    trip_id: str
    train_number: str
    train_type: str
    stops: List[ScheduleStop]
    dep_min: float
    arr_min: float


@dataclass
class ConnectionInfo:
    """Information sur une correspondance entre deux trains."""
    station_name: str
    arrive_time: str
    depart_time: str
    wait_min: float


@dataclass
class ScheduleResult:
    """Horaire complet avec legs et correspondances."""
    legs: List[TrainLeg]
    connections: List[ConnectionInfo]
    total_time_min: float
    total_wait_min: float


# ─── Classe principale ───────────────────────────────────────────────────────

class SNCFPathFinder:
    """
    Construit un graphe ferroviaire à partir de fichiers GTFS et exécute
    Dijkstra pour trouver le plus court chemin.
    """

    def __init__(self, gtfs_dir: str, geo_csv_path: Optional[str] = None):
        self.gtfs_dir = gtfs_dir

        # Graphe : station_id -> [(voisin_id, temps_min, nb_trains)]
        self.graph: Dict[str, List[Tuple[str, float, int]]] = defaultdict(list)
        self.edge_time_by_pair: Dict[Tuple[str, str], float] = {}
        self.edge_count_by_pair: Dict[Tuple[str, str], int] = {}

        # Référentiel gares
        self.area_id_to_name: Dict[str, str] = {}
        self.area_id_to_lat: Dict[str, float] = {}
        self.area_id_to_lon: Dict[str, float] = {}
        self.station_degree: Dict[str, int] = {}

        self._is_ready = False

        # Données horaires
        self._trip_headsign: Dict[str, str] = {}
        self._trip_schedule: Dict[str, List[Tuple[str, float, float]]] = {}
        self._trip_type: Dict[str, str] = {}
        self._edge_trips: Dict[Tuple[str, str], List[Tuple[str, float, float]]] = defaultdict(list)
        self._station_trips: Dict[str, List[Tuple[str, int, float]]] = defaultdict(list)

        # Géolocalisation (fallback pour villes sans gare)
        self._geo_csv_path = geo_csv_path
        self._geo_data: Optional[pd.DataFrame] = None
        self._geo_name_to_idx: Dict[str, int] = {}

    # ─── Utilitaires internes ─────────────────────────────────────────────

    @staticmethod
    def _normalize(text: str) -> str:
        raw = text.lower().strip()
        no_accents = "".join(
            c for c in unicodedata.normalize("NFD", raw) if unicodedata.category(c) != "Mn"
        )
        return re.sub(r"\s+", " ", no_accents)

    @staticmethod
    def _time_to_minutes(time_str: str) -> float:
        parts = str(time_str).strip().split(":")
        if len(parts) < 2:
            return float("nan")
        try:
            h, m = int(parts[0]), int(parts[1])
            s = int(parts[2]) if len(parts) > 2 else 0
            return h * 60 + m + s / 60.0
        except (ValueError, TypeError):
            return float("nan")

    @staticmethod
    def _minutes_to_hhmm(minutes: float) -> str:
        total = int(round(minutes))
        h = (total // 60) % 24
        m = total % 60
        return f"{h:02d}:{m:02d}"

    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Distance de Haversine en km entre deux points (lat, lon)."""
        R = 6371.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def _read_csv(self, filename: str) -> pd.DataFrame:
        full_path = os.path.join(self.gtfs_dir, filename)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Fichier GTFS manquant : {full_path}")
        return pd.read_csv(full_path, low_memory=False)

    # ─── Construction du graphe ───────────────────────────────────────────

    def build(self) -> None:
        """Charge les fichiers GTFS et construit le graphe (une seule fois)."""
        if self._is_ready:
            return

        stops = self._read_csv("stops.txt")
        stop_times = self._read_csv("stop_times.txt")

        # Séparer stop points et stop areas
        stop_points = stops[stops["location_type"] == 0][["stop_id", "parent_station"]].copy()
        stop_areas = stops[stops["location_type"] == 1][
            ["stop_id", "stop_name", "stop_lat", "stop_lon"]
        ].copy()

        stoppoint_to_area = dict(zip(stop_points["stop_id"], stop_points["parent_station"]))
        self.area_id_to_name = dict(zip(stop_areas["stop_id"], stop_areas["stop_name"]))
        self.area_id_to_lat = dict(zip(stop_areas["stop_id"], stop_areas["stop_lat"]))
        self.area_id_to_lon = dict(zip(stop_areas["stop_id"], stop_areas["stop_lon"]))

        # Préparer stop_times avec parent station et temps en minutes
        stop_times = stop_times.copy()
        stop_times["parent_station"] = stop_times["stop_id"].map(stoppoint_to_area)
        stop_times["arrival_min"] = stop_times["arrival_time"].apply(self._time_to_minutes)
        stop_times["departure_min"] = stop_times["departure_time"].apply(self._time_to_minutes)
        stop_times = stop_times.dropna(subset=["parent_station", "arrival_min", "departure_min"])

        # Arrêts consécutifs dans un même trip
        st = stop_times.sort_values(["trip_id", "stop_sequence"]).copy()
        st["prev_trip"] = st["trip_id"].shift(1)
        st["prev_station"] = st["parent_station"].shift(1)
        st["prev_departure"] = st["departure_min"].shift(1)

        consecutive = st[st["trip_id"] == st["prev_trip"]].copy()
        consecutive["travel_time"] = consecutive["arrival_min"] - consecutive["prev_departure"]
        consecutive = consecutive[consecutive["travel_time"] > 0]

        # Agréger par paire de gares
        edge_stats = (
            consecutive.groupby(["prev_station", "parent_station"])
            .agg(min_time=("travel_time", "min"), nb_trains=("travel_time", "count"))
            .reset_index()
        )
        edge_stats.columns = ["station_from", "station_to", "min_time", "nb_trains"]

        for _, row in edge_stats.iterrows():
            source, target = row["station_from"], row["station_to"]
            min_time = float(row["min_time"])
            train_count = int(row["nb_trains"])
            self.graph[source].append((target, min_time, train_count))
            self.edge_time_by_pair[(source, target)] = min_time
            self.edge_count_by_pair[(source, target)] = train_count
            self.station_degree[source] = self.station_degree.get(source, 0) + 1
            self.station_degree[target] = self.station_degree.get(target, 0) + 1

        # ── Données horaires (trips, types de train, edge_trips) ──
        trips_df = self._read_csv("trips.txt")
        self._trip_headsign = dict(
            zip(trips_df["trip_id"], trips_df["trip_headsign"].astype(str))
        )

        trip_data = st[
            ["trip_id", "parent_station", "arrival_min", "departure_min", "stop_id"]
        ].values

        cur_tid: Optional[str] = None
        cur_sched: List[Tuple[str, float, float]] = []

        def _flush_trip(tid: str, sched: List[Tuple[str, float, float]]) -> None:
            self._trip_schedule[tid] = sched
            for k in range(len(sched) - 1):
                s0, s1 = sched[k], sched[k + 1]
                if s1[1] - s0[2] > 0:
                    self._edge_trips[(s0[0], s1[0])].append((tid, s0[2], s1[1]))

        for row in trip_data:
            tid = str(row[0])
            ps, am, dm, sid = str(row[1]), float(row[2]), float(row[3]), str(row[4])
            if tid != cur_tid:
                if cur_tid is not None and cur_sched:
                    _flush_trip(cur_tid, cur_sched)
                cur_tid = tid
                cur_sched = []
                m = re.match(r"StopPoint:OCE(.+)-\d+", sid)
                if m:
                    self._trip_type[tid] = m.group(1)
            cur_sched.append((ps, am, dm))

        if cur_tid is not None and cur_sched:
            _flush_trip(cur_tid, cur_sched)

        for key in self._edge_trips:
            self._edge_trips[key].sort(key=lambda x: x[1])

        # Index inverse station → trips départs
        for trip_id, sched in self._trip_schedule.items():
            for idx, (station, _arr, dep) in enumerate(sched):
                self._station_trips[station].append((trip_id, idx, dep))
        for key in self._station_trips:
            self._station_trips[key].sort(key=lambda x: x[2])

        self._is_ready = True

    # ─── Géolocalisation (fallback) ───────────────────────────────────────

    def _load_geo(self) -> None:
        """Charge geo.csv paresseusement."""
        if self._geo_data is not None:
            return
        if self._geo_csv_path is None or not os.path.exists(self._geo_csv_path):
            self._geo_data = pd.DataFrame()
            return
        df = pd.read_csv(self._geo_csv_path)
        df = df.dropna(subset=["latitude", "longitude"])
        self._geo_data = df
        for idx, row in df.iterrows():
            key = self._normalize(str(row["nom"]))
            self._geo_name_to_idx[key] = idx

    def find_nearest_station(
        self, lat: float, lon: float, top_n: int = 3
    ) -> List[Tuple[str, str, float]]:
        """
        Trouve les gares SNCF les plus proches de coordonnées données.
        Retourne [(station_id, station_name, distance_km)].
        """
        self.build()
        results: List[Tuple[str, str, float]] = []
        for station_id, station_name in self.area_id_to_name.items():
            slat = self.area_id_to_lat.get(station_id)
            slon = self.area_id_to_lon.get(station_id)
            if slat is None or slon is None or pd.isna(slat) or pd.isna(slon):
                continue
            if station_id not in self.graph and station_id not in self.station_degree:
                continue
            dist = self._haversine_km(lat, lon, float(slat), float(slon))
            results.append((station_id, station_name, dist))
        results.sort(key=lambda x: x[2])
        return results[:top_n]

    def find_nearest_station_for_city(
        self, city: str, top_n: int = 3
    ) -> List[Tuple[str, str, float]]:
        """
        Cherche une ville dans geo.csv, puis trouve la gare SNCF la plus proche.
        Retourne [(station_id, station_name, distance_km)] ou [].
        """
        self._load_geo()
        if self._geo_data is None or self._geo_data.empty:
            return []

        city_norm = self._normalize(city)

        # Match exact
        idx = self._geo_name_to_idx.get(city_norm)
        if idx is not None:
            row = self._geo_data.loc[idx]
            return self.find_nearest_station(float(row["latitude"]), float(row["longitude"]), top_n)

        # Match partiel
        best_match = None
        best_score = 0.0
        for norm_name, idx in self._geo_name_to_idx.items():
            if city_norm in norm_name:
                score = len(city_norm) / len(norm_name)
                if score > best_score:
                    best_score = score
                    best_match = idx
            elif norm_name in city_norm:
                score = len(norm_name) / len(city_norm) * 0.9
                if score > best_score:
                    best_score = score
                    best_match = idx

        if best_match is not None and best_score > 0.5:
            row = self._geo_data.loc[best_match]
            return self.find_nearest_station(float(row["latitude"]), float(row["longitude"]), top_n)

        return []

    # ─── Recherche de gares ───────────────────────────────────────────────

    def find_station_candidates(
        self, query: str, top_n: int = 5
    ) -> List[Tuple[str, str, float]]:
        """
        Recherche des gares correspondant à une requête textuelle.
        Retourne [(station_id, station_name, score)].
        """
        self.build()
        query_norm = self._normalize(query)
        results: List[Tuple[str, str, float]] = []

        for station_id, station_name in self.area_id_to_name.items():
            station_norm = self._normalize(station_name)
            degree_bonus = min(0.04, self.station_degree.get(station_id, 0) / 1000.0)

            if query_norm == station_norm:
                results.append((station_id, station_name, 1.0 + degree_bonus))
                continue
            if station_norm.startswith(query_norm):
                rest = station_norm[len(query_norm):]
                if rest == "":
                    score = 1.0
                elif rest.startswith(" "):
                    score = 0.93
                elif rest.startswith("-"):
                    score = 0.80
                else:
                    score = 0.75
                results.append((station_id, station_name, score + degree_bonus))
                continue
            if query_norm in station_norm:
                results.append((station_id, station_name, 0.70 + degree_bonus))

        results.sort(key=lambda item: item[2], reverse=True)
        return results[:top_n]

    # ─── Dijkstra ─────────────────────────────────────────────────────────

    def dijkstra(self, start_id: str, end_id: str) -> Tuple[Optional[List[str]], float]:
        """
        Plus court chemin entre deux gares (Dijkstra).
        Retourne (liste d'IDs, temps total en minutes) ou (None, inf).
        """
        self.build()
        if start_id == end_id:
            return [start_id], 0.0

        counter = 0
        heap = [(0.0, counter, start_id)]
        distances = {start_id: 0.0}
        previous: Dict[str, Optional[str]] = {start_id: None}
        visited: set = set()

        while heap:
            current_dist, _, node = heapq.heappop(heap)
            if node in visited:
                continue
            visited.add(node)

            if node == end_id:
                path: List[str] = []
                cur: Optional[str] = node
                while cur is not None:
                    path.append(cur)
                    cur = previous[cur]
                path.reverse()
                return path, current_dist

            for neigh, weight, _ in self.graph.get(node, []):
                if neigh in visited:
                    continue
                alt = current_dist + weight
                if alt < distances.get(neigh, float("inf")):
                    distances[neigh] = alt
                    previous[neigh] = node
                    counter += 1
                    heapq.heappush(heap, (alt, counter, neigh))

        return None, float("inf")

    # ─── Routage ──────────────────────────────────────────────────────────

    def _pick_candidates(self, city: str) -> List[Tuple[str, str, float]]:
        """Trouve les meilleures gares candidates pour une ville."""
        candidates = self.find_station_candidates(city, top_n=5)
        if candidates:
            min_score = candidates[0][2] * 0.7
            filtered = [item for item in candidates if item[2] >= min_score][:3]
            if filtered:
                return filtered

        # Fallback géolocalisation
        geo_candidates = self.find_nearest_station_for_city(city, top_n=3)
        if geo_candidates:
            scored = []
            for station_id, station_name, dist_km in geo_candidates:
                score = max(0.1, 1.0 - dist_km / 100.0)
                scored.append((station_id, station_name, score))
            return scored

        return []

    def route_between_cities(self, city_from: str, city_to: str) -> RouteResult:
        """Calcule le plus court chemin entre deux villes."""
        self.build()
        starts = self._pick_candidates(city_from)
        ends = self._pick_candidates(city_to)

        if not starts:
            raise ValueError(
                f"Aucune gare trouvée pour '{city_from}' (ni par nom, ni par géolocalisation)."
            )
        if not ends:
            raise ValueError(
                f"Aucune gare trouvée pour '{city_to}' (ni par nom, ni par géolocalisation)."
            )

        best_path = None
        best_time = float("inf")
        best_start_name = ""
        best_end_name = ""

        for start_id, start_name, _ in starts:
            for end_id, end_name, _ in ends:
                if start_id == end_id:
                    continue
                path, total_time = self.dijkstra(start_id, end_id)
                if path and total_time < best_time:
                    best_path = path
                    best_time = total_time
                    best_start_name = start_name
                    best_end_name = end_name

        if not best_path:
            raise ValueError(f"Aucun itinéraire trouvé entre '{city_from}' et '{city_to}'.")

        segments: List[RouteSegment] = []
        for idx in range(len(best_path) - 1):
            src, dst = best_path[idx], best_path[idx + 1]
            segments.append(
                RouteSegment(
                    from_station_id=src,
                    to_station_id=dst,
                    from_station_name=self.area_id_to_name.get(src, src),
                    to_station_name=self.area_id_to_name.get(dst, dst),
                    time_min=self.edge_time_by_pair.get((src, dst), 0.0),
                    train_count=self.edge_count_by_pair.get((src, dst), 0),
                )
            )

        station_names = [self.area_id_to_name.get(sid, sid) for sid in best_path]
        return RouteResult(
            station_ids=best_path,
            station_names=station_names,
            segments=segments,
            total_time_min=best_time,
            requested_cities=[city_from, city_to],
            matched_stations=[best_start_name, best_end_name],
        )

    def route_with_waypoints(self, cities: List[str]) -> RouteResult:
        """Calcule un itinéraire passant par plusieurs villes (waypoints)."""
        if len(cities) < 2:
            raise ValueError("Il faut au moins 2 villes.")

        global_station_ids: List[str] = []
        global_segments: List[RouteSegment] = []
        matched_stations: List[str] = []
        total_time = 0.0

        for i in range(len(cities) - 1):
            partial = self.route_between_cities(cities[i], cities[i + 1])
            total_time += partial.total_time_min

            if i == 0:
                global_station_ids.extend(partial.station_ids)
            else:
                global_station_ids.extend(partial.station_ids[1:])

            global_segments.extend(partial.segments)
            if i == 0:
                matched_stations.append(partial.matched_stations[0])
            matched_stations.append(partial.matched_stations[1])

        station_names = [self.area_id_to_name.get(sid, sid) for sid in global_station_ids]
        return RouteResult(
            station_ids=global_station_ids,
            station_names=station_names,
            segments=global_segments,
            total_time_min=total_time,
            requested_cities=cities,
            matched_stations=matched_stations,
        )

    def station_coordinates(self, station_id: str) -> Optional[Tuple[float, float]]:
        """Retourne (lat, lon) d'une gare, ou None."""
        lat = self.area_id_to_lat.get(station_id)
        lon = self.area_id_to_lon.get(station_id)
        if lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
            return None
        return float(lat), float(lon)

    # ─── Horaires concrets ────────────────────────────────────────────────

    def find_schedule_for_route(
        self,
        station_ids: List[str],
        min_departure: float = 480.0,
    ) -> Optional[ScheduleResult]:
        """
        Trouve un horaire concret pour un itinéraire donné.

        Algorithme à couverture : pour chaque gare du parcours, cherche le
        train couvrant le plus de gares suivantes (évite les correspondances
        inutiles).

        Args:
            station_ids: liste ordonnée des parent-station IDs.
            min_departure: heure de départ min en minutes (480 = 08h00).

        Returns:
            ScheduleResult avec legs, correspondances, durées — ou None.
        """
        self.build()
        if len(station_ids) < 2:
            return None

        legs: List[TrainLeg] = []
        connections: List[ConnectionInfo] = []
        current_time = min_departure
        i = 0

        while i < len(station_ids) - 1:
            from_st = station_ids[i]

            # Trains partant de cette gare après current_time
            from_trips = self._station_trips.get(from_st, [])
            valid = [(tid, sidx, dep) for tid, sidx, dep in from_trips if dep >= current_time]
            if not valid:
                valid = from_trips[:20]  # fallback : premiers départs du jour
            if not valid:
                return None

            # Sélectionner le train couvrant le plus de gares du parcours
            best_trip_id: Optional[str] = None
            best_coverage = 0
            best_dep = float("inf")
            best_start_idx = 0

            for tid, sidx, dep in valid:
                trip_sched = self._trip_schedule.get(tid, [])
                cov = 0
                pj = i + 1
                tj = sidx + 1
                while pj < len(station_ids) and tj < len(trip_sched):
                    if trip_sched[tj][0] == station_ids[pj]:
                        cov += 1
                        pj += 1
                    tj += 1

                if cov > 0 and (
                    cov > best_coverage or (cov == best_coverage and dep < best_dep)
                ):
                    best_trip_id = tid
                    best_coverage = cov
                    best_dep = dep
                    best_start_idx = sidx

            if best_trip_id is None:
                return None

            trip_sched = self._trip_schedule.get(best_trip_id, [])
            dep_min = best_dep

            # Enregistrer la correspondance si ce n'est pas le premier leg
            if legs:
                wait = max(0.0, dep_min - current_time)
                connections.append(ConnectionInfo(
                    station_name=self.area_id_to_name.get(from_st, from_st),
                    arrive_time=self._minutes_to_hhmm(current_time),
                    depart_time=self._minutes_to_hhmm(dep_min),
                    wait_min=round(wait, 1),
                ))

            # Construire les arrêts du leg
            leg_stops: List[ScheduleStop] = []
            ts = trip_sched[best_start_idx]
            leg_stops.append(ScheduleStop(
                station_id=ts[0],
                station_name=self.area_id_to_name.get(ts[0], ts[0]),
                arrival_time=self._minutes_to_hhmm(ts[1]),
                departure_time=self._minutes_to_hhmm(ts[2]),
            ))

            j = i + 1
            tidx = best_start_idx + 1
            last_arr_min = dep_min

            while j < len(station_ids) and tidx < len(trip_sched):
                if trip_sched[tidx][0] == station_ids[j]:
                    ts = trip_sched[tidx]
                    leg_stops.append(ScheduleStop(
                        station_id=ts[0],
                        station_name=self.area_id_to_name.get(ts[0], ts[0]),
                        arrival_time=self._minutes_to_hhmm(ts[1]),
                        departure_time=self._minutes_to_hhmm(ts[2]),
                    ))
                    last_arr_min = ts[1]
                    j += 1
                tidx += 1

            if len(leg_stops) < 2:
                return None

            train_number = self._trip_headsign.get(best_trip_id, "")
            train_type = self._trip_type.get(best_trip_id, "Train")

            legs.append(TrainLeg(
                trip_id=best_trip_id,
                train_number=train_number,
                train_type=train_type,
                stops=leg_stops,
                dep_min=dep_min,
                arr_min=last_arr_min,
            ))

            current_time = last_arr_min
            i = j

        if not legs:
            return None

        total_time = legs[-1].arr_min - legs[0].dep_min
        total_wait = sum(c.wait_min for c in connections)

        return ScheduleResult(
            legs=legs,
            connections=connections,
            total_time_min=round(total_time, 1),
            total_wait_min=round(total_wait, 1),
        )
