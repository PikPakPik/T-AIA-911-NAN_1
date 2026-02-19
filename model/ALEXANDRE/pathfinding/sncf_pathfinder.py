import heapq
import math
import os
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class RouteSegment:
    from_station_id: str
    to_station_id: str
    from_station_name: str
    to_station_name: str
    time_min: float
    train_count: int


@dataclass
class RouteResult:
    station_ids: List[str]
    station_names: List[str]
    segments: List[RouteSegment]
    total_time_min: float
    requested_cities: List[str]
    matched_stations: List[str]


class SNCFPathFinder:
    """Build a railway graph from GTFS files and run Dijkstra."""

    def __init__(self, gtfs_dir: str, geo_csv_path: Optional[str] = None):
        self.gtfs_dir = gtfs_dir
        self.graph: Dict[str, List[Tuple[str, float, int]]] = defaultdict(list)
        self.edge_time_by_pair: Dict[Tuple[str, str], float] = {}
        self.edge_count_by_pair: Dict[Tuple[str, str], int] = {}

        self.area_id_to_name: Dict[str, str] = {}
        self.area_id_to_lat: Dict[str, float] = {}
        self.area_id_to_lon: Dict[str, float] = {}
        self.station_degree: Dict[str, int] = {}
        self._is_ready = False

        # Geo data for city -> nearest station fallback
        self._geo_csv_path = geo_csv_path
        self._geo_data: Optional[pd.DataFrame] = None
        self._geo_name_to_idx: Dict[str, int] = {}

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
            h = int(parts[0])
            m = int(parts[1])
            s = int(parts[2]) if len(parts) > 2 else 0
            return h * 60 + m + s / 60.0
        except (ValueError, TypeError):
            return float("nan")

    def _read_csv(self, filename: str) -> pd.DataFrame:
        full_path = os.path.join(self.gtfs_dir, filename)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Missing GTFS file: {full_path}")
        return pd.read_csv(full_path, low_memory=False)

    def build(self) -> None:
        """Load GTFS and build graph once."""
        if self._is_ready:
            return

        stops = self._read_csv("stops.txt")
        stop_times = self._read_csv("stop_times.txt")

        stop_points = stops[stops["location_type"] == 0][["stop_id", "parent_station"]].copy()
        stop_areas = stops[stops["location_type"] == 1][
            ["stop_id", "stop_name", "stop_lat", "stop_lon"]
        ].copy()

        stoppoint_to_area = dict(zip(stop_points["stop_id"], stop_points["parent_station"]))
        self.area_id_to_name = dict(zip(stop_areas["stop_id"], stop_areas["stop_name"]))
        self.area_id_to_lat = dict(zip(stop_areas["stop_id"], stop_areas["stop_lat"]))
        self.area_id_to_lon = dict(zip(stop_areas["stop_id"], stop_areas["stop_lon"]))

        stop_times = stop_times.copy()
        stop_times["parent_station"] = stop_times["stop_id"].map(stoppoint_to_area)
        stop_times["arrival_min"] = stop_times["arrival_time"].apply(self._time_to_minutes)
        stop_times["departure_min"] = stop_times["departure_time"].apply(self._time_to_minutes)
        stop_times = stop_times.dropna(subset=["parent_station", "arrival_min", "departure_min"])

        st = stop_times.sort_values(["trip_id", "stop_sequence"]).copy()
        st["prev_trip"] = st["trip_id"].shift(1)
        st["prev_station"] = st["parent_station"].shift(1)
        st["prev_departure"] = st["departure_min"].shift(1)

        consecutive = st[st["trip_id"] == st["prev_trip"]].copy()
        consecutive["travel_time"] = consecutive["arrival_min"] - consecutive["prev_departure"]
        consecutive = consecutive[consecutive["travel_time"] > 0]

        edge_stats = (
            consecutive.groupby(["prev_station", "parent_station"])
            .agg(min_time=("travel_time", "min"), nb_trains=("travel_time", "count"))
            .reset_index()
        )
        edge_stats.columns = ["station_from", "station_to", "min_time", "nb_trains"]

        for _, row in edge_stats.iterrows():
            source = row["station_from"]
            target = row["station_to"]
            min_time = float(row["min_time"])
            train_count = int(row["nb_trains"])
            self.graph[source].append((target, min_time, train_count))
            self.edge_time_by_pair[(source, target)] = min_time
            self.edge_count_by_pair[(source, target)] = train_count

            self.station_degree[source] = self.station_degree.get(source, 0) + 1
            self.station_degree[target] = self.station_degree.get(target, 0) + 1

        self._is_ready = True

    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Haversine distance in km between two (lat, lon) points."""
        R = 6371.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def _load_geo(self) -> None:
        """Load geo.csv lazily."""
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

    def find_nearest_station(self, lat: float, lon: float, top_n: int = 3) -> List[Tuple[str, str, float]]:
        """
        Find the nearest SNCF stations to given coordinates.
        Returns list of (station_id, station_name, distance_km).
        Only considers stations that are actually in the graph (have connections).
        """
        self.build()
        results: List[Tuple[str, str, float]] = []
        for station_id, station_name in self.area_id_to_name.items():
            slat = self.area_id_to_lat.get(station_id)
            slon = self.area_id_to_lon.get(station_id)
            if slat is None or slon is None or pd.isna(slat) or pd.isna(slon):
                continue
            # Only consider stations that have at least one connection
            if station_id not in self.graph and station_id not in self.station_degree:
                continue
            dist = self._haversine_km(lat, lon, float(slat), float(slon))
            results.append((station_id, station_name, dist))
        results.sort(key=lambda x: x[2])
        return results[:top_n]

    def find_nearest_station_for_city(self, city: str, top_n: int = 3) -> List[Tuple[str, str, float]]:
        """
        Look up a city in geo.csv by name, then find the nearest SNCF station(s).
        Returns list of (station_id, station_name, distance_km) or empty list.
        """
        self._load_geo()
        if self._geo_data is None or self._geo_data.empty:
            return []

        city_norm = self._normalize(city)

        # Exact match
        idx = self._geo_name_to_idx.get(city_norm)
        if idx is not None:
            row = self._geo_data.loc[idx]
            return self.find_nearest_station(float(row["latitude"]), float(row["longitude"]), top_n)

        # Partial / fuzzy match: find cities that contain the query or vice versa
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

    def find_station_candidates(self, query: str, top_n: int = 5) -> List[Tuple[str, str, float]]:
        """Return list of (station_id, station_name, score)."""
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
                rest = station_norm[len(query_norm) :]
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

    def dijkstra(self, start_id: str, end_id: str) -> Tuple[Optional[List[str]], float]:
        self.build()
        if start_id == end_id:
            return [start_id], 0.0

        counter = 0
        heap = [(0.0, counter, start_id)]
        distances = {start_id: 0.0}
        previous = {start_id: None}
        visited = set()

        while heap:
            current_dist, _, node = heapq.heappop(heap)
            if node in visited:
                continue
            visited.add(node)

            if node == end_id:
                path = []
                cur = node
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

    def _pick_candidates(self, city: str) -> List[Tuple[str, str, float]]:
        candidates = self.find_station_candidates(city, top_n=5)
        if candidates:
            min_score = candidates[0][2] * 0.7
            filtered = [item for item in candidates if item[2] >= min_score][:3]
            if filtered:
                return filtered

        # Fallback: look up city in geo.csv and find the nearest station
        geo_candidates = self.find_nearest_station_for_city(city, top_n=3)
        if geo_candidates:
            # Convert (station_id, station_name, distance_km) to scoring format
            # Use a synthetic score: closer = higher score, cap at 100km
            scored = []
            for station_id, station_name, dist_km in geo_candidates:
                score = max(0.1, 1.0 - dist_km / 100.0)
                scored.append((station_id, station_name, score))
            return scored

        return []

    def route_between_cities(self, city_from: str, city_to: str) -> RouteResult:
        self.build()
        starts = self._pick_candidates(city_from)
        ends = self._pick_candidates(city_to)

        if not starts:
            raise ValueError(f"Aucune gare trouvée pour '{city_from}' (ni par nom, ni par géolocalisation).")
        if not ends:
            raise ValueError(f"Aucune gare trouvée pour '{city_to}' (ni par nom, ni par géolocalisation).")

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
            raise ValueError(f"No route found between '{city_from}' and '{city_to}'.")

        segments: List[RouteSegment] = []
        for idx in range(len(best_path) - 1):
            src = best_path[idx]
            dst = best_path[idx + 1]
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

        station_names = [self.area_id_to_name.get(st_id, st_id) for st_id in best_path]
        return RouteResult(
            station_ids=best_path,
            station_names=station_names,
            segments=segments,
            total_time_min=best_time,
            requested_cities=[city_from, city_to],
            matched_stations=[best_start_name, best_end_name],
        )

    def route_with_waypoints(self, cities: List[str]) -> RouteResult:
        if len(cities) < 2:
            raise ValueError("Need at least 2 cities.")

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

        station_names = [self.area_id_to_name.get(st_id, st_id) for st_id in global_station_ids]
        return RouteResult(
            station_ids=global_station_ids,
            station_names=station_names,
            segments=global_segments,
            total_time_min=total_time,
            requested_cities=cities,
            matched_stations=matched_stations,
        )

    def station_coordinates(self, station_id: str) -> Optional[Tuple[float, float]]:
        lat = self.area_id_to_lat.get(station_id)
        lon = self.area_id_to_lon.get(station_id)
        if pd.isna(lat) or pd.isna(lon):
            return None
        return float(lat), float(lon)
