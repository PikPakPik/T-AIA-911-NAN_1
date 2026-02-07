import csv
import math
from typing import List, Dict, Tuple, Optional

class PathFinder:
    """Classe pour le calcul de chemins optimaux entre lieux géographiques"""
    
    def __init__(self, geo_csv_path: str = "./dataset/geo.csv"):
        self.locations = {}  # {nom: {lat, lon, label}}
        self.load_locations(geo_csv_path)
    
    def load_locations(self, csv_path: str):
        """Charge les lieux depuis geo.csv"""
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                loaded = 0
                skipped = 0
                for row in reader:
                    name = row['nom']
                    lat_str = row.get('latitude', '').strip()
                    lon_str = row.get('longitude', '').strip()
                    
                    # Ignorer les lignes sans coordonnées
                    if not lat_str or not lon_str:
                        skipped += 1
                        continue
                    
                    try:
                        lat = float(lat_str)
                        lon = float(lon_str)
                        
                        self.locations[name.lower()] = {
                            'name': name,
                            'lat': lat,
                            'lon': lon,
                            'label': row.get('label', '')
                        }
                        loaded += 1
                    except ValueError:
                        skipped += 1
                        continue
                
            print(f"✓ {loaded} lieux chargés depuis {csv_path} ({skipped} ignorés - coordonnées manquantes)")
        except Exception as e:
            print(f"Erreur lors du chargement de {csv_path}: {e}")
            self.locations = {}
    
    def haversine_distance(self, loc1: str, loc2: str) -> float:
        """Calcule la distance en km entre deux lieux (formule de Haversine)"""
        loc1_lower = loc1.lower()
        loc2_lower = loc2.lower()
        
        if loc1_lower not in self.locations or loc2_lower not in self.locations:
            return float('inf')
        
        lat1 = math.radians(self.locations[loc1_lower]['lat'])
        lon1 = math.radians(self.locations[loc1_lower]['lon'])
        lat2 = math.radians(self.locations[loc2_lower]['lat'])
        lon2 = math.radians(self.locations[loc2_lower]['lon'])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Rayon de la Terre en km
        R = 6371
        return R * c
    
    def find_closest_match(self, location_name: str) -> Optional[str]:
        """Trouve la correspondance la plus proche dans geo.csv"""
        if not location_name:
            return None
            
        location_lower = location_name.lower().strip()
        
        # Correspondance exacte
        if location_lower in self.locations:
            return self.locations[location_lower]['name']
        
        # Recherche partielle (le lieu cherché contient ou est contenu dans geo.csv)
        for loc_key, loc_data in self.locations.items():
            if location_lower in loc_key or loc_key in location_lower:
                return loc_data['name']
        
        return None
    
    def calculate_route_distance(self, locations: List[str]) -> Tuple[float, List[Tuple[str, str, float]]]:
        """
        Calcule la distance totale d'un itinéraire
        Returns: (distance_totale, [(départ, arrivée, distance), ...])
        """
        if len(locations) < 2:
            return 0.0, []
        
        total_distance = 0.0
        segments = []
        
        for i in range(len(locations) - 1):
            loc1 = self.find_closest_match(locations[i])
            loc2 = self.find_closest_match(locations[i + 1])
            
            if loc1 and loc2:
                distance = self.haversine_distance(loc1, loc2)
                total_distance += distance
                segments.append((loc1, loc2, distance))
            else:
                # Lieu non trouvé dans geo.csv
                segments.append((locations[i], locations[i + 1], float('inf')))
        
        return total_distance, segments
    
    def find_shortest_path(self, start: str, end: str, waypoints: List[str] = None) -> Dict:
        """
        Trouve le chemin le plus court entre start et end
        avec des points de passage optionnels (waypoints)
        
        Returns: {
            'path': [lieux],
            'distance': float,
            'segments': [(lieu1, lieu2, distance), ...],
            'valid': bool
        }
        """
        if waypoints is None:
            waypoints = []
        
        # Ordre des lieux à visiter
        route = [start] + waypoints + [end]
        
        # Normaliser les noms
        normalized_route = []
        unmatched = []
        
        for loc in route:
            matched = self.find_closest_match(loc)
            if matched:
                normalized_route.append(matched)
            else:
                normalized_route.append(loc)
                unmatched.append(loc)
        
        total_distance, segments = self.calculate_route_distance(normalized_route)
        
        return {
            'path': normalized_route,
            'distance': total_distance,
            'segments': segments,
            'valid': len(unmatched) == 0,
            'unmatched': unmatched
        }
    
    def suggest_optimal_order(self, locations: List[str]) -> Dict:
        """
        Suggère l'ordre optimal pour visiter tous les lieux (TSP simplifié)
        Utilise une approche greedy (plus proche voisin)
        """
        if len(locations) < 2:
            return {
                'path': locations, 
                'distance': 0.0, 
                'segments': [],
                'optimization': 'none',
                'valid': True
            }
        
        # Normaliser les noms
        normalized = []
        for loc in locations:
            matched = self.find_closest_match(loc)
            normalized.append(matched if matched else loc)
        
        # Algorithme du plus proche voisin
        unvisited = normalized[1:]
        current = normalized[0]
        path = [current]
        
        while unvisited:
            closest = min(unvisited, key=lambda x: self.haversine_distance(current, x))
            path.append(closest)
            unvisited.remove(closest)
            current = closest
        
        total_distance, segments = self.calculate_route_distance(path)
        
        return {
            'path': path,
            'distance': total_distance,
            'segments': segments,
            'optimization': 'nearest_neighbor',
            'valid': total_distance != float('inf')
        }