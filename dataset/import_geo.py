import requests
import csv
import json
import os
from pathlib import Path
from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(timestamp)

fieldnames = ["nom", "latitude", "longitude", "label"]
BASE_URL = "https://geo.api.gouv.fr"
ERROR_LOG_FILE = "import_geo_errors.log"

def log_error(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} {message}")
    with open(ERROR_LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] {message}\n")

def get_coordinates(entity_type, query):
    params = {
        entity_type: query,
        'country': 'France',
        'format': 'json',
        'limit': 1,
        'countrycodes': 'fr'
    }
    
    try:
        response = requests.get(
            "https://nominatim.openstreetmap.org/search", 
            params=params, 
            headers={'User-Agent': 'GeoDataImporter/1.0'},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        if data and len(data) > 0:
            result = data[0]
            lat = result.get('lat', '')
            lon = result.get('lon', '')
            return (lat, lon)
        else:
            return (None, None)
    except requests.exceptions.RequestException as e:
        log_error(f"  Erreur lors de la recherche pour '{query}': {e}")
        return (None, None)
    except Exception as e:
        log_error(f"  Erreur inattendue pour '{query}': {e}")
        return (None, None)

def get_railway_stations():
    query = """
    [out:json][timeout:60];
    area["ISO3166-1"="FR"][admin_level=2]->.fr;
    (
        node["railway"="station"](area.fr);
        way["railway"="station"](area.fr);
        relation["public_transport"="station"](area.fr);
    );
    out center tags;
    """
    
    try:
        response = requests.post(
            "https://overpass-api.de/api/interpreter",
            data=query,
            headers={'Content-Type': 'text/plain'},
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        
        stations = []
        for element in data.get('elements', []):
            element_type = element.get('type', '')
            name = ''
            lat = None
            lon = None
            
            # Pour les éléments de type "way", récupérer depuis center et tags
            if element_type == "way":
                center = element.get('center', {})
                lat = center.get('lat')
                lon = center.get('lon')
                tags = element.get('tags', {})
                name = tags.get('name', '')
            # Pour les nodes, utiliser directement lat/lon
            elif element_type == "node":
                lat = element.get('lat')
                lon = element.get('lon')
                tags = element.get('tags', {})
                name = tags.get('name', '')
            # Pour les relations, utiliser center si disponible
            elif element_type == "relation":
                center = element.get('center', {})
                if center:
                    lat = center.get('lat')
                    lon = center.get('lon')
                tags = element.get('tags', {})
                name = tags.get('name', '')
            
            if name and lat is not None and lon is not None:
                stations.append({
                    "nom": name,
                    "latitude": lat,
                    "longitude": lon,
                    "label": "LOC_STATION"
                })
        
        return stations
    except requests.exceptions.RequestException as e:
        log_error(f"  Erreur lors de la récupération des gares: {e}")
        return []
    except Exception as e:
        log_error(f"  Erreur inattendue lors de la récupération des gares: {e}")
        return []

def get_aeroways():
    query = """
    [out:json][timeout:60];
    area["ISO3166-1"="FR"][admin_level=2]->.fr;
    (
        way["aeroway"="aerodrome"](area.fr);
        relation["aeroway"="aerodrome"](area.fr);
    );
    out center tags;
    """

    try:
        response = requests.post(
            "https://overpass-api.de/api/interpreter",
            data=query,
            headers={'Content-Type': 'text/plain'},
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        
        aeroways = []
        for element in data.get('elements', []):
            element_type = element.get('type', '')
            name = ''
            lat = None
            lon = None
            
            # Pour les éléments de type "way", récupérer depuis center et tags
            if element_type == "way":
                center = element.get('center', {})
                lat = center.get('lat')
                lon = center.get('lon')
                tags = element.get('tags', {})
                name = tags.get('name', '')
            # Pour les nodes, utiliser directement lat/lon
            elif element_type == "node":
                lat = element.get('lat')
                lon = element.get('lon')
                tags = element.get('tags', {})
                name = tags.get('name', '')
            # Pour les relations, utiliser center si disponible
            elif element_type == "relation":
                center = element.get('center', {})
                if center:
                    lat = center.get('lat')
                    lon = center.get('lon')
                tags = element.get('tags', {})
                name = tags.get('name', '')
            
            if name and lat is not None and lon is not None:
                aeroways.append({
                    "nom": name,
                    "latitude": lat,
                    "longitude": lon,
                    "label": "LOC_AEROWAY"
                })
        
        return aeroways
    except requests.exceptions.RequestException as e:
        log_error(f"  Erreur lors de la récupération des gares: {e}")
        return []
    except Exception as e:
        log_error(f"  Erreur inattendue lors de la récupération des gares: {e}")
        return []

def merge_csv_files(output_file, *input_files, encoding='utf-8'):
    stats = {
        'total_lines': 0,
        'files_processed': 0,
        'files_missing': [],
        'errors': []
    }
    
    try:
        with open(output_file, 'w', newline='', encoding=encoding) as outfile:
            header_written = False
            
            for i, input_file in enumerate(input_files, 1):
                # Vérification de l'existence du fichier
                if not os.path.exists(input_file):
                    print(f"Fichier manquant: {input_file}")
                    stats['files_missing'].append(input_file)
                    continue
                
                try:
                    with open(input_file, 'r', newline='', encoding=encoding) as infile:
                        lines = infile.readlines()
                        
                        if not lines:
                            print(f"Fichier vide: {input_file}")
                            continue
                        
                        # Écrire le header uniquement pour le premier fichier
                        if not header_written:
                            outfile.write(lines[0])
                            header_written = True
                            print(f"  ✓ Header écrit depuis: {input_file}")
                        
                        # Écrire toutes les lignes sauf le header
                        data_lines = lines[1:]
                        outfile.writelines(data_lines)
                        
                        stats['total_lines'] += len(data_lines)
                        stats['files_processed'] += 1
                        
                        print(f"  ✓ [{i}/{len(input_files)}] {input_file}: {len(data_lines)} lignes ajoutées")
                
                except Exception as e:
                    error_msg = f"Erreur lors de la lecture de {input_file}: {e}"
                    print(f"{error_msg}")
                    stats['errors'].append(error_msg)
        
        print(f"\nFusion terminée: {stats['total_lines']} lignes écrites dans '{output_file}'")
        return stats
    
    except Exception as e:
        print(f"Erreur critique lors de la fusion: {e}")
        stats['errors'].append(str(e))
        return stats

print('Récupération des régions de France')
response = requests.get(f"{BASE_URL}/regions")
regions = response.json()

with open('geo_regions.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    total_regions = len(regions)
    print(f"\nTraitement de {total_regions} régions...")
    for i, region in enumerate(regions, 1):
        nom = region["nom"]
        print(f"Région {i}/{total_regions}: {nom}")
        lat, lon = get_coordinates('state', nom)
        writer.writerow({
            "nom": nom,
            "latitude": lat if lat else "",
            "longitude": lon if lon else "",
            "label": "LOC_REGION"
        })

print('Récupération des départements de France')
response = requests.get(f"{BASE_URL}/departements")
departements = response.json()

with open('geo_departments.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    total_departements = len(departements)
    print(f"\nTraitement de {total_departements} départements...")
    for i, departement in enumerate(departements, 1):
        nom = departement["nom"]
        print(f"Département {i}/{total_departements}: {nom}")
        lat, lon = get_coordinates('county', nom)
        writer.writerow({
            "nom": nom,
            "latitude": lat if lat else "",
            "longitude": lon if lon else "",
            "label": "LOC_DEPARTMENT"
        })

response = requests.get(f"{BASE_URL}/communes")
communes = response.json()
response = requests.get(f"{BASE_URL}/communes_associees_deleguees")
communes_associees_deleguees = response.json()

with open('geo_communes.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    total_communes = len(communes)
    print(f"\nTraitement de {total_communes} communes...")
    for i, commune in enumerate(communes, 1):
        nom = commune["nom"]
        print(f"Commune {i}/{total_communes}: {nom}")
        lat, lon = get_coordinates('city', nom)
        writer.writerow({
            "nom": nom,
            "latitude": lat if lat else "",
            "longitude": lon if lon else "",
            "label": "LOC_CITY"
        })

    total_communes_associees_deleguees = len(communes_associees_deleguees)
    print(f"\nTraitement de {total_communes_associees_deleguees} communes associées déléguées...")
    for i, commune in enumerate(communes_associees_deleguees, 1):
        nom = commune["nom"]
        print(f"Commune associée déléguée {i}/{total_communes_associees_deleguees}: {nom}")
        # lat, lon = get_coordinates('city', nom)
        writer.writerow({
            "nom": nom,
            "latitude": lat if lat else "",
            "longitude": lon if lon else "",
            "label": "LOC_CITY"
        })

print('Récupération des gares SNCF')
stations = get_railway_stations()

with open('geo_railways.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    total_stations = len(stations)
    print(f"\nTraitement de {total_stations} gares SNCF...")
    for i, station in enumerate(stations, 1):
        print(f"Gare de chemin de fer {i}/{total_stations}: {station['nom']}")
        writer.writerow(station)

print('Récupération des aéroports')
aeroways = get_aeroways()

with open('geo_aeroways.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    total_aeroways = len(aeroways)
    print(f"\nTraitement de {total_aeroways} aéroports...")
    for i, aeroway in enumerate(aeroways, 1):
        print(f"Gare de chemin de fer {i}/{aeroways}: {aeroway['nom']}")
        writer.writerow(aeroway)

files_to_merge = [
    'geo_regions.csv',
    'geo_departments.csv',
    'geo_communes.csv',
    'geo_railways.csv',
    'geo_aeroways.csv'
]

stats = merge_csv_files('geo.csv', *files_to_merge)

print(f"\nStatistiques de fusion:")
print(f"Fichiers traités: {stats['files_processed']}/{len(files_to_merge)}")
print(f"Lignes totales: {stats['total_lines']:,}")

for file in files_to_merge:
    os.remove(file) 