#!/usr/bin/env python3
"""
Convertit claude_data.txt (format: id,"sentence",origin,destination)
en deux fichiers séparés :
  - manual_input.txt  : sentenceID,phrase
  - manual_output.txt : sentenceID,origin,destination
"""

import csv
import os
import sys

# --- Configuration ---
# Modifie ces chemins si besoin
INPUT_FILE = 'claude_data.txt'
OUTPUT_DIR = '.'  # même dossier
INPUT_OUT = os.path.join(OUTPUT_DIR, 'manual_input.txt')
OUTPUT_OUT = os.path.join(OUTPUT_DIR, 'manual_output.txt')


def main():
    # Permet de passer le fichier en argument : python manual_data_format.py mon_fichier.txt
    input_file = sys.argv[1] if len(sys.argv) > 1 else INPUT_FILE

    if not os.path.exists(input_file):
        print(f"ERREUR : fichier '{input_file}' introuvable !")
        print(f"Fichiers dans le dossier courant : {os.listdir('.')}")
        sys.exit(1)

    count_valid = 0
    count_invalid = 0
    count_total = 0

    with open(input_file, 'r', encoding='utf-8') as f, \
         open(INPUT_OUT, 'w', encoding='utf-8') as f_in, \
         open(OUTPUT_OUT, 'w', encoding='utf-8') as f_out:

        reader = csv.reader(f)

        # Détecter et sauter le header s'il existe
        first_row = next(reader)
        header_keywords = ['id', 'sentence', 'origin', 'destination']
        is_header = any(kw in col.lower() for col in first_row for kw in header_keywords)

        if not is_header:
            # C'est pas un header, c'est déjà une ligne de données -> on la traite
            rows_to_process = [first_row]
        else:
            print(f"Header détecté et ignoré : {first_row}")
            rows_to_process = []

        # Traiter toutes les lignes
        for row in list(rows_to_process) + list(reader):
            count_total += 1

            # Gérer les lignes avec nombre de colonnes variable
            if len(row) < 3:
                print(f"  WARN ligne {count_total}: pas assez de colonnes -> {row}")
                continue

            sid = row[0].strip()
            sentence = row[1].strip().strip('"')
            origin = row[2].strip() if len(row) > 2 else ''
            dest = row[3].strip() if len(row) > 3 else ''

            # Nettoyer les virgules dans la phrase pour le format CSV simple
            sentence_clean = sentence.replace(',', ' ')

            # Écrire l'input
            f_in.write(f"{sid},{sentence_clean}\n")

            # Écrire l'output
            if origin.upper() == 'INVALID' or origin == '':
                f_out.write(f"{sid},INVALID,\n")
                count_invalid += 1
            else:
                f_out.write(f"{sid},{origin},{dest}\n")
                count_valid += 1

    print(f"\n{'='*50}")
    print(f" CONVERSION TERMINÉE")
    print(f"{'='*50}")
    print(f" Fichier source    : {input_file}")
    print(f" Total lignes      : {count_total}")
    print(f" Valides (O -> D)  : {count_valid}")
    print(f" Invalides         : {count_invalid}")
    print(f"{'='*50}")
    print(f" -> {INPUT_OUT}  ({count_total} phrases)")
    print(f" -> {OUTPUT_OUT} ({count_total} entrées)")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()