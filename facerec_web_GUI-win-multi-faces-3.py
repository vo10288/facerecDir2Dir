# 20250924 - Versione modificata per gestire più volti per immagine
# https://tsurugi-linux.org
# by Visi@n - Modified for multiple face detection
# LICENSE
# THIS SCRIPT USE FACE_RECOGNITION LIBRARY [https://github.com/ageitgey/face_recognition/blob/master/LICENSE]
# THIS SCRIPT HAS BEEN MODIFIED BY Antonio 'Visi@n' Broi [antonio@tsurugi-linux.org] and it's licensed under the MIT License
# Special thanks to Adam Ageitgey [https://adamgeitgey.com] the creator of face_recognition and to all Python community

import tkinter as tk
import sys
import os
import subprocess
import time
import glob
import shutil
from pathlib import Path
import webbrowser
import face_recognition
import cv2
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont
import dlib
from imutils import face_utils
import platform

class MultiFaceRecognitionApp:
    def __init__(self):
        self.home_dir = os.path.expanduser('~')
        self.work_dir = os.path.join(self.home_dir, '02.computer_vision')
        self.known_sources = os.path.join(self.work_dir, '01.known_sources')
        self.unknown_targets = os.path.join(self.work_dir, '02.unknown_targets')
        self.reports_dir = os.path.join(self.work_dir, '03.reports')
        self.tmp_dir = os.path.join(self.work_dir, '99.TMP')
        self.landmarks_dir = os.path.join(self.reports_dir, 'landmarks')
        self.recondbase_dir = os.path.join(self.reports_dir, 'recondbase')
        self.recontarget_dir = os.path.join(self.reports_dir, 'recontarget')
        self.originals_known_dir = os.path.join(self.reports_dir, 'originals_known')
        self.originals_targets_dir = os.path.join(self.reports_dir, 'originals_targets')
        
        # Crea tutte le directory necessarie
        for directory in [self.landmarks_dir, self.recondbase_dir, self.recontarget_dir, 
                         self.originals_known_dir, self.originals_targets_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Inizializza il detector e predictor di dlib
        self.detector = dlib.get_frontal_face_detector()
        
        # Carica il modello dalla directory dello script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        shape_predictor_path = os.path.join(script_dir, 'shape_predictor_68_face_landmarks.dat')
        
        # Verifica che il file esista
        if not os.path.exists(shape_predictor_path):
            print(f"[ERROR] Modello non trovato: {shape_predictor_path}")
            print("[INFO] Assicurati che il file 'shape_predictor_68_face_landmarks.dat' sia nella stessa directory dello script")
            print(f"[INFO] Directory dello script: {script_dir}")
            raise FileNotFoundError(f"Il file {shape_predictor_path} non è stato trovato")
        
        print(f"[INFO] Caricamento modello da: {shape_predictor_path}")
        self.predictor = dlib.shape_predictor(shape_predictor_path)
        
    def filename_normalization(self):
        """Rinomina i file sostituendo gli spazi con underscore"""
        for directory in [self.known_sources, self.unknown_targets]:
            if os.path.exists(directory):
                for filename in os.listdir(directory):
                    if ' ' in filename:
                        old_path = os.path.join(directory, filename)
                        new_filename = filename.replace(' ', '_')
                        new_path = os.path.join(directory, new_filename)
                        try:
                            os.rename(old_path, new_path)
                        except Exception as e:
                            print(f"Errore nel rinominare {old_path}: {e}")
    
    def image_parsing(self):
        """Converte immagini in formato JPG"""
        # Crea directory TMP se non esiste
        os.makedirs(self.tmp_dir, exist_ok=True)
        
        # Estensioni da convertire
        extensions = ['png', 'jpeg', 'bmp', 'tiff', 'tif', 'gif']
        
        # Converti immagini in JPG per entrambe le directory
        for directory_name in ['01.known_sources', '02.unknown_targets']:
            directory = os.path.join(self.work_dir, directory_name)
            if os.path.exists(directory):
                for ext in extensions:
                    # Trova tutti i file con l'estensione corrente
                    pattern = os.path.join(directory, f'*.{ext}')
                    files = glob.glob(pattern)
                    
                    if files:
                        # Converti in JPG usando mogrify
                        for file_path in files:
                            try:
                                subprocess.run(['mogrify', '-format', 'jpg', file_path], 
                                             check=True, capture_output=True)
                                # Sposta il file originale in TMP
                                shutil.move(file_path, self.tmp_dir)
                            except subprocess.CalledProcessError as e:
                                print(f"Errore nella conversione di {file_path}: {e}")
                            except Exception as e:
                                print(f"Errore nello spostamento di {file_path}: {e}")

    def draw_border(self, frame, pt1, pt2, color, thickness, r, d):
        """Disegna un bordo arrotondato intorno al volto (da recon.py)"""
        x1, y1 = pt1
        x2, y2 = pt2

        # Top left
        cv2.line(frame, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
        cv2.line(frame, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
        cv2.ellipse(frame, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

        # Top right
        cv2.line(frame, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
        cv2.line(frame, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
        cv2.ellipse(frame, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

        # Bottom left
        cv2.line(frame, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
        cv2.line(frame, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
        cv2.ellipse(frame, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

        # Bottom right
        cv2.line(frame, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
        cv2.line(frame, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
        cv2.ellipse(frame, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    def recon_landmarks(self, source_dir, destination_dir):
        """Funzione integrata che sostituisce recon.py"""
        print(f"[INFO] Processing landmarks: {source_dir} -> {destination_dir}")
        
        # Crea directory di destinazione se non esiste
        os.makedirs(destination_dir, exist_ok=True)
        
        # Lista tutti i file nella directory sorgente
        if not os.path.exists(source_dir):
            print(f"[ERROR] Directory sorgente non trovata: {source_dir}")
            return False
        
        files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not files:
            print(f"[WARNING] Nessuna immagine trovata in: {source_dir}")
            return True
        
        print(f"[INFO] Elaborazione di {len(files)} immagini...")
        
        for file in files:
            try:
                file_path = os.path.join(source_dir, file)
                print(f"[INFO] Elaborando: {file_path}")
                
                # Carica l'immagine
                frame = cv2.imread(file_path)
                if frame is None:
                    print(f"[ERROR] Impossibile caricare: {file_path}")
                    continue
                
                # Ridimensiona per face_recognition
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                
                # Trova le posizioni dei volti
                face_locations = face_recognition.face_locations(small_frame)
                
                # Converti in scala di grigi per dlib
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Rileva volti con dlib
                rects = self.detector(gray, 0)
                
                # Disegna i landmarks per ogni volto rilevato da dlib
                for rect in rects:
                    shape = self.predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)
                    
                    # Disegna ogni punto landmark
                    for (x, y) in shape:
                        cv2.circle(frame, (x, y), 1, (255, 0, 0), 1)
                
                # Disegna i bordi per ogni volto rilevato da face_recognition
                for top, right, bottom, left in face_locations:
                    # Scala i coordinate dal frame ridotto a quello originale
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4
                    
                    # Disegna il bordo arrotondato
                    self.draw_border(frame, (left, top), (right, bottom), (255, 0, 0), 1, 10, 20)
                
                # Salva l'immagine processata
                output_path = os.path.join(destination_dir, file)
                cv2.imwrite(output_path, frame)
                print(f"[INFO] Salvato: {output_path}")
                
            except Exception as e:
                print(f"[ERROR] Errore nell'elaborazione di {file}: {e}")
                continue
        
    def copy_original_images(self):
        """Copia le immagini originali nella directory reports per l'accesso via browser"""
        print("[INFO] Copiando immagini originali per accesso web...")
        
        # Copia immagini conosciute
        if os.path.exists(self.known_sources):
            for filename in os.listdir(self.known_sources):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    src_path = os.path.join(self.known_sources, filename)
                    dst_path = os.path.join(self.originals_known_dir, filename)
                    try:
                        shutil.copy2(src_path, dst_path)
                        print(f"[INFO] Copiato: {filename} -> originals_known/")
                    except Exception as e:
                        print(f"[ERROR] Errore copia {filename}: {e}")
        
        # Copia immagini target
        if os.path.exists(self.unknown_targets):
            for filename in os.listdir(self.unknown_targets):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    src_path = os.path.join(self.unknown_targets, filename)
                    dst_path = os.path.join(self.originals_targets_dir, filename)
                    try:
                        shutil.copy2(src_path, dst_path)
                        print(f"[INFO] Copiato: {filename} -> originals_targets/")
                    except Exception as e:
                        print(f"[ERROR] Errore copia {filename}: {e}")

    def integrated_landmarks_processing(self):
        """Elabora i landmarks per entrambe le directory sostituendo i subprocess di recon.py"""
        print("[INFO] Avvio elaborazione landmarks integrata...")
        
        # Processa known_sources -> recondbase
        success_known = self.recon_landmarks(self.known_sources, self.recondbase_dir)
        
        # Processa unknown_targets -> recontarget  
        success_unknown = self.recon_landmarks(self.unknown_targets, self.recontarget_dir)
        
        if success_known and success_unknown:
            print("[INFO] Elaborazione landmarks completata con successo!")
            return True
        else:
            print("[ERROR] Errori nell'elaborazione landmarks")
            return False

    def extract_face_data(self, image_path):
        """Estrae tutti i volti e i loro encoding da un'immagine"""
        try:
            # Carica l'immagine
            image = face_recognition.load_image_file(image_path)
            
            # Trova tutte le posizioni dei volti
            face_locations = face_recognition.face_locations(image)
            
            # Trova tutti i face landmarks
            face_landmarks_list = face_recognition.face_landmarks(image)
            
            # Calcola gli encoding per tutti i volti
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            return {
                'image': image,
                'face_locations': face_locations,
                'face_landmarks': face_landmarks_list,
                'face_encodings': face_encodings,
                'filename': os.path.basename(image_path)
            }
        except Exception as e:
            print(f"Errore nell'elaborazione di {image_path}: {e}")
            return None

    def draw_landmarks_on_match(self, image_data, face_index, match_info, output_path):
        """Disegna i landmarks solo sui volti che hanno un match"""
        try:
            # Converte da RGB a BGR per OpenCV
            image = cv2.cvtColor(image_data['image'], cv2.COLOR_RGB2BGR)
            
            if face_index < len(image_data['face_landmarks']):
                landmarks = image_data['face_landmarks'][face_index]
                
                # Colori per diverse parti del viso
                colors = {
                    'chin': (0, 255, 0),           # Verde
                    'left_eyebrow': (255, 0, 0),    # Blu
                    'right_eyebrow': (255, 0, 0),   # Blu
                    'nose_bridge': (0, 0, 255),     # Rosso
                    'nose_tip': (0, 0, 255),        # Rosso
                    'left_eye': (255, 255, 0),      # Ciano
                    'right_eye': (255, 255, 0),     # Ciano
                    'top_lip': (255, 0, 255),       # Magenta
                    'bottom_lip': (255, 0, 255)     # Magenta
                }
                
                # Disegna i landmarks
                for feature, points in landmarks.items():
                    color = colors.get(feature, (0, 255, 0))
                    for point in points:
                        cv2.circle(image, point, 2, color, -1)
                
                # Disegna un rettangolo intorno al volto
                if face_index < len(image_data['face_locations']):
                    top, right, bottom, left = image_data['face_locations'][face_index]
                    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
                    
                    # Aggiungi testo con la percentuale di match
                    percentage_text = f"Match: {match_info['percentage']:.1f}%"
                    cv2.putText(image, percentage_text, (left, top-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Salva l'immagine con i landmarks
            cv2.imwrite(output_path, image)
            return True
            
        except Exception as e:
            print(f"Errore nel disegnare i landmarks: {e}")
            return False

    def multi_face_recognition(self, tolerance_percent):
        """Esegue il riconoscimento facciale multiplo con la tolleranza specificata"""
        # Calcola la tolleranza per face_recognition (0.0 - 1.0)
        tolerance = (100 - tolerance_percent) / 100.0
        
        # File di output
        raw_data_file = os.path.join(self.reports_dir, 'raw_data_multiface.txt')
        results_file = os.path.join(self.reports_dir, 'multiface_results.json')
        
        # Notifica all'utente
        try:
            subprocess.run(['notify-send', '-i', '/usr/share/icons/tsurugi/tsurugi.png',
                          'Processing multiple faces, please wait...'])
        except:
            print("Elaborazione volti multipli in corso...")
        
        # Elabora tutte le immagini conosciute
        known_faces_data = {}
        print("Elaborazione immagini conosciute...")
        
        for filename in os.listdir(self.known_sources):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(self.known_sources, filename)
                face_data = self.extract_face_data(image_path)
                if face_data and face_data['face_encodings']:
                    known_faces_data[filename] = face_data
                    print(f"Trovati {len(face_data['face_encodings'])} volti in {filename}")
        
        # Elabora tutte le immagini target
        unknown_faces_data = {}
        print("Elaborazione immagini target...")
        
        for filename in os.listdir(self.unknown_targets):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(self.unknown_targets, filename)
                face_data = self.extract_face_data(image_path)
                if face_data and face_data['face_encodings']:
                    unknown_faces_data[filename] = face_data
                    print(f"Trovati {len(face_data['face_encodings'])} volti in {filename}")
        
        # Esegui i confronti
        results = []
        match_counter = 0
        
        print("Esecuzione confronti...")
        for unknown_filename, unknown_data in unknown_faces_data.items():
            for unknown_face_idx, unknown_encoding in enumerate(unknown_data['face_encodings']):
                
                for known_filename, known_data in known_faces_data.items():
                    for known_face_idx, known_encoding in enumerate(known_data['face_encodings']):
                        
                        # Calcola la distanza
                        distances = face_recognition.face_distance([known_encoding], unknown_encoding)
                        distance = distances[0]
                        
                        # Se la distanza è sotto la soglia di tolleranza
                        if distance <= tolerance:
                            percentage = (1 - distance) * 100
                            match_counter += 1
                            
                            match_info = {
                                'known_image': known_filename,
                                'known_face_index': known_face_idx,
                                'unknown_image': unknown_filename,
                                'unknown_face_index': unknown_face_idx,
                                'distance': float(distance),
                                'percentage': percentage,
                                'match_id': match_counter
                            }
                            
                            results.append(match_info)
                            
                            # Crea immagini con landmarks per i match
                            known_landmarks_path = os.path.join(
                                self.landmarks_dir, 
                                f"known_{match_counter}_{known_filename}"
                            )
                            unknown_landmarks_path = os.path.join(
                                self.landmarks_dir, 
                                f"unknown_{match_counter}_{unknown_filename}"
                            )
                            
                            # Disegna landmarks sui volti che hanno match
                            self.draw_landmarks_on_match(
                                known_data, known_face_idx, match_info, known_landmarks_path
                            )
                            self.draw_landmarks_on_match(
                                unknown_data, unknown_face_idx, match_info, unknown_landmarks_path
                            )
                            
                            print(f"Match {match_counter}: {known_filename}[{known_face_idx}] -> "
                                  f"{unknown_filename}[{unknown_face_idx}] ({percentage:.1f}%)")
        
        # Salva i risultati in JSON
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Salva anche in formato testo per compatibilità
        with open(raw_data_file, 'w') as f:
            for result in results:
                f.write(f"{result['known_image']},{result['unknown_image']},{result['distance']:.6f},"
                       f"{result['percentage']:.2f}%,faces:{result['known_face_index']}->{result['unknown_face_index']}\n")
        
        print(f"Trovati {len(results)} match totali")
        return len(results) > 0

    def generate_multiface_report(self, tolerance_percent):
        """Genera il report HTML per volti multipli"""
        index_file = os.path.join(self.reports_dir, 'multiface_index.html')
        results_file = os.path.join(self.reports_dir, 'multiface_results.json')
        
        # Carica i risultati
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
        except:
            results = []
        
        # Inizio HTML con JavaScript per ordinamento
        html_content = '''<html><head><title>Multi-Face Recognition Results</title>
<style>
    body { font-family: Arial, sans-serif; background-color: #f0f0f0; margin: 0; padding: 0; }
    .header { background-color: #333; color: white; padding: 10px; text-align: center; position: sticky; top: 0; z-index: 100; box-shadow: 0 2px 5px rgba(0,0,0,0.3); }
    .controls { background-color: #f8f9fa; padding: 15px; text-align: center; border-bottom: 2px solid #ddd; position: sticky; top: 80px; z-index: 99; }
    .sort-btn { background-color: #007bff; color: white; border: none; padding: 10px 20px; margin: 5px; border-radius: 5px; cursor: pointer; font-size: 14px; transition: all 0.3s; }
    .sort-btn:hover { background-color: #0056b3; transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.2); }
    .sort-btn.active { background-color: #28a745; }
    .sort-btn.desc { background-color: #dc3545; }
    .sort-btn.desc:hover { background-color: #c82333; }
    .results-container { padding: 10px; }
    .match-container { background-color: white; margin: 10px; padding: 15px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); transition: all 0.3s; }
    .match-container:hover { box-shadow: 0 4px 10px rgba(0,0,0,0.2); }
    .image-pair { display: flex; align-items: center; gap: 20px; flex-wrap: wrap; justify-content: center; }
    .image-info { text-align: center; flex: 1; min-width: 300px; }
    .image-info img { max-width: 400px; max-height: 300px; border: 2px solid #ddd; border-radius: 5px; cursor: pointer; transition: transform 0.2s; }
    .image-info img:hover { transform: scale(1.05); border-color: #007bff; }
    .image-link { text-decoration: none; }
    .image-link:hover { text-decoration: underline; }
    .original-link { background-color: #007bff; color: white; padding: 5px 10px; border-radius: 3px; text-decoration: none; font-size: 12px; margin-top: 5px; display: inline-block; }
    .original-link:hover { background-color: #0056b3; text-decoration: none; color: white; }
    .match-info { background-color: #e8f5e8; padding: 15px; border-radius: 5px; min-width: 200px; flex: 0 0 auto; }
    .percentage { font-size: 24px; font-weight: bold; color: #2e7d32; }
    .match-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
    .match-number { color: #666; font-size: 18px; font-weight: bold; }
    @media (max-width: 768px) {
        .image-pair { flex-direction: column; }
    }
</style>
<script>
let currentSort = 'original';
let sortDirection = 'asc';

function sortResults(sortType) {
    const container = document.getElementById('resultsContainer');
    const matches = Array.from(container.getElementsByClassName('match-container'));
    
    // Toggle direction if clicking same button
    if (currentSort === sortType) {
        sortDirection = sortDirection === 'asc' ? 'desc' : 'asc';
    } else {
        sortDirection = 'asc';
        currentSort = sortType;
    }
    
    // Update button states
    document.querySelectorAll('.sort-btn').forEach(btn => {
        btn.classList.remove('active', 'desc');
    });
    const activeBtn = document.getElementById('btn-' + sortType);
    activeBtn.classList.add('active');
    if (sortDirection === 'desc') {
        activeBtn.classList.add('desc');
    }
    
    // Sort matches
    matches.sort((a, b) => {
        let valA, valB;
        
        switch(sortType) {
            case 'percentage':
                valA = parseFloat(a.getAttribute('data-percentage'));
                valB = parseFloat(b.getAttribute('data-percentage'));
                break;
            case 'known':
                valA = a.getAttribute('data-known').toLowerCase();
                valB = b.getAttribute('data-known').toLowerCase();
                break;
            case 'target':
                valA = a.getAttribute('data-target').toLowerCase();
                valB = b.getAttribute('data-target').toLowerCase();
                break;
            case 'distance':
                valA = parseFloat(a.getAttribute('data-distance'));
                valB = parseFloat(b.getAttribute('data-distance'));
                break;
            default: // original
                valA = parseInt(a.getAttribute('data-original-order'));
                valB = parseInt(b.getAttribute('data-original-order'));
        }
        
        if (sortType === 'percentage') {
            // For percentage, higher is better
            return sortDirection === 'asc' ? valA - valB : valB - valA;
        } else if (sortType === 'distance') {
            // For distance, lower is better
            return sortDirection === 'asc' ? valA - valB : valB - valA;
        } else if (sortType === 'known' || sortType === 'target') {
            // Alphabetical
            if (valA < valB) return sortDirection === 'asc' ? -1 : 1;
            if (valA > valB) return sortDirection === 'asc' ? 1 : -1;
            return 0;
        } else {
            // Original order
            return valA - valB;
        }
    });
    
    // Reorder DOM elements
    matches.forEach(match => container.appendChild(match));
    
    // Update match numbers
    updateMatchNumbers();
}

function updateMatchNumbers() {
    const matches = document.getElementsByClassName('match-container');
    Array.from(matches).forEach((match, index) => {
        const numberSpan = match.querySelector('.match-number');
        if (numberSpan) {
            numberSpan.textContent = 'Match #' + (index + 1);
        }
    });
}

// Initialize on page load
window.onload = function() {
    document.getElementById('btn-original').classList.add('active');
};
</script>
</head>
<body>
<div class="header">
    <h1>🔍 MULTI-FACE RECOGNITION RESULTS</h1>
    <p>Tolerance: >''' + str(tolerance_percent) + '''% | Total Matches: ''' + str(len(results)) + '''</p>
</div>

<div class="controls">
    <strong>🔄 Sort Results:</strong><br><br>
    <button id="btn-original" class="sort-btn" onclick="sortResults('original')" title="Restore original order">
        📋 Original Order
    </button>
    <button id="btn-percentage" class="sort-btn" onclick="sortResults('percentage')" title="Sort by match percentage (click again to reverse)">
        📊 By Percentage
    </button>
    <button id="btn-distance" class="sort-btn" onclick="sortResults('distance')" title="Sort by distance value (click again to reverse)">
        📏 By Distance
    </button>
    <button id="btn-known" class="sort-btn" onclick="sortResults('known')" title="Sort alphabetically by known image name">
        🔤 By Known Name (A-Z)
    </button>
    <button id="btn-target" class="sort-btn" onclick="sortResults('target')" title="Sort alphabetically by target image name">
        🔤 By Target Name (A-Z)
    </button>
    <br>
    <small style="color: #666; margin-top: 10px; display: inline-block;">
        💡 Click any button again to reverse sort order | Green = Active Sort | Red = Descending
    </small>
</div>

<div id="resultsContainer" class="results-container">
'''
        
        # Aggiungi ogni match con attributi data per ordinamento
        for i, result in enumerate(results, 1):
            known_landmarks_path = f"landmarks/known_{result['match_id']}_{result['known_image']}"
            unknown_landmarks_path = f"landmarks/unknown_{result['match_id']}_{result['unknown_image']}"
            
            html_content += f'''
<div class="match-container" 
     data-original-order="{i}"
     data-percentage="{result['percentage']:.2f}"
     data-distance="{result['distance']:.6f}"
     data-known="{result['known_image']}"
     data-target="{result['unknown_image']}">
    <div class="match-header">
        <span class="match-number">Match #{i}</span>
        <span style="color: #999; font-size: 14px;">Match ID: {result['match_id']}</span>
    </div>
    <div class="image-pair">
        <div class="image-info">
            <h4>Known Face</h4>
            <a href="{known_landmarks_path}" target="_blank" class="image-link">
                <img src="{known_landmarks_path}" alt="Known face with landmarks" title="Click to open landmarks in new tab">
            </a>
            <p><strong>{result['known_image']}</strong><br>Face Index: {result['known_face_index']}</p>
            <a href="originals_known/{result['known_image']}" target="_blank" class="original-link">
                📷 View Original Image
            </a>
        </div>
        
        <div class="match-info">
            <div class="percentage">{result['percentage']:.1f}%</div>
            <p><strong>Match Quality:</strong><br>
            {'🟢 Excellent' if result['percentage'] >= 60 else 
             '🟡 Good' if result['percentage'] >= 50 else 
             '🟠 Fair' if result['percentage'] >= 40 else 
             '🔴 Poor'}</p>
            <p><strong>Distance:</strong> {result['distance']:.4f}</p>
            <p><strong>Faces Found:</strong><br>Known: {result['known_face_index']+1}, Target: {result['unknown_face_index']+1}</p>
        </div>
        
        <div class="image-info">
            <h4>Target Face</h4>
            <a href="{unknown_landmarks_path}" target="_blank" class="image-link">
                <img src="{unknown_landmarks_path}" alt="Target face with landmarks" title="Click to open landmarks in new tab">
            </a>
            <p><strong>{result['unknown_image']}</strong><br>Face Index: {result['unknown_face_index']}</p>
            <a href="originals_targets/{result['unknown_image']}" target="_blank" class="original-link">
                📷 View Original Image
            </a>
        </div>
    </div>
</div>
'''
        
        # Fine HTML
        html_content += '''
<div style="text-align: center; margin: 20px; color: #666;">
    <p><strong>Legend:</strong></p>
    <p>🔴 &lt;40% Insufficient | 🟠 &gt;40% Minimum | 🟡 &gt;50% Optimal | 🟢 &gt;60% Maximum</p>
    <p><em>Face landmarks are shown only for matched faces</em></p>
    <tt>Enhanced by Visi@n - Multi-Face Recognition System</tt>
</div>
</body>
</html>'''
        
        # Scrivi il file HTML
        try:
            with open(index_file, 'w') as f:
                f.write(html_content)
            return index_file
        except Exception as e:
            print(f"Errore nella scrittura del file HTML: {e}")
            return None

    def show_multiface_report(self):
        """Mostra il report multiface nel browser"""
        index_file = os.path.join(self.reports_dir, 'multiface_index.html')
        
        # Apri il browser con il report HTML
        if os.path.exists(index_file):
            try:
                webbrowser.open(f'file://{index_file}')
            except:
                subprocess.run(['firefox', '-new-tab', '-url', index_file])

    def process_multiface_recognition(self, tolerance_percent):
        """Processo completo di riconoscimento facciale multiplo con landmarks integrati"""
        print("Avvio processo riconoscimento volti multipli...")
        self.filename_normalization()
        self.image_parsing()
        
        # Sostituisce i subprocess di recon.py con funzione integrata
        if not self.integrated_landmarks_processing():
            print("Errore nell'elaborazione landmarks, continuo comunque...")
        
        # Copia le immagini originali per l'accesso web
        self.copy_original_images()
        
        if self.multi_face_recognition(tolerance_percent):
            self.generate_multiface_report(tolerance_percent)
            self.show_multiface_report()
            print("Processo completato con successo!")
        else:
            print("Nessun match trovato con la tolleranza specificata.")

# Mantengo le funzioni originali per compatibilità
class FaceRecognitionApp(MultiFaceRecognitionApp):
    """Classe originale per mantenere compatibilità"""
    pass

# Funzioni GUI aggiornate
def openWorkingDirectory():
    """Apre la directory di lavoro e crea le sottocartelle necessarie"""
    os.chdir(os.path.expanduser('~'))
    
    # Crea le directory se non esistono
    directories = [
        '02.computer_vision/01.known_sources',
        '02.computer_vision/02.unknown_targets', 
        '02.computer_vision/03.reports',
        '02.computer_vision/03.reports/landmarks',
        '02.computer_vision/03.reports/recondbase',
        '02.computer_vision/03.reports/recontarget',
        '02.computer_vision/99.TMP'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Apri il file manager
    command = 'caja ~/02.computer_vision &'
    os.system(command)

def multiface_compare(tolerance):
    """Funzione per eseguire il confronto multi-volto con una data tolleranza"""
    app = MultiFaceRecognitionApp()
    app.process_multiface_recognition(tolerance)

def compare01():
    multiface_compare(1)

def compare40():
    multiface_compare(40)

def compare45():
    multiface_compare(45)

def compare50():
    multiface_compare(50)

def compare55():
    multiface_compare(55)

def compare60():
    multiface_compare(60)

# GUI principale aggiornata
root = tk.Tk()
frame = tk.Frame(root)
frame.pack()
root.wm_title("Multi-Face Recognition")
root.geometry("300x480")

label = tk.Label(text="Multi-Face Recognition HTML",
                fg="red",
                font=("helvetica", 12, "bold"))
label.pack(ipadx=5, ipady=4, pady=15, padx=10)
label.place(x=20, y=0)

info_label = tk.Label(text="Detects and compares ALL faces\nin each image with landmarks",
                     fg="blue",
                     font=("helvetica", 9))
info_label.pack(ipadx=5, ipady=2, pady=5)
info_label.place(x=50, y=30)

button = tk.Button(frame,
                   text="QUIT",
                   fg="#ffffff",
                   bg="#000000",
                   command=quit)
button.pack(ipadx=85, ipady=4, pady=25)

slogan = tk.Button(frame,
                   text="OPEN WORK DIRECTORY",
                   fg="#ffffff",
                   bg="#550000",
                   command=openWorkingDirectory)
slogan.pack(ipadx=20, ipady=4, pady=2)

slogan = tk.Button(frame,
                   text="MULTI-FACE RECOGNITION >1%",
                   fg="#ffffff",
                   bg="#770000",
                   command=compare01)
slogan.pack(ipadx=8, ipady=4, pady=2)

slogan = tk.Button(frame,
                   text="MULTI-FACE RECOGNITION >40%",
                   fg="#ffffff",
                   bg="#990000",
                   command=compare40)
slogan.pack(ipadx=4, ipady=4, pady=2)

slogan = tk.Button(frame,
                   text="MULTI-FACE RECOGNITION >45%",
                   fg="#ffffff",
                   bg="#990000",
                   command=compare45)
slogan.pack(ipadx=4, ipady=4, pady=2)

slogan = tk.Button(frame,
                   text="MULTI-FACE RECOGNITION >50%",
                   fg="#ffffff",
                   bg="#bb0000",
                   command=compare50)
slogan.pack(ipadx=4, ipady=4, pady=2)

slogan = tk.Button(frame,
                   text="MULTI-FACE RECOGNITION >55%",
                   fg="#ffffff",
                   bg="#dd0000",
                   command=compare55)
slogan.pack(ipadx=4, ipady=4, pady=2)

slogan = tk.Button(frame,
                   text="MULTI-FACE RECOGNITION >60%",
                   fg="#ffffff",
                   bg="#ff0000",
                   command=compare60)
slogan.pack(ipadx=4, ipady=4, pady=2)

label = tk.Label(text="Enhanced Multi-Face by Visi@n",
                fg="red",
                font=("helvetica", 10))
label.pack(ipadx=5, ipady=4, pady=10)
label.place(x=80, y=440)

root.mainloop()
