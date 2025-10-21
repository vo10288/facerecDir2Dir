#!/usr/bin/env python3

# Multi-Face Recognition System v2.2
# Enhanced version with multiple face detection, comparison and VIDEO SUPPORT
# https://tsurugi-linux.org
# by Visi@n
# LICENSE: MIT License
# Based on face_recognition library by Adam Geitgey

import tkinter as tk
from tkinter import filedialog, messagebox
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
import urllib.request
import bz2

class MultiFaceRecognitionApp:
    def __init__(self, custom_known_dir=None, custom_unknown_dir=None):
        self.home_dir = os.path.expanduser('~')
        self.work_dir = os.path.join(self.home_dir, '02.computer_vision')
        
        # Usa directory personalizzate se fornite, altrimenti usa quelle di default
        self.known_sources = custom_known_dir if custom_known_dir else os.path.join(self.work_dir, '01.known_sources')
        self.unknown_targets = custom_unknown_dir if custom_unknown_dir else os.path.join(self.work_dir, '02.unknown_targets')
        
        self.reports_dir = os.path.join(self.work_dir, '03.reports')
        self.tmp_dir = os.path.join(self.work_dir, '99.TMP')
        self.landmarks_dir = os.path.join(self.reports_dir, 'landmarks')
        self.recondbase_dir = os.path.join(self.reports_dir, 'recondbase')
        self.recontarget_dir = os.path.join(self.reports_dir, 'recontarget')
        self.originals_known_dir = os.path.join(self.reports_dir, 'originals_known')
        self.originals_targets_dir = os.path.join(self.reports_dir, 'originals_targets')
        self.video_frames_dir = os.path.join(self.work_dir, '98.VIDEO_FRAMES')
        
        # Crea tutte le directory necessarie
        for directory in [self.landmarks_dir, self.recondbase_dir, self.recontarget_dir, 
                         self.originals_known_dir, self.originals_targets_dir, self.video_frames_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Inizializza il detector e predictor di dlib
        self.detector = dlib.get_frontal_face_detector()
        
        # Carica il modello dalla directory dello script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        shape_predictor_path = os.path.join(script_dir, 'shape_predictor_68_face_landmarks.dat')
        
        # Scarica il modello se non esiste
        if not os.path.exists(shape_predictor_path):
            print(f"[INFO] Modello non trovato: {shape_predictor_path}")
            print("[INFO] Avvio download automatico del modello...")
            if not self.download_shape_predictor(script_dir):
                raise FileNotFoundError(f"Impossibile scaricare o trovare il modello shape_predictor_68_face_landmarks.dat")
        
        print(f"[INFO] Caricamento modello da: {shape_predictor_path}")
        self.predictor = dlib.shape_predictor(shape_predictor_path)
    
    def download_shape_predictor(self, destination_dir):
        """Scarica automaticamente il modello shape_predictor_68_face_landmarks.dat"""
        model_url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        compressed_file = os.path.join(destination_dir, "shape_predictor_68_face_landmarks.dat.bz2")
        extracted_file = os.path.join(destination_dir, "shape_predictor_68_face_landmarks.dat")
        
        try:
            print(f"[INFO] Download del modello da: {model_url}")
            print("[INFO] Questo potrebbe richiedere alcuni minuti (file ~100MB)...")
            
            # Download con barra di progresso
            def report_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(downloaded * 100 / total_size, 100)
                print(f"\r[INFO] Download: {percent:.1f}% ({downloaded / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB)", end='')
            
            urllib.request.urlretrieve(model_url, compressed_file, reporthook=report_progress)
            print("\n[INFO] Download completato!")
            
            # Decomprimi il file
            print("[INFO] Decompressione del modello...")
            with bz2.open(compressed_file, 'rb') as source:
                with open(extracted_file, 'wb') as dest:
                    dest.write(source.read())
            
            # Rimuovi il file compresso
            os.remove(compressed_file)
            print(f"[INFO] Modello scaricato e estratto con successo: {extracted_file}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Errore nel download del modello: {e}")
            print("[INFO] Puoi scaricare manualmente il modello da:")
            print(f"       {model_url}")
            print(f"[INFO] Decomprimi il file .bz2 e mettilo in: {destination_dir}")
            return False
    
    def check_ffmpeg(self):
        """Verifica se ffmpeg è installato nel sistema"""
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=5)
            if result.returncode == 0:
                print("[INFO] ffmpeg trovato e funzionante")
                return True
            else:
                return False
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def extract_frames_from_video(self, video_path, output_dir, fps=1):
        """Estrae frame da un video usando ffmpeg"""
        try:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            frame_output_pattern = os.path.join(output_dir, f"{video_name}_frame_%04d.jpg")
            
            print(f"[INFO] Estrazione frame da: {video_path}")
            
            # Comando ffmpeg per estrarre frame
            # fps=1 significa 1 frame al secondo
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vf', f'fps={fps}',
                '-q:v', '2',  # Qualità JPEG (1-31, più basso = migliore)
                frame_output_pattern,
                '-y'  # Sovrascrivi file esistenti
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Conta i frame estratti
                extracted_frames = glob.glob(os.path.join(output_dir, f"{video_name}_frame_*.jpg"))
                print(f"[INFO] Estratti {len(extracted_frames)} frame da {video_name}")
                return True
            else:
                print(f"[ERROR] Errore nell'estrazione frame da {video_path}")
                print(f"[ERROR] {result.stderr}")
                return False
                
        except Exception as e:
            print(f"[ERROR] Errore nell'elaborazione video {video_path}: {e}")
            return False
    
    def process_videos_in_directory(self, directory, fps=1):
        """Processa tutti i video in una directory ed estrae i frame"""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.mpeg', '.mpg']
        videos_found = []
        
        if not os.path.exists(directory):
            return videos_found
        
        # Trova tutti i video
        for file in os.listdir(directory):
            if any(file.lower().endswith(ext) for ext in video_extensions):
                videos_found.append(os.path.join(directory, file))
        
        if not videos_found:
            return videos_found
        
        print(f"[INFO] Trovati {len(videos_found)} video in {directory}")
        
        # Verifica ffmpeg
        if not self.check_ffmpeg():
            error_msg = (
                "⚠️ FFMPEG NON TROVATO!\n\n"
                f"Sono stati trovati {len(videos_found)} file video ma ffmpeg non è installato.\n\n"
                "Per elaborare i video, installa ffmpeg:\n"
                "• Windows: https://ffmpeg.org/download.html\n"
                "• macOS: brew install ffmpeg\n"
                "• Linux: sudo apt install ffmpeg\n\n"
                "I video verranno ignorati."
            )
            print(f"\n{'='*60}")
            print("[ERROR] FFMPEG NON TROVATO!")
            print(f"{'='*60}")
            print(f"[ERROR] Trovati {len(videos_found)} video ma ffmpeg non è installato")
            print("[INFO] Per elaborare i video, installa ffmpeg:")
            print("  • Windows: https://ffmpeg.org/download.html")
            print("  • macOS: brew install ffmpeg")
            print("  • Linux: sudo apt install ffmpeg")
            print(f"{'='*60}\n")
            
            # Mostra messaggio nella GUI
            try:
                messagebox.showwarning("FFMPEG Non Trovato", error_msg)
            except:
                pass
            
            return []
        
        # Crea directory per i frame del video corrente
        for video_path in videos_found:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            video_frames_output = os.path.join(self.video_frames_dir, video_name)
            os.makedirs(video_frames_output, exist_ok=True)
            
            # Estrai frame
            if self.extract_frames_from_video(video_path, video_frames_output, fps):
                # Copia i frame estratti nella directory principale
                extracted_frames = glob.glob(os.path.join(video_frames_output, "*.jpg"))
                for frame in extracted_frames:
                    dest_path = os.path.join(directory, os.path.basename(frame))
                    shutil.copy2(frame, dest_path)
                    print(f"[INFO] Frame copiato: {os.path.basename(frame)}")
        
        return videos_found
    
    def extract_video_frames(self):
        """Estrae frame da tutti i video nelle directory known e unknown"""
        print("\n" + "="*60)
        print("[INFO] CONTROLLO VIDEO E ESTRAZIONE FRAME")
        print("="*60)
        
        # Processa video in known_sources
        known_videos = self.process_videos_in_directory(self.known_sources, fps=1)
        
        # Processa video in unknown_targets
        unknown_videos = self.process_videos_in_directory(self.unknown_targets, fps=1)
        
        total_videos = len(known_videos) + len(unknown_videos)
        
        if total_videos > 0:
            print(f"\n[INFO] Processati {total_videos} video totali")
            print(f"[INFO] Known Sources: {len(known_videos)} video")
            print(f"[INFO] Unknown Targets: {len(unknown_videos)} video")
            print("="*60 + "\n")
        else:
            print("[INFO] Nessun video trovato nelle directory")
            print("="*60 + "\n")
        
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
        for directory in [self.known_sources, self.unknown_targets]:
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
        
        print(f"[INFO] Elaborazione landmarks completata: {destination_dir}")
        return True

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
</div>

<div style="text-align: center; margin: 20px; padding: 20px; background-color: white; border-radius: 5px;">
    <h3>📖 Legend & Instructions</h3>
    <p><strong>Match Quality Scale:</strong></p>
    <p>🔴 &lt;40% Insufficient | 🟠 &gt;40% Minimum | 🟡 &gt;50% Optimal | 🟢 &gt;60% Maximum</p>
    <p><em>Face landmarks are shown only for matched faces</em></p>
    <hr style="margin: 20px 0; border: none; border-top: 1px solid #ddd;">
    <p><strong>🖱️ Navigation Tips:</strong></p>
    <p>• Click on any image to open in new tab<br>
    • 📷 "View Original" opens the source image without processing<br>
    • Use sort buttons to organize results by different criteria<br>
    • Click the same sort button again to reverse the order</p>
    <hr style="margin: 20px 0; border: none; border-top: 1px solid #ddd;">
    <p><strong>🔄 Sorting Options:</strong></p>
    <p>• <strong>Original Order:</strong> Results as they were detected<br>
    • <strong>By Percentage:</strong> Sort by match accuracy (highest/lowest first)<br>
    • <strong>By Distance:</strong> Sort by similarity distance (closest/farthest first)<br>
    • <strong>By Known/Target Name:</strong> Alphabetical sorting by filename</p>
    <tt style="display: block; margin-top: 20px; color: #666;">Enhanced Multi-Face by Visi@n - Interactive Face Recognition System v2.2</tt>
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
        
        # STEP 1: Estrai frame dai video se presenti
        self.extract_video_frames()
        
        # STEP 2: Normalizzazione e parsing
        self.filename_normalization()
        self.image_parsing()
        
        # STEP 3: Elaborazione landmarks
        if not self.integrated_landmarks_processing():
            print("Errore nell'elaborazione landmarks, continuo comunque...")
        
        # STEP 4: Copia immagini originali
        self.copy_original_images()
        
        # STEP 5: Riconoscimento e report
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

# Funzioni GUI
def openWorkingDirectory():
    """Apre la directory di lavoro con il file manager appropriato per il sistema operativo"""
    home_dir = os.path.expanduser('~')
    work_dir = os.path.join(home_dir, '02.computer_vision')
    
    # Crea le directory se non esistono
    directories = [
        '02.computer_vision/01.known_sources',
        '02.computer_vision/02.unknown_targets', 
        '02.computer_vision/03.reports',
        '02.computer_vision/03.reports/landmarks',
        '02.computer_vision/03.reports/recondbase',
        '02.computer_vision/03.reports/recontarget',
        '02.computer_vision/03.reports/originals_known',
        '02.computer_vision/03.reports/originals_targets',
        '02.computer_vision/98.VIDEO_FRAMES',
        '02.computer_vision/99.TMP'
    ]
    
    os.chdir(home_dir)
    for directory in directories:
        dir_path = os.path.join(home_dir, directory)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    # Rileva il sistema operativo e apri il file manager appropriato
    system = platform.system()
    
    try:
        if system == 'Windows':
            # Windows - usa explorer
            os.startfile(work_dir)
            print(f"[INFO] Apertura directory con Windows Explorer: {work_dir}")
            
        elif system == 'Darwin':
            # macOS - usa Finder
            subprocess.Popen(['open', work_dir])
            print(f"[INFO] Apertura directory con Finder: {work_dir}")
            
        elif system == 'Linux':
            # Linux - prova diversi file manager in ordine di preferenza
            file_managers = [
                'xdg-open',      # Standard Linux (usa il default del sistema)
                'nautilus',      # GNOME
                'dolphin',       # KDE
                'thunar',        # XFCE
                'pcmanfm',       # LXDE
                'nemo',          # Cinnamon
                'caja',          # MATE (mantenuto come fallback)
                'konqueror',     # KDE alternativo
            ]
            
            opened = False
            for fm in file_managers:
                try:
                    # Verifica se il comando esiste
                    if shutil.which(fm):
                        subprocess.Popen([fm, work_dir])
                        print(f"[INFO] Apertura directory con {fm}: {work_dir}")
                        opened = True
                        break
                except Exception as e:
                    continue
            
            if not opened:
                print(f"[WARNING] Nessun file manager trovato. Directory: {work_dir}")
                print("[INFO] Puoi aprire manualmente la directory dal terminale con:")
                print(f"       cd {work_dir}")
        else:
            print(f"[WARNING] Sistema operativo non riconosciuto: {system}")
            print(f"[INFO] Directory di lavoro: {work_dir}")
            
    except Exception as e:
        print(f"[ERROR] Errore nell'apertura della directory: {e}")
        print(f"[INFO] Directory di lavoro: {work_dir}")
        print("[INFO] Puoi aprirla manualmente dal file manager")

def multiface_compare(tolerance):
    """Funzione per eseguire il confronto multi-volto con una data tolleranza"""
    # Usa le directory globali configurabili
    app = MultiFaceRecognitionApp(custom_known_dir=known_dir, custom_unknown_dir=unknown_dir)
    app.process_multiface_recognition(tolerance)

def change_known_directory():
    """Permette di cambiare la directory delle immagini conosciute"""
    new_dir = filedialog.askdirectory(title="Seleziona la directory delle immagini conosciute")
    if new_dir:
        # Aggiorna il percorso globale
        global known_dir
        known_dir = new_dir
        known_path_label.config(text=known_dir)
        print(f"[INFO] Directory Known Sources cambiata in: {known_dir}")

def change_unknown_directory():
    """Permette di cambiare la directory delle immagini target"""
    new_dir = filedialog.askdirectory(title="Seleziona la directory delle immagini target")
    if new_dir:
        # Aggiorna il percorso globale
        global unknown_dir
        unknown_dir = new_dir
        unknown_path_label.config(text=unknown_dir)
        print(f"[INFO] Directory Unknown Targets cambiata in: {unknown_dir}")

def compare01():
    multiface_compare(1)

def compare10():
    multiface_compare(10)

def compare20():
    multiface_compare(20)

def compare30():
    multiface_compare(30)

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

def compare65():
    multiface_compare(65)

def compare70():
    multiface_compare(70)

def compare75():
    multiface_compare(75)

def compare80():
    multiface_compare(80)

# GUI principale aggiornata
root = tk.Tk()
frame = tk.Frame(root)
frame.pack()
root.wm_title("Multi-Face Recognition v2.2")
root.geometry("580x790")

# Configurazione dello stile
root.configure(bg="#f0f0f0")

# Header principale
header_frame = tk.Frame(root, bg="#2c3e50", height=70)
header_frame.pack(fill="x")
header_frame.pack_propagate(False)

label = tk.Label(header_frame, 
                text="🔍 Multi-Face Recognition System",
                fg="white",
                bg="#2c3e50",
                font=("helvetica", 16, "bold"))
label.pack(pady=10)

info_label = tk.Label(header_frame,
                     text="Advanced facial detection and comparison with landmarks analysis • Video support with ffmpeg",
                     fg="#ecf0f1",
                     bg="#2c3e50",
                     font=("helvetica", 9, "italic"))
info_label.pack()

# Directory Info Section
dir_frame = tk.Frame(root, bg="#ecf0f1", relief="ridge", borderwidth=2)
dir_frame.pack(fill="x", padx=10, pady=10)

dir_info_label = tk.Label(dir_frame,
                         text="📁 Working Directories Configuration",
                         fg="#2c3e50",
                         bg="#ecf0f1",
                         font=("helvetica", 11, "bold"))
dir_info_label.pack(pady=5)

# Known Sources
known_frame = tk.Frame(dir_frame, bg="#ecf0f1")
known_frame.pack(fill="x", padx=10, pady=5)

known_label = tk.Label(known_frame,
                      text="Known Sources (Reference Images):",
                      fg="#27ae60",
                      bg="#ecf0f1",
                      font=("helvetica", 9, "bold"))
known_label.pack(anchor="w")

home_dir = os.path.expanduser('~')
work_dir = os.path.join(home_dir, '02.computer_vision')
known_dir = os.path.join(work_dir, '01.known_sources')
unknown_dir = os.path.join(work_dir, '02.unknown_targets')

known_path_label = tk.Label(known_frame,
                           text=known_dir,
                           fg="#34495e",
                           bg="#ecf0f1",
                           font=("courier", 8),
                           anchor="w",
                           justify="left")
known_path_label.pack(anchor="w", fill="x")

btn_change_known = tk.Button(known_frame,
                             text="📂 Change Directory",
                             fg="white",
                             bg="#3498db",
                             font=("helvetica", 8),
                             command=change_known_directory)
btn_change_known.pack(anchor="w", pady=2)

# Unknown Targets
unknown_frame = tk.Frame(dir_frame, bg="#ecf0f1")
unknown_frame.pack(fill="x", padx=10, pady=5)

unknown_label = tk.Label(unknown_frame,
                        text="Unknown Targets (Images to Analyze):",
                        fg="#e74c3c",
                        bg="#ecf0f1",
                        font=("helvetica", 9, "bold"))
unknown_label.pack(anchor="w")

unknown_path_label = tk.Label(unknown_frame,
                             text=unknown_dir,
                             fg="#34495e",
                             bg="#ecf0f1",
                             font=("courier", 8),
                             anchor="w",
                             justify="left")
unknown_path_label.pack(anchor="w", fill="x")

btn_change_unknown = tk.Button(unknown_frame,
                               text="📂 Change Directory",
                               fg="white",
                               bg="#3498db",
                               font=("helvetica", 8),
                               command=change_unknown_directory)
btn_change_unknown.pack(anchor="w", pady=2)

# Control Buttons Section
control_frame = tk.Frame(root, bg="#f0f0f0")
control_frame.pack(fill="both", expand=True, padx=10, pady=10)

# Main action buttons
main_btn_frame = tk.Frame(control_frame, bg="#f0f0f0")
main_btn_frame.pack(pady=5)

button = tk.Button(main_btn_frame,
                   text="🗂️  OPEN WORK DIRECTORY",
                   fg="#ffffff",
                   bg="#16a085",
                   font=("helvetica", 10, "bold"),
                   width=35,
                   command=openWorkingDirectory)
button.pack(pady=5)

# Separator
sep_label = tk.Label(control_frame,
                    text="━━━━━━━ Recognition Tolerance Settings ━━━━━━━",
                    fg="#7f8c8d",
                    bg="#f0f0f0",
                    font=("helvetica", 9, "bold"))
sep_label.pack(pady=10)

# Recognition buttons in grid
rec_frame = tk.Frame(control_frame, bg="#f0f0f0")
rec_frame.pack()

# Row 1
row1 = tk.Frame(rec_frame, bg="#f0f0f0")
row1.pack(pady=2)

tk.Button(row1, text="🔍 >1%", fg="#fff", bg="#4a0000", font=("helvetica", 9), 
          width=12, command=compare01).pack(side="left", padx=2)
tk.Button(row1, text="🔍 >10%", fg="#fff", bg="#5a0000", font=("helvetica", 9), 
          width=12, command=compare10).pack(side="left", padx=2)
tk.Button(row1, text="🔍 >20%", fg="#fff", bg="#6a0000", font=("helvetica", 9), 
          width=12, command=compare20).pack(side="left", padx=2)
tk.Button(row1, text="🔍 >30%", fg="#fff", bg="#7a0000", font=("helvetica", 9), 
          width=12, command=compare30).pack(side="left", padx=2)

# Row 2
row2 = tk.Frame(rec_frame, bg="#f0f0f0")
row2.pack(pady=2)

tk.Button(row2, text="🔍 >40%", fg="#fff", bg="#880000", font=("helvetica", 9), 
          width=12, command=compare40).pack(side="left", padx=2)
tk.Button(row2, text="🔍 >45%", fg="#fff", bg="#990000", font=("helvetica", 9), 
          width=12, command=compare45).pack(side="left", padx=2)
tk.Button(row2, text="🔍 >50%", fg="#fff", bg="#aa0000", font=("helvetica", 9, "bold"), 
          width=12, command=compare50).pack(side="left", padx=2)
tk.Button(row2, text="🔍 >55%", fg="#fff", bg="#bb0000", font=("helvetica", 9), 
          width=12, command=compare55).pack(side="left", padx=2)

# Row 3
row3 = tk.Frame(rec_frame, bg="#f0f0f0")
row3.pack(pady=2)

tk.Button(row3, text="🔍 >60%", fg="#fff", bg="#cc0000", font=("helvetica", 9, "bold"), 
          width=12, command=compare60).pack(side="left", padx=2)
tk.Button(row3, text="🔍 >65%", fg="#fff", bg="#dd0000", font=("helvetica", 9), 
          width=12, command=compare65).pack(side="left", padx=2)
tk.Button(row3, text="🔍 >70%", fg="#fff", bg="#ee0000", font=("helvetica", 9, "bold"), 
          width=12, command=compare70).pack(side="left", padx=2)
tk.Button(row3, text="🔍 >75%", fg="#fff", bg="#ff0000", font=("helvetica", 9), 
          width=12, command=compare75).pack(side="left", padx=2)

# Row 4
row4 = tk.Frame(rec_frame, bg="#f0f0f0")
row4.pack(pady=2)

tk.Button(row4, text="🔍 >80%", fg="#fff", bg="#ff1100", font=("helvetica", 9, "bold"), 
          width=12, command=compare80).pack(side="left", padx=2)

# Info box
info_box = tk.Label(control_frame,
                   text="💡 Lower percentage = More matches (less strict)\n"
                        "Higher percentage = Fewer matches (more strict)\n"
                        "Recommended: 50-60% for optimal accuracy\n\n"
                        "🎬 VIDEO SUPPORT: Place video files (.mp4, .avi, .mov, etc.) in directories\n"
                        "Frames will be auto-extracted (requires ffmpeg installed)",
                   fg="#34495e",
                   bg="#d5dbdb",
                   font=("helvetica", 8),
                   justify="left",
                   relief="groove",
                   borderwidth=2,
                   padx=10,
                   pady=8)
info_box.pack(pady=10, fill="x")

# Quit button
quit_btn = tk.Button(control_frame,
                    text="❌ QUIT APPLICATION",
                    fg="#ffffff",
                    bg="#2c3e50",
                    font=("helvetica", 10, "bold"),
                    width=35,
                    command=quit)
quit_btn.pack(pady=10)

# Footer
footer_frame = tk.Frame(root, bg="#34495e", height=40)
footer_frame.pack(fill="x", side="bottom")
footer_frame.pack_propagate(False)

footer_label = tk.Label(footer_frame,
                       text="Enhanced Multi-Face Recognition System v2.2 with Video Support • Developed by Visi@n • https://tsurugi-linux.org",
                       fg="#ecf0f1",
                       bg="#34495e",
                       font=("helvetica", 8, "italic"))
footer_label.pack(pady=10)

root.mainloop()
