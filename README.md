# Multi-Face Recognition System

Un sistema avanzato di riconoscimento facciale multiplo che può rilevare e confrontare tutti i volti presenti in ogni immagine, generando report HTML dettagliati con landmarks facciali.

## 🚀 Caratteristiche

- **Riconoscimento multiplo**: Rileva e confronta TUTTI i volti in ogni immagine
- **Landmarks facciali**: Visualizza punti caratteristici del viso sui match trovati
- **Report HTML interattivi**: Risultati ordinabili con percentuali di match
- **Compatibilità Windows**: Ottimizzato per sistemi Windows
- **Conversione automatica**: Supporta PNG, JPEG, BMP, TIFF, GIF → JPG
- **Soglie personalizzabili**: Multiple opzioni di tolleranza (1%, 40%, 45%, 50%, 55%, 60%)
- **GUI intuitiva**: Interfaccia grafica semplice con Tkinter

## 📋 Requisiti di Sistema

- **Python 3.7+**
- **Windows 10/11** (ottimizzato, ma funziona anche su Linux/macOS)
- **Webcam** (opzionale)
- **4GB RAM** (minimo, 8GB consigliato)

## 🛠️ Installazione

### 1. Clona o scarica il progetto
```bash
git clone <repository-url>
cd multi-face-recognition
```

### 2. Installa le dipendenze
```bash
pip install -r requirements.txt
```

### 3. Scarica il Shape Predictor (Opzionale ma consigliato)
Il file `shape_predictor_68_face_landmarks.dat` è necessario per i landmarks facciali:

1. Scarica da: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
2. Estrai il file `.dat` nella directory dello script
3. **Dimensione**: ~95MB (dopo estrazione)

**Nota**: Il sistema funziona anche senza questo file, ma non mostrerà i landmarks.

## 🎯 Utilizzo

### Avvio del programma
```bash
python facerec_web_GUI-windows-orders.py
```

### Struttura delle directory
Il programma crea automaticamente questa struttura:
```
~/02.computer_vision/
├── 01.known_sources/          # Immagini di riferimento (volti noti)
├── 02.unknown_targets/        # Immagini da analizzare (volti da cercare)
├── 03.reports/               # Report HTML e risultati
│   ├── landmarks/            # Immagini con landmarks
│   ├── originals_known/      # Copie immagini note
│   ├── originals_targets/    # Copie immagini target
│   └── multiface_index.html  # Report principale
└── 99.TMP/                   # File temporanei
```

### Procedura passo-passo

1. **Apri Work Directory**: Clicca il pulsante per creare le cartelle
2. **Aggiungi immagini**:
   - Metti le foto di riferimento in `01.known_sources/`
   - Metti le foto da analizzare in `02.unknown_targets/`
3. **Scegli la tolleranza**: Clicca uno dei pulsanti di riconoscimento
4. **Visualizza risultati**: Si aprirà automaticamente il report HTML

## 🎛️ Opzioni di Tolleranza

| Soglia | Uso Consigliato | Caratteristiche |
|--------|-----------------|-----------------|
| **>1%** | Test/Debug | Mostra quasi tutti i possibili match |
| **>40%** | Ricerca ampia | Include match parziali |
| **>45%** | Uso generale | Buon compromesso |
| **>50%** | Ricerca precisa | Match di qualità |
| **>55%** | Alta precisione | Solo match molto buoni |
| **>60%** | Massima precisione | Solo match eccellenti |

## 📊 Interpretazione dei Risultati

### Qualità dei Match
- **[****] Excellent (>60%)**: Match di altissima qualità
- **[***] Good (>50%)**: Match affidabili
- **[**] Fair (>40%)**: Match accettabili
- **[*] Poor (<40%)**: Match incerti

### Funzioni del Report HTML
- **Ordinamento**: Clicca i pulsanti per ordinare per percentuale, distanza, nome
- **Zoom**: Clicca sulle immagini per ingrandire
- **Navigazione**: Link diretti alle immagini originali
- **Responsive**: Adatta automaticamente la visualizzazione

## 🔧 Risoluzione Problemi

### Errore: "shape_predictor_68_face_landmarks.dat non trovato"
**Soluzione**: Scarica il file come indicato nella sezione installazione

### Errore: "No module named 'face_recognition'"
**Soluzione**: 
```bash
pip install face_recognition
# Se problemi su Windows:
pip install --upgrade setuptools wheel
pip install dlib
pip install face_recognition
```

### Errore: "Microsoft Visual C++ 14.0 is required"
**Soluzione**: Installa Visual Studio Build Tools o usa conda:
```bash
conda install -c conda-forge dlib
conda install -c conda-forge face_recognition
```

### Le immagini non vengono processate
**Controlli**:
1. Verifica che le immagini siano nei formati supportati (JPG, PNG, BMP, TIFF, GIF)
2. Controlla che non ci siano spazi nei nomi file (vengono automaticamente sostituiti)
3. Assicurati che le immagini contengano volti visibili

### Il report HTML è vuoto
**Possibili cause**:
1. Nessun match trovato con la soglia selezionata (prova soglie più basse)
2. Immagini di qualità troppo bassa
3. Volti troppo piccoli o non frontali

## 📁 Formati Supportati

### Immagini Input
- **JPEG/JPG** (consigliato)
- **PNG**
- **BMP** 
- **TIFF/TIF**
- **GIF** (primo frame)

### Output
- **HTML**: Report interattivo
- **JSON**: Dati strutturati
- **TXT**: Dati grezzi

## ⚡ Performance

### Tempi di Elaborazione Tipici
- **1 volto vs 1 volto**: ~1-2 secondi
- **10 volti vs 10 volti**: ~30-60 secondi  
- **50 volti vs 50 volti**: ~10-15 minuti

### Fattori che Influenzano le Performance
- **Risoluzione immagini**: Immagini più grandi = più tempo
- **Numero di volti**: Crescita esponenziale
- **Qualità hardware**: CPU e RAM disponibili

## 🔒 Privacy e Sicurezza

- **Elaborazione locale**: Nessun dato inviato online
- **File temporanei**: Automaticamente gestiti in `99.TMP/`
- **Dati sensibili**: Mantieni le cartelle di lavoro sicure

## 📄 Licenze

- **Script**: MIT License (modificato da Visi@n)
- **face_recognition**: MIT License by Adam Geitgey
- **dlib**: Boost Software License
- **OpenCV**: Apache 2.0 License

## 🤝 Contributi

Questo script è basato sulla libreria `face_recognition` di Adam Geitgey ed è stato modificato da Antonio 'Visi@n' Broi per supportare il riconoscimento di volti multipli.

## 📞 Supporto

Per problemi o domande:
1. Verifica la sezione "Risoluzione Problemi"
2. Controlla che tutte le dipendenze siano installate
3. Assicurati che le immagini siano di buona qualità

## 🔄 Aggiornamenti

### Versione Attuale: Windows Compatible
- ✅ Risolto errore encoding Unicode
- ✅ Conversione immagini con PIL
- ✅ Compatibilità Windows migliorata
- ✅ GUI ottimizzata
- ✅ Gestione errori avanzata

---

**Enhanced Multi-Face Recognition System by Visi@n**
