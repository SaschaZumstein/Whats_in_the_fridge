import os
import shutil

# Pfade anpassen:
quelle_ordner = [
    "C:/Users/FHGR/Documents/Bildverarbeitung_1/yoghurt_001",
    "C:/Users/FHGR/Documents/Bildverarbeitung_1/yoghurt_002",
    "C:/Users/FHGR/Documents/Bildverarbeitung_1/yoghurt_003",
    "C:/Users/FHGR/Documents/Bildverarbeitung_1/yoghurt_004",
    "C:/Users/FHGR/Documents/Bildverarbeitung_1/yoghurt_005"
]
zielordner = "C:/Users/FHGR/Documents/Bildverarbeitung_1/pictures_yoghurt_001"

# Zielordner erstellen, falls nicht vorhanden
os.makedirs(zielordner, exist_ok=True)

# Zähler für eindeutige Dateinamen
global_counter = 1

# Bilder aus allen Quellordnern durchgehen
for ordner in quelle_ordner:
    for datei in os.listdir(ordner):
        quellpfad = os.path.join(ordner, datei)
        if os.path.isfile(quellpfad):
            # Dateiendung behalten
            _, dateiendung = os.path.splitext(datei)
            # Neuen eindeutigen Dateinamen erstellen
            neuer_name = f"{global_counter:05}{dateiendung.lower()}"
            zielpfad = os.path.join(zielordner, neuer_name)

            # Datei kopieren
            shutil.copy2(quellpfad, zielpfad)

            print(f"Kopiert: {quellpfad} -> {zielpfad}")
            global_counter += 1

print("Fertig!")
