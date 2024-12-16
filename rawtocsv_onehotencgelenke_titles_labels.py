import os
import csv


def apply_osp(input_path, output_path):
    for root, dirs, files in os.walk(input_path):
        files = sorted([f for f in files if f.endswith(".txt")])  # Nur .txt-Dateien bearbeiten
        for file in files:
            input_file_path = os.path.join(root, file)

            # Ziel-Dateipfad mit .csv-Endung erstellen
            relative_path = os.path.relpath(input_file_path, input_path)
            csv_filename = os.path.splitext(relative_path)[0] + ".csv"  # .txt -> .csv
            output_file_path = os.path.join(output_path, csv_filename)

            # Sicherstellen, dass der Zielordner existiert
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

            # Anpassen und Konvertieren
            apply_txttocsvconvert(input_file_path, output_file_path, file)


def apply_txttocsvconvert(input_file, output_file, filename):
    """
    Liest eine .txt-Datei, verarbeitet jede Zeile und speichert die Ausgabe als .csv-Datei.
    Fügt zusätzlich eine 'Klasse' Spalte basierend auf der letzten Ziffer im Dateinamen hinzu.

    Schritte:
    2. Entfernt Leerzeichen und die Zeichen "(" und ")".
    3. Entfernt die ersten beiden Spalten.
    4. Fügt eine neue 'Klasse' Spalte hinzu, die die letzte Ziffer im Dateinamen enthält.
    5. Speichert das Ergebnis mit definierten Spaltenüberschriften in eine CSV-Datei.
    """
    with open(input_file, 'r') as in_file:
        lines = in_file.readlines()  # Lies alle Zeilen der Datei

    # 1. Verarbeite jede Zeile: Entferne Leerzeichen und Klammern
    processed_lines = [line.strip().replace("(", "").replace(")", "")
                       .replace("SpineBase", "1")
                       .replace("SpineMid", "2")
                       .replace("Neck", "3")
                       .replace("Head", "4")
                       .replace("ShoulderLeft", "5")
                       .replace("ElbowLeft", "6")
                       .replace("WristLeft", "7")
                       .replace("HandLeft", "8")
                       .replace("ShoulderRight", "9")
                       .replace("ElbowRight", "10")
                       .replace("WristRight", "11")
                       .replace("HandRight", "12")
                       .replace("HipLeft", "13")
                       .replace("KneeLeft", "14")
                       .replace("AnkleLeft", "15")
                       .replace("FootLeft", "16")
                       .replace("HipRight", "17")
                       .replace("KneeRight", "18")
                       .replace("AnkleRight", "19")
                       .replace("FootRight", "20")
                       .replace("SpineShoulder", "21")
                       .replace("HandTipLeft", "22")
                       .replace("ThumbLeft", "23")
                       .replace("HandTipRight", "24")
                       .replace("ThumbRight", "25")

                       .replace("Tracked", "0")
                       .replace("Inferred", "1")

                       for line in lines]

    # 5 Extrahiere die Klasse (letzte Ziffer im Dateinamen vor .txt)
    class_label = int(filename.split('_')[-2].split('.')[0])

    # 3. Schreibe die verarbeiteten Daten in eine CSV-Datei
    with open(output_file, 'w', newline='') as out_file:
        writer = csv.writer(out_file)
        #Von header und class Label alles unnötig, weil hstack und vstack das Dataframe Überschriften zerstört.
        # Definiere die Spaltenüberschriften (mit der zusätzlichen 'Klasse' Spalte)
        header = ('Gelenk', 'Measure', 'x3d', 'y3d', 'z3d', 'x2d', 'y2d')

        # Wiederhole die Überschriften 24-mal in den Spalten
        header_row = header * 25  # Überschriften 24-mal wiederholen
        writer.writerow(list(header_row) + ['Klasse'])  # Füge "Klasse" als letzte Spalte hinzu

        # Verarbeite jede Zeile und entferne die ersten beiden Spalten
        for line in processed_lines:
            columns = line.split(",")  # Spalten trennen
            if len(columns) > 3:  # Sicherstellen, dass genügend Spalten vorhanden sind
                # Schreibe die Spalten ab der dritten Spalte und füge die Klasse am Ende hinzu
                row = columns[3:] + [class_label]
                writer.writerow(row )


# Beispielaufruf
if __name__ == '__main__':
    input_path = '/home/boris.grillborzer/PycharmProjects/SecondOrial/IntelliRehabDSv0/SkeletonData/RawData'
    output_path = '/home/boris.grillborzer/PycharmProjects/SecondOrial/IntelliRehabDSv0/SkeletonData_csv_echt'
    apply_osp(input_path, output_path)
