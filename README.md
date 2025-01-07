# DiSi-Hausarbeit

## Idee
Nach zeichnen einer Form in der Luft (Kreis, Dreick), wird dieser durch den Bot auf ein Papier gezeichnet.


## Aufgabe:
- Ein Programm entwickeln, welches auf zwei verschiedene Steuerungsbefehle mit jeweils einer spezifischen Aktion reagiert.
- In unserem Fall soll der Bot entweder einen Kreis oder ein Dreieck zeichnen.
- Die Steuerungsbefehle sollen aus einem eigens antrainierten Entscheidungsbaum stammen.
- Signale kommen von einem Handy (unseres oder aus dem Labor)
- Beide Befehle sollen von der gleichen Position aus gemessen werden.
- Ergebnis muss reproduzierbar sein.

## Geforderte Programmkomponenten:
- [x] Sensordaten einlesen und vorverarbeiten/bereinigen
- [x] Feature Engineering der Sensordaten
- [x] Entscheidungsbaum trainieren und validieren
- [x] Programmcode für die Dobot-Aktionen
- [x] Trainierten Entscheidungsbaum nutzen, also neue Daten klassifizieren und Aktionen ausführen („Inferenz“)
  

## Wichtige Programmteile:
- [x] Eine Funktion zum Einlesen der Sensordaten, welche diese auch gleich vorverarbeitet mit Filtern wie Moving Average, Median Filter, etc.
- [x] Eine Funktion zum Feature Engineering, welche die Sensordaten in ein geeignetes Format für den Entscheidungsbaum bringt.
- [x]  Funktion zum Trainieren des Entscheidungsbaums.
- [x] Eine Funktion zum Klassifizieren neuer Daten und Ausführen der Aktionen.
- [x] Eine Funktion zum Zeichnen des Kreises oder Dreiecks. (Haben wir schon gemacht, weil wir Macher sind)

# Änderungen:

- Dateien wie Triangle.csv oder Circle.csv sind 30 Sekunden aufgenommen worden (kontinuierliche Gesture)
- Dateien wie circle.csv von Luca sind jeweils nur eine Bewegung
- Neue gesture_recognition ist erstmal als gesture2 gelabelt, falls man die andere noch braucht. Dabei werden nur circle und triangle erkannt.
- Gleiches für die main
- Programm schaut nun ob eine gesture_model.plk da ist. Falls nein wird eine neu erstellt. Vorherige vorher löschen falls man das nochmal machen will.
- Danach lädt er das Model und man kann seine Datei einlesen.
- Features noch anpassen.


