# DiSi-Hausarbeit

## Idee

Nach zeichnen einer Form in der Luft (Kreis, Dreick, Viereck), wird dieser durch den Bot auf ein Papier gezeichnet.

## Aufgabe:

- Ein Programm entwickeln, welches auf zwei verschiedene Steuerungsbefehle mit jeweils einer spezifischen Aktion
  reagiert.
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

- [x] Eine Funktion zum Einlesen der Sensordaten, welche diese auch gleich vorverarbeitet mit Filtern wie Moving
  Average, Median Filter, etc.
- [x] Eine Funktion zum Feature Engineering, welche die Sensordaten in ein geeignetes Format für den Entscheidungsbaum
  bringt.
- [x]  Funktion zum Trainieren des Entscheidungsbaums.
- [x] Eine Funktion zum Klassifizieren neuer Daten und Ausführen der Aktionen.
- [x] Eine Funktion zum Zeichnen des Kreises oder Dreiecks.

