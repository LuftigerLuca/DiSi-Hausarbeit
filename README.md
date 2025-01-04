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
- Sensordaten einlesen und vorverarbeiten/bereinigen
- Feature Engineering der Sensordaten
- Entscheidungsbaum trainieren und validieren
- [x] Programmcode für die Dobot-Aktionen
- Trainierten Entscheidungsbaum nutzen, also neue Daten klassifizieren und Aktionen ausführen („Inferenz“)
  

## Dateien:
- Eine Funktion zum einlesen der CSV Daten