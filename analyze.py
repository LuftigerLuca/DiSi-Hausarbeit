import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Funktionen zur Datenverarbeitung und Analyse
def preprocess_data(df):
    """
    Vorverarbeitung der Daten: Fehlende Werte entfernen, Moving Average Filter anwenden.
    """
    # Fehlende Werte entfernen
    df = df.dropna()

    # Moving Average Filter auf alle Beschleunigungs-Spalten anwenden
    cols_to_filter = [
        "Linear Acceleration x (m/s^2)",
        "Linear Acceleration y (m/s^2)",
        "Linear Acceleration z (m/s^2)",
        "Absolute acceleration (m/s^2)"
    ]
    for col in cols_to_filter:
        df[col] = df[col].rolling(window=5, center=True).mean()

    return df

def extract_features(df):
    """
    Extrahiert Features aus den Sensordaten eines DataFrames.
    """
    features = {}
    cols = [
        "Linear Acceleration x (m/s^2)",
        "Linear Acceleration y (m/s^2)",
        "Linear Acceleration z (m/s^2)",
        "Absolute acceleration (m/s^2)"
    ]
    for col in cols:
        # Statistische Merkmale
        features[f"{col}_mean"] = df[col].mean()
        features[f"{col}_std"] = df[col].std()
        features[f"{col}_max"] = df[col].max()
        features[f"{col}_min"] = df[col].min()
        # Dynamische Merkmale (z. B. Bereich, Energie)
        features[f"{col}_range"] = df[col].max() - df[col].min()
        features[f"{col}_energy"] = np.sum(df[col] ** 2) / len(df[col])

    return features

# Einlesen der Dateien
data_files = {
    "Circle_1": "datasets/circle/Circle_1.csv",
    "Circle_2": "datasets/circle/Circle_2.csv",
    "Circle_3": "datasets/circle/Circle_3.csv",
    "Circle_4": "datasets/circle/Circle_4.csv",
    "Shake_1": "datasets/shake/Shake_1.csv",
    "Shake_2": "datasets/shake/Shake_2.csv",
    "Shake_3": "datasets/shake/Shake_3.csv",
    "Shake_4": "datasets/shake/Shake_4.csv",
    "Triangle_1": "datasets/triangle/Triangle_1.csv",
    "Triangle_2": "datasets/triangle/Triangle_2.csv",
    "Triangle_3": "datasets/triangle/Triangle_3.csv",
    "Triangle_4": "datasets/triangle/Triangle_4.csv",
    "Triangle_5": "datasets/triangle/Triangle_5.csv",
    "Triangle_6": "datasets/triangle/Triangle_6.csv",
}

# Daten einlesen und vorverarbeiten
dataframes = {name: pd.read_csv(path) for name, path in data_files.items()}
processed_data = {name: preprocess_data(df) for name, df in dataframes.items()}

# Feature Engineering
feature_data = {name: extract_features(df.dropna()) for name, df in processed_data.items()}
features_df = pd.DataFrame.from_dict(feature_data, orient="index")
features_df.reset_index(inplace=True)
features_df.rename(columns={"index": "Label"}, inplace=True)
features_df["Label"] = features_df["Label"].str.extract(r'([A-Za-z]+)_')[0]  # Labels extrahieren

# Training und Evaluierung eines Entscheidungsbaum-Modells
X = features_df.drop(columns=["Label"])
y = features_df["Label"]

# Daten aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entscheidungsbaum trainieren
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Modell evaluieren
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

plt.figure(figsize=(12, 6))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=clf.classes_)
plt.show()

print("Genauigkeit:", accuracy)
print("Klassifikationsbericht:\n", report)

def classify_new_file(file_path):
    """
    Klassifiziert eine neue Datei, um die Bewegung zu bestimmen.
    """
    try:
        # Datei einlesen und vorverarbeiten
        new_data = pd.read_csv(file_path)
        new_data = preprocess_data(new_data)

        # Features extrahieren
        new_features = extract_features(new_data)
        new_features_df = pd.DataFrame([new_features])

        # Klassifikation
        prediction = clf.predict(new_features_df)
        print(f"Die vorhergesagte Bewegung ist: {prediction[0]}")
    except Exception as e:
        print(f"Fehler bei der Klassifikation der Datei: {e}")

# Beispielaufruf der Klassifikation
classify_new_file("datasets/Carlo_Triangle.csv")