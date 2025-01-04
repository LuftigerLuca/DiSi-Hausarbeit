import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def load_and_preprocess_data():
    # Dateinamen der Bewegungsdaten
    circle_files = [f"datasets/circle/Circle_{i}.csv" for i in range(1, 6)]
    shake_files = [f"datasets/shake/Shake_{i}.csv" for i in range(1, 6)]
    triangle_files = [f"datasets/triangle/Triangle_{i}.csv" for i in range(1, 8)]

    dfs = []

    # Daten einlesen und Label hinzufügen
    for files, label in zip([circle_files, shake_files, triangle_files], ['Circle', 'Shake', 'Triangle']):
        for file in files:
            df = pd.read_csv(file)
            df['Label'] = label
            dfs.append(df)

    # Alle Daten in einem DataFrame kombinieren
    data = pd.concat(dfs, ignore_index=True)

    return data


def engineer_features(data):
    # Statistische Kennzahlen als Features berechnen
    features = data.groupby(['Label']).agg([np.mean, np.std, np.min, np.max])
    features.columns = ['_'.join(col) for col in features.columns]
    features = features.reset_index()

    return features


def train_decision_tree(features):
    # Features und Label trennen
    X = features.drop('Label', axis=1)
    y = features['Label']

    # Daten in Trainings- und Testset aufteilen
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entscheidungsbaum trainieren
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Genauigkeit auf Testdaten überprüfen
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    return clf


def predict_new_motion(model, new_data_file):
    # Neue Bewegungsdaten laden
    new_data = pd.read_csv(new_data_file)

    # Features für neue Daten berechnen
    new_features = new_data.agg([np.mean, np.std, np.min, np.max])

    # Vorhersage treffen
    prediction = model.predict(new_features.values.reshape(1, -1))[0]

    print(f"Die Bewegung in {new_data_file} wurde als {prediction} klassifiziert.")


# Main
data = load_and_preprocess_data()
features = engineer_features(data)
model = train_decision_tree(features)

# Beispielvorhersage für neue Daten
predict_new_motion(model, 'datasets/circle/Circle_1.csv')