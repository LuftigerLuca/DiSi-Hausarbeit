import os
from collections import Counter

import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report, accuracy_score
import plotly.graph_objects as go


class GestureRecognition:
    def __init__(self, base_path):
        X, y = self.load_training_data(base_path)
        self.clf = self.train_classifier(X, y)

    def plot_filter(self, data, title, filename=None):
        """Visualizes the import data"""
        fig = go.Figure()
        for column in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data[column], mode="lines", name=column)
            )

        # Add title and labels
        plot_title = title
        if filename:
            plot_title = f"{title} von {filename}"

        fig.update_layout(title=plot_title, xaxis_title="Zeit", yaxis_title="Werte")
        fig.show()

    # Load and preprocess data
    def load_and_preprocess(self, file_path):
        """Loads and preprocesses accelerometer data."""
        df = pd.read_csv(file_path)
        """Welche Reihen sollen analysiert werden?:"""
        acc_columns = [
            "Linear Acceleration x (m/s^2)",
            "Linear Acceleration y (m/s^2)",
            "Linear Acceleration z (m/s^2)",
        ]
        raw_data = df[acc_columns].dropna()
        ma_data = self.apply_moving_average(raw_data)
        z_data = self.apply_zscore_filter(ma_data)
        lp_data = self.apply_lowpass_filter(z_data)

        return lp_data

    # Z-score filter
    def apply_zscore_filter(self, df):
        """Removes outliers using Z-score filtering."""
        z_scores = np.abs((df - df.mean()) / df.std())
        z_result = df[(z_scores < 3).all(axis=1)]
        return z_result

    # Low-pass filter
    def apply_lowpass_filter(self, df, cutoff=10, fs=100, order=4):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        lp_result = df.apply(lambda x: filtfilt(b, a, x), axis=0)
        return lp_result

    # Moving average filter
    def apply_moving_average(self, df, window_size=5):
        """Applies a moving average filter to the data."""
        ma_result = df.rolling(window=window_size, center=True).mean()
        return ma_result

    # Feature extraction
    def extract_features(self, df):
        """Extracts statistical features from the dataset."""
        features = {}
        for col in df.columns:
            # Amplitude-based features
            features[f"{col}_mean"] = df[col].mean()
            features[f"{col}_std"] = df[col].std()
            features[f"{col}_max"] = df[col].max()
            features[f"{col}_min"] = df[col].min()
            features[f"{col}_range"] = df[col].max() - df[col].min()
            features[f"{col}_median"] = df[col].median()
            features[f"{col}_energy"] = np.sum(df[col] ** 2)

            # Dynamic features
            features[f"{col}_mean_diff"] = df[col].diff().mean()
            features[f"{col}_std_diff"] = df[col].diff().std()
            features[f"{col}_max_diff"] = df[col].diff().max()

            # Frequency-based features
            fft = np.fft.fft(df[col].values)
            fft_magnitude = np.abs(fft[: len(fft) // 2])
            fft_frequencies = np.fft.fftfreq(len(fft), d=1)[: len(fft) // 2]
            features[f"{col}_dominant_freq"] = fft_frequencies[np.argmax(fft_magnitude)]
            features[f"{col}_freq_energy"] = np.sum(fft_magnitude**2)

            for col2 in df.columns:
                if col != col2:
                    features[f"{col}_corr_{col2}"] = df[col].corr(df[col2])

        return features

    # Windowing
    """Größere Windowsize und Overlap = Mehr einzelne Fenster, dafür aber längere Berechnungszeit"""

    def apply_windowing(self, df, window_size=25, overlap=24):
        """Splits the data into overlapping windows and extracts features for each window."""
        step = window_size - overlap
        windows = []
        for start in range(0, len(df) - window_size + 1, step):
            window = df.iloc[start : start + window_size]
            windows.append(self.extract_features(window))
        return pd.DataFrame(windows)

    # Load training data
    def load_training_data(self, base_path):
        """Loads and processes training data from specified directories."""
        print("Loading training data...")
        gestures = ["circle", "shake", "triangle", "square"]
        features, labels = [], []
        for gesture in gestures:
            gesture_path = os.path.join(base_path, gesture)
            for file_name in os.listdir(gesture_path):
                if file_name.endswith(".csv"):
                    file_path = os.path.join(gesture_path, file_name)
                    df = self.load_and_preprocess(file_path)
                    windowed_features = self.apply_windowing(df)
                    features.extend(windowed_features.to_dict(orient="records"))
                    labels.extend([gesture] * len(windowed_features))
        return pd.DataFrame(features), labels

    # Train decision tree classifier
    def train_classifier(self, X, y):
        """Trains and evaluates a decision tree classifier."""
        print("Training classifier...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        return clf

    def predict_new_gesture(self, file_path):
        """Predicts the class of a new gesture."""
        if not os.path.exists(file_path):
            file_path = os.path.join("datasets", file_path)

        if not os.path.exists(file_path):
            print("File not found.")
            return

        df = pd.read_csv(file_path)
        acc_columns = [
            "Linear Acceleration x (m/s^2)",
            "Linear Acceleration y (m/s^2)",
            "Linear Acceleration z (m/s^2)",
        ]
        raw_data = df[acc_columns].dropna()
        self.plot_filter(raw_data, "Input Data")

        processed_data = self.load_and_preprocess(file_path)
        windowed_features = self.apply_windowing(processed_data)
        predictions = self.clf.predict(windowed_features)

        # Anzeigen der Ergebnisse pro Window
        print("Window predictions:")
        prediction_counts = Counter(predictions)
        for gesture, count in prediction_counts.items():
            print(f"{gesture}: {count} windows")

        # Mehrheitliche Abstimmung
        majority_vote = prediction_counts.most_common(1)[0][0]
        print(f"Mehrheitsentscheidung: {majority_vote}")
        return majority_vote
