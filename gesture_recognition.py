import os
import pickle
from collections import Counter
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score
import plotly.graph_objects as go


class GestureRecognition:
    def __init__(self, base_path, model_path='gesture_model.pkl'):
        self.model_path = model_path
        self.feature_names = None
        if os.path.exists(model_path):
            self.load_model()
        else:
            X, y = self.load_training_data(base_path)
            self.feature_names = X.columns.tolist()
            self.clf = self.train_classifier(X, y)
            self.visualize_decision_tree()
            self.save_model()


    def save_model(self):
        """Saves the trained model to a pickle file."""
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.clf, f)
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        """Loads the trained model from a pickle file."""
        with open(self.model_path, 'rb') as f:
            self.clf = pickle.load(f)
        print(f"Model loaded from {self.model_path}")

    def plot_filter(self, data, title, filename=None):
        """Visualizes the input data"""
        fig = go.Figure()
        for column in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data[column], mode="lines", name=column)
            )
        plot_title = f"{title} {'from ' + filename if filename else ''}"
        fig.update_layout(title=plot_title, xaxis_title="Time", yaxis_title="Values")
        fig.show()

    def load_and_preprocess(self, file_path):
        """Loads and preprocesses accelerometer data."""

        df = pd.read_csv(file_path)
        acc_columns = [
            "Linear Acceleration x (m/s^2)",
            "Linear Acceleration y (m/s^2)",
            "Linear Acceleration z (m/s^2)",
        ]
        raw_data = df[acc_columns]
        ma_data = self.apply_moving_average(raw_data)
        z_data = self.apply_zscore_filter(ma_data)


        return z_data

    def apply_zscore_filter(self, df):
        """Removes outliers using Z-score filtering."""
        z_scores = np.abs((df - df.mean()) / df.std())
        return df[(z_scores < 3).all(axis=1)]


    def apply_moving_average(self, df, window_size=30):
        """Applies a moving average filter to smooth the data."""
        return df.rolling(window=window_size, center=True).mean()

    def extract_features(self, df):
        """Extracts relevant statistical features from the dataset."""
        features = {}
        for col in df.columns:
            # Core statistical features
            features[f"{col}_mean"] = df[col].mean()
            features[f"{col}_std"] = df[col].std()
            features[f"{col}_max"] = df[col].max()
            features[f"{col}_min"] = df[col].min()
            features[f"{col}_energy"] = np.sum(df[col] ** 2)
            features[f"{col}_mean_diff"] = df[col].diff().mean()
            features[f"{col}_std_diff"] = df[col].diff().std()
        return features

    def apply_windowing(self, df, window_size=100, overlap=50):
        """Splits the data into overlapping windows with larger window size."""
        step = window_size - overlap
        windows = []
        for start in range(0, len(df) - window_size + 1, step):
            window = df.iloc[start:start + window_size]
            windows.append(self.extract_features(window))
        return pd.DataFrame(windows)

    def load_training_data(self, base_path):
        """Loads and processes training data with expanded dataset."""
        print("Loading training data...")
        gestures = ["circle", "triangle", "square"]
        features, labels = [], []

        for gesture in gestures:
            gesture_path = os.path.join(base_path, gesture)
            if not os.path.exists(gesture_path):
                print(f"Warning: Path {gesture_path} does not exist")
                continue

            for file_name in os.listdir(gesture_path):
                if file_name.endswith(".csv"):
                    file_path = os.path.join(gesture_path, file_name)
                    try:
                        df = self.load_and_preprocess(file_path)
                        windowed_features = self.apply_windowing(df)
                        features.extend(windowed_features.to_dict(orient="records"))
                        labels.extend([gesture] * len(windowed_features))
                    except Exception as e:
                        print(f"Error processing {file_path}: {str(e)}")
        return pd.DataFrame(features), labels

    def train_classifier(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )


        clf = DecisionTreeClassifier(max_depth=3)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        return clf

    def predict_new_gesture(self, file_path):
        """Predicts the class of a new gesture with improved window handling."""

        if not os.path.exists(file_path):
            file_path = os.path.join("datasets", file_path)
            if not os.path.exists(file_path):
                print("File not found.")
                return None

        # Load and preprocess the data
        self.plot_filter(pd.read_csv(file_path), "Raw Data", file_path)
        df = self.load_and_preprocess(file_path)
        self.plot_filter(df, "Processed Data", file_path)


        # Extract features from windows
        windowed_features = self.apply_windowing(df)
        if len(windowed_features) == 0:
            print("No valid windows found in the data.")
            return None

        new_predictions = self.clf.predict(windowed_features)

        prediction_counts = Counter(new_predictions)
        total_windows = len(new_predictions)
        print("\nPrediction confidence:")
        for gesture, count in prediction_counts.items():
            confidence = (count / total_windows) * 100
            print(f"{gesture}: {confidence:.1f}% ({count} windows)")
        majority_vote = prediction_counts.most_common(1)[0]
        return majority_vote[0]

    def visualize_decision_tree(self):
        """Visualisiert den Decision Tree"""
        if not hasattr(self, 'clf') or not self.feature_names:
            print("Kein trainierter Classifier oder Feature-Namen gefunden!")
            return

        # Erstelle Figure für den Tree
        plt.figure(figsize=(20, 10))
        plot_tree(self.clf,
                  feature_names=self.feature_names,
                  class_names=self.clf.classes_,
                  filled=True,
                  rounded=True,
                  fontsize=10)
        plt.title('Decision Tree für Gesture Recognition')
        plt.show()