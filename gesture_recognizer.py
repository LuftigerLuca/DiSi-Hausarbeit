#Write gesture recognition here
import os

import pandas as pd
import plotly.graph_objs as go


def data_input(filename):
    try:
        dataframe = pd.read_csv(os.getcwd() + filename, delimiter=',')
        print("Data loaded successfully")
        print(dataframe.head())
        return dataframe

    except Exception as e:
        print(f"Error loading data: {e}")
        return None


'''Funktion erstmal zum Visualisieren, um zu checken ob die Bereinigung funktioniert hat'''
def plot_filter(data, title):
    fig = go.Figure()
    for column in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data[column], mode='lines', name=column))

    fig.update_layout(title=title,
                      xaxis_title='Zeit',
                      yaxis_title='Werte')
    fig.show()


daten = data_input("/datasets/circle/circle_accelerometer.csv")
plot_filter(daten, "Input")