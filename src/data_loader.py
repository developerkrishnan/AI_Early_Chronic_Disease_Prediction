import pandas as pd
import numpy as np
import os

DATA_DIR = "data"

def load_diabetes_data():
    """Loads Diabetes dataset. Adds headers as they are missing in the raw file."""
    path = os.path.join(DATA_DIR, "diabetes.csv")
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    try:
        df = pd.read_csv(path, names=columns, header=None)
        # Ensure Outcome is numeric
        df['Outcome'] = pd.to_numeric(df['Outcome'], errors='coerce')
        df.dropna(inplace=True)
        return df
    except FileNotFoundError:
        print("Diabetes data file not found.")
        return None

def load_heart_data():
    """Loads Heart Disease dataset."""
    path = os.path.join(DATA_DIR, "heart.csv")
    try:
        df = pd.read_csv(path)
        # Heart dataset usually has 'target' column
        return df
    except FileNotFoundError:
        print("Heart data file not found.")
        return None

def load_kidney_data():
    """Loads Kidney Disease dataset."""
    path = os.path.join(DATA_DIR, "kidney.csv")
    try:
        df = pd.read_csv(path)
        # Handle '?' values if any remain (though final.csv should be clean)
        df.replace('?', np.nan, inplace=True)
        df.dropna(inplace=True)
        return df
    except FileNotFoundError:
        print("Kidney data file not found.")
        return None
