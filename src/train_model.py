import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from data_loader import load_diabetes_data, load_heart_data, load_kidney_data

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_diabetes_model():
    print("Training Diabetes Model...")
    df = load_diabetes_data()
    if df is None: return
    
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Diabetes Model Accuracy: {acc:.4f}")
    
    with open(os.path.join(MODEL_DIR, "diabetes_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    print("Saved Diabetes Model.")

def train_heart_model():
    print("Training Heart Disease Model...")
    df = load_heart_data()
    if df is None: return
    
    target_col = 'target'
    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found in Heart dataset.")
        print(f"Columns found: {df.columns.tolist()}")
        return

    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(max_iter=3000, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Heart Disease Model Accuracy: {acc:.4f}")
    
    with open(os.path.join(MODEL_DIR, "heart_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    print("Saved Heart Model.")

def train_kidney_model():
    print("Training Kidney Disease Model...")
    df = load_kidney_data()
    if df is None: return

    # Drop ID if exists
    if 'id' in df.columns:
        df.drop('id', axis=1, inplace=True)

    # Encode categorical variables and save encoders
    le_dict = {}
    for col in df.columns:
        if df[col].dtype == 'object':
             le = LabelEncoder()
             df[col] = le.fit_transform(df[col].astype(str))
             le_dict[col] = le
    
    target_col = 'class'
    if target_col not in df.columns:
         print("Target column 'class' not found in Kidney dataset.")
         return

    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Kidney Disease Model Accuracy: {acc:.4f}")
    
    with open(os.path.join(MODEL_DIR, "kidney_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    
    with open(os.path.join(MODEL_DIR, "kidney_encoders.pkl"), "wb") as f:
        pickle.dump(le_dict, f)
    print("Saved Kidney Model and Encoders.")

if __name__ == "__main__":
    train_diabetes_model()
    train_heart_model()
    train_kidney_model()
