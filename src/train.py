# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 18:20:30 2025

@author: Ingri
"""

import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import mlflow
import os

# Charger les données traitées
X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv")

# Créer le dossier models s'il n'existe pas
os.makedirs("models", exist_ok=True)

# Définir l'expérience MLflow
mlflow.set_experiment("HeartDiseasePipeline")

with mlflow.start_run():
    # Définir un petit réseau de neurones pour régression
    model = MLPRegressor(hidden_layer_sizes=(10, 10),
                         max_iter=500,
                         random_state=42)

    # Entraîner le modèle
    model.fit(X_train, y_train.values.ravel())

    # Prédire sur le jeu de test
    y_pred = model.predict(X_test)

    # Évaluer avec des métriques de régression
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"📉 Mean Squared Error: {mse:.4f}")
    print(f"📈 R2 Score: {r2:.4f}")

    # Logger les paramètres et métriques dans MLflow
    mlflow.log_param("hidden_layers", (10, 10))
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("R2", r2)

    # Sauvegarder le modèle
    model_path = "models/heart_model.pkl"
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path)

print("✅ Modèle de régression entraîné et sauvegardé dans models/")
