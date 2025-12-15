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
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Charger les données traitées
X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv")

# Créer le dossier models s'il n'existe pas
os.makedirs("models", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)  # pour MLflow

# Définir l'expérience MLflow
mlflow.set_experiment("Présentation du 16 Decembre 2025")

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

    # --- Graphique 1 : prédictions vs vrai ---
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred)
    plt.xlabel("Vrai")
    plt.ylabel("Prédit")
    plt.title("Prédictions vs Vérité")
    plot_path1 = "artifacts/pred_vs_true.png"
    plt.savefig(plot_path1)
    mlflow.log_artifact(plot_path1)

    # --- Graphique 2 : distribution des features ---
    plt.figure(figsize=(10,6))
    X_train_norm = (X_train - X_train.mean())/X_train.std()  # normalisation simple
    sns.boxplot(data=X_train_norm)
    plt.xticks(rotation=45)
    plt.title("Distribution normalisée des features")
    plot_path2 = "artifacts/features_distribution.png"
    plt.savefig(plot_path2)
    mlflow.log_artifact(plot_path2)

    # --- Graphique 3 : histogramme des erreurs ---
    errors = y_test.values.ravel() - y_pred
    plt.figure(figsize=(6,4))
    plt.hist(errors, bins=10)
    plt.xlabel("Erreur (y_true - y_pred)")
    plt.ylabel("Nombre d'observations")
    plt.title("Distribution des erreurs")
    plot_path3 = "artifacts/errors_histogram.png"
    plt.savefig(plot_path3)
    mlflow.log_artifact(plot_path3)

    # --- CSV de prédictions ---
    df_pred = pd.DataFrame({
        'y_true': y_test.values.ravel(),
        'y_pred': y_pred
    })
    csv_path = "artifacts/predictions.csv"
    df_pred.to_csv(csv_path, index=False)
    mlflow.log_artifact(csv_path)

    print("Run complet loggé avec paramètres, métriques, graphiques et dataset")
