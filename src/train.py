 

# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 18:20:30 2025

@author: Ingri
"""

import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import joblib
import mlflow
import os
import matplotlib.pyplot as plt

# -------------------------------
# 1️⃣ Charger les données traitées
# -------------------------------
X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv")

# -------------------------------
# 2️⃣ Créer les dossiers nécessaire
# -------------------------------
os.makedirs("models", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

# -------------------------------
# 3️⃣ Définir l'expérience MLflow
# -------------------------------
mlflow.set_experiment("HeartDiseasePipeline")

with mlflow.start_run():
    # -------------------------------
    # 4️⃣ Définir et entraîner le modèle
    # -------------------------------
    model = MLPRegressor(hidden_layer_sizes=(10, 10),
                         max_iter=500,
                         random_state=42)
    model.fit(X_train, y_train.values.ravel())

    # -------------------------------
    # 5️⃣ Prédictions et métriques
    # -------------------------------
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"📉 Mean Squared Error: {mse:.4f}")
    print(f"📈 R2 Score: {r2:.4f}")

    # Logger paramètres et métriques
    mlflow.log_param("hidden_layers", (10, 10))
    mlflow.log_param("max_iter", 500)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("R2", r2)

    # -------------------------------
    # 6️Sauvegarder le modèle
    # -------------------------------
    model_path = "models/heart_model.pkl"
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path)

    # -------------------------------
    # 7️Graphique : Réel vs Prédit
    # -------------------------------
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Valeurs réelles')
    plt.ylabel('Prédictions')
    plt.title('Réel vs Prédit')
    real_pred_path = 'artifacts/real_vs_pred.png'
    plt.savefig(real_pred_path)
    mlflow.log_artifact(real_pred_path)

    # -------------------------------
    # 8️⃣ Histogramme des résidus
    # -------------------------------
    residus = y_test.values.ravel() - y_pred
    plt.figure(figsize=(6,4))
    plt.hist(residus, bins=20, color='green')
    plt.xlabel('Erreur (y_true - y_pred)')
    plt.ylabel('Fréquence')
    plt.title('Histogramme des résidus')
    residus_path = 'artifacts/residus.png'
    plt.savefig(residus_path)
    mlflow.log_artifact(residus_path)

    # -------------------------------
    # 9️⃣ Importance des features (Permutation)
    # -------------------------------
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    importances = result.importances_mean
    plt.figure(figsize=(8,4))
    plt.bar(X_test.columns, importances, color='orange')
    plt.xticks(rotation=45)
    plt.ylabel('Importance moyenne (permutation)')
    plt.title('Feature Importances')
    feature_path = 'artifacts/feature_importances.png'
    plt.savefig(feature_path)
    mlflow.log_artifact(feature_path)

    # -------------------------------
    # 10️⃣ Log du dataset prédictions
    # -------------------------------
    df = pd.DataFrame({'y_true': y_test.values.ravel(), 'y_pred': y_pred})
    csv_path = 'artifacts/predictions.csv'
    df.to_csv(csv_path, index=False)
    mlflow.log_artifact(csv_path)

    print("Run complet loggé avec paramètres, métriques, graphiques et dataset")
