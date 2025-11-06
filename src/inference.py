# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 18:22:50 2025

@author: Ingri
"""

import joblib
import numpy as np

# Charger le modèle entraîné
model = joblib.load("models/heart_model.pkl")

# Exemple de nouvelle donnée (à remplacer par une vraie observation)
new_data = np.array([[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]])  # exemple

# Prédiction
prediction = model.predict(new_data)
print(f"🩺 Prédiction (1 = maladie cardiaque, 0 = sain) : {prediction[0]}")
