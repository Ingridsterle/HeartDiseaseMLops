import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Charger le dataset brut
df = pd.read_csv(r"C:\Users\Ingri\Desktop\(Ingrid LBD) Ingé 3 S1\PROJET MLops\heart\Data\raw\heart.csv", sep=";")

# Vérifier les premières lignes et les colonnes
print("Premieres lignes")
print(df.head())
print("Première colonnes")
print(df.columns)

# Nettoyer les colonnes (enlever espaces éventuels et mettre en minuscules)
df.columns = df.columns.str.strip().str.lower()

# Séparer les features et la cible
X = df.drop("target", axis=1)
y = df["target"]

# Séparer en 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer le dossier de sauvegarde s'il n'existe pas
os.makedirs("data/processed", exist_ok=True)

# Sauvegarder les fichiers traités
X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

print("✅ Données traitées et sauvegardées dans data/processed/")
