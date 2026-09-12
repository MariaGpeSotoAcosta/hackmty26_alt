import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Cargar los datos ya procesados (no hace falta re-extraer features)
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_val = np.load("X_val.npy")
y_val = np.load("y_val.npy")

# Entrenar
modelo = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
modelo.fit(X_train, y_train)

# Evaluar en TRAIN (para detectar overfitting)
y_pred_train = modelo.predict(X_train)
print("--- Resultado en TRAIN (el modelo YA vio estos datos) ---")
print(classification_report(y_train, y_pred_train, target_names=["human", "synthetic"]))

# Evaluar en VAL (el examen real)
y_pred_val = modelo.predict(X_val)
print("\n--- Resultado en VAL (el modelo NUNCA vio estos datos) ---")
print(confusion_matrix(y_val, y_pred_val))
print(classification_report(y_val, y_pred_val, target_names=["human", "synthetic"]))

# Guardar el modelo entrenado
joblib.dump(modelo, "modelo_entrenado.pkl")
print("\nModelo guardado en modelo_entrenado.pkl")