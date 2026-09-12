import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

X_train = np.load("X_train_v2.npy")
y_train = np.load("y_train_v2.npy")
X_val = np.load("X_val_v2.npy")
y_val = np.load("y_val_v2.npy")

# Entrenamos con TODO (train + val) para el modelo final que va al endpoint,
# ya que val ya cumplio su proposito de evaluarnos honestamente
X_final = np.concatenate([X_train, X_val])
y_final = np.concatenate([y_train, y_val])

modelo = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
modelo.fit(X_final, y_final)

joblib.dump(modelo, "modelo_final_v2.pkl")
print("Modelo final guardado: modelo_final_v2.pkl")
print(f"Entrenado con {len(X_final)} llamadas (train + val combinados)")