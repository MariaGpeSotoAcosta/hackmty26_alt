import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report

# Cargar ambos tipos de features
X_ac_train = np.load("X_train_v2.npy")
X_ac_val = np.load("X_val_v2.npy")
X_conv_train = np.load("X_conv_train.npy")
X_conv_val = np.load("X_conv_val.npy")
y_train = np.load("y_train_v2.npy")
y_val = np.load("y_val_v2.npy")

# Combinar: cada fila ahora tiene features acusticas + conversacionales juntas
X_train_combo = np.concatenate([X_ac_train, X_conv_train], axis=1)
X_val_combo = np.concatenate([X_ac_val, X_conv_val], axis=1)

print("Shape combinado:", X_train_combo.shape)  # deberia ser (282, 92) -> 84 + 8

modelo = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
modelo.fit(X_train_combo, y_train)
y_pred = modelo.predict(X_val_combo)

print("\n--- Modelo COMBINADO (acustico + conversacional) ---")
print(classification_report(y_val, y_pred, target_names=["human", "synthetic"]))

# Cross-validation para confirmar robustez, igual que antes
X_todo = np.concatenate([X_train_combo, X_val_combo])
y_todo = np.concatenate([y_train, y_val])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(modelo, X_todo, y_todo, cv=cv, scoring="f1_macro")
print(f"\nCross-validation F1 macro: {scores.mean():.3f} (+/- {scores.std():.3f})")

# Bonus: solo features conversacionales, para ver cuanto aportan SOLAS
modelo_solo_conv = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
modelo_solo_conv.fit(X_conv_train, y_train)
y_pred_conv = modelo_solo_conv.predict(X_conv_val)
print("\n--- Modelo SOLO conversacional (para referencia) ---")
print(classification_report(y_val, y_pred_conv, target_names=["human", "synthetic"]))