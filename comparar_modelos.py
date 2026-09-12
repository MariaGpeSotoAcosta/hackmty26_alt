import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score

X_train = np.load("X_train_v2.npy")
y_train = np.load("y_train_v2.npy")
X_val = np.load("X_val_v2.npy")
y_val = np.load("y_val_v2.npy")

# Algunos modelos (SVM y LogisticRegression) funcionan mejor si las features
# estan en la misma escala, asi que las normalizamos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

modelos = {
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "SVM": SVC(probability=True, class_weight="balanced", random_state=42),
}

resultados = []

for nombre, modelo in modelos.items():
    if nombre in ["LogisticRegression", "SVM"]:
        modelo.fit(X_train_scaled, y_train)
        y_pred = modelo.predict(X_val_scaled)
    else:
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_val)

    f1 = f1_score(y_val, y_pred, average="macro")
    resultados.append((nombre, f1))

    print(f"\n=== {nombre} (F1 macro = {f1:.3f}) ===")
    print(classification_report(y_val, y_pred, target_names=["human", "synthetic"]))

print("\n--- Resumen ---")
for nombre, f1 in sorted(resultados, key=lambda x: -x[1]):
    print(f"{nombre}: F1 macro = {f1:.3f}")