import numpy as np
from sklearn.ensemble import RandomForestClassifier

X_train = np.load("X_train_v2.npy")
y_train = np.load("y_train_v2.npy")

# Nombres de cada bloque de features, en el mismo orden que las generamos
nombres = (
    [f"mfcc_mean_{i}" for i in range(13)] + [f"mfcc_std_{i}" for i in range(13)] +
    [f"delta_mean_{i}" for i in range(13)] + [f"delta_std_{i}" for i in range(13)] +
    [f"delta2_mean_{i}" for i in range(13)] + [f"delta2_std_{i}" for i in range(13)] +
    ["centroid_mean", "centroid_std"] +
    ["flatness_mean", "flatness_std"] +
    ["zcr_mean", "zcr_std"]
)

modelo = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
modelo.fit(X_train, y_train)

importancias = modelo.feature_importances_

# Ordenar de mas a menos importante
orden = np.argsort(importancias)[::-1]

print("--- Top 10 features mas importantes segun el modelo ---")
for i in orden[:10]:
    print(f"{nombres[i]}: {importancias[i]:.4f}")

# Ver si alguna feature separa PERFECTAMENTE las dos clases (senal de shortcut)
print("\n--- Revisando separacion perfecta por feature individual ---")
for i in orden[:10]:
    valores_human = X_train[y_train == 0, i]
    valores_synthetic = X_train[y_train == 1, i]
    print(f"\n{nombres[i]}:")
    print(f"  human      -> min={valores_human.min():.4f}  max={valores_human.max():.4f}  mean={valores_human.mean():.4f}")
    print(f"  synthetic  -> min={valores_synthetic.min():.4f}  max={valores_synthetic.max():.4f}  mean={valores_synthetic.mean():.4f}")