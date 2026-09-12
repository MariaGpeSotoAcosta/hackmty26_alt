import pandas as pd
import soundfile as sf
import librosa
import numpy as np
import time

def extraer_features(filepath):
    audio, sr = sf.read(filepath)
    canal_caller = audio[:, 0]
    mfccs = librosa.feature.mfcc(y=canal_caller, sr=sr, n_mfcc=13)
    mean = np.mean(mfccs, axis=1)
    std = np.std(mfccs, axis=1)
    return np.concatenate([mean, std])

df = pd.read_csv("manifest.csv")

X_train, y_train = [], []
X_val, y_val = [], []

inicio = time.time()

for i, row in df.iterrows():
    filepath = f"audio/{row['anon_id']}.wav"

    try:
        features = extraer_features(filepath)
    except Exception as e:
        print(f"Error con {row['anon_id']}: {e}")
        continue

    label = 1 if row["label"] == "synthetic" else 0

    if row["split"] == "train":
        X_train.append(features)
        y_train.append(label)
    else:
        X_val.append(features)
        y_val.append(label)

    # Mostrar progreso cada 25 llamadas
    if (i + 1) % 25 == 0:
        transcurrido = time.time() - inicio
        print(f"Procesadas {i+1}/{len(df)} llamadas ({transcurrido:.1f}s transcurridos)")

X_train = np.array(X_train)
y_train = np.array(y_train)
X_val = np.array(X_val)
y_val = np.array(y_val)

print("\n--- Resultado final ---")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val:", X_val.shape, "y_val:", y_val.shape)
print(f"Tiempo total: {time.time() - inicio:.1f} segundos")

# Guardamos en disco para no tener que repetir esto despues
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_val.npy", X_val)
np.save("y_val.npy", y_val)
print("\nGuardado: X_train.npy, y_train.npy, X_val.npy, y_val.npy")