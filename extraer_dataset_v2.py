import pandas as pd
import soundfile as sf
import librosa
import numpy as np
import time

def extraer_features_v2(filepath):
    audio, sr = sf.read(filepath)
    canal_caller = audio[:, 0]

    # MFCC base (igual que antes)
    mfccs = librosa.feature.mfcc(y=canal_caller, sr=sr, n_mfcc=13)

    # NUEVO: delta y delta-delta (velocidad y aceleracion de cambio del MFCC)
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)

    # NUEVO: features espectrales
    centroid = librosa.feature.spectral_centroid(y=canal_caller, sr=sr)
    flatness = librosa.feature.spectral_flatness(y=canal_caller)
    zcr = librosa.feature.zero_crossing_rate(canal_caller)

    # Resumimos TODO con mean + std, igual patron que antes
    def resumir(matriz):
        return np.concatenate([np.mean(matriz, axis=1), np.std(matriz, axis=1)])

    features = np.concatenate([
        resumir(mfccs),         # 26 numeros (13 mean + 13 std)
        resumir(delta_mfccs),   # 26 numeros
        resumir(delta2_mfccs),  # 26 numeros
        resumir(centroid),      # 2 numeros
        resumir(flatness),      # 2 numeros
        resumir(zcr),           # 2 numeros
    ])
    return features  # total: 84 numeros

df = pd.read_csv("manifest.csv")

X_train, y_train = [], []
X_val, y_val = [], []

inicio = time.time()

for i, row in df.iterrows():
    filepath = f"audio/{row['anon_id']}.wav"
    try:
        features = extraer_features_v2(filepath)
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

    if (i + 1) % 25 == 0:
        print(f"Procesadas {i+1}/{len(df)} llamadas ({time.time()-inicio:.1f}s)")

X_train, y_train = np.array(X_train), np.array(y_train)
X_val, y_val = np.array(X_val), np.array(y_val)

print("\nX_train:", X_train.shape, "X_val:", X_val.shape)

np.save("X_train_v2.npy", X_train)
np.save("y_train_v2.npy", y_train)
np.save("X_val_v2.npy", X_val)
np.save("y_val_v2.npy", y_val)
print("Guardado con sufijo _v2")