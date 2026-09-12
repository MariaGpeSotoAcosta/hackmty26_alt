import soundfile as sf
import librosa
import numpy as np

# Cargar el audio
audio, sr = sf.read("audio/call_0181ce113ebe.wav")

# Separar canal 0 (caller) del canal 1 (agente)
canal_caller = audio[:, 0]
canal_agente = audio[:, 1]

print("Canal caller shape:", canal_caller.shape)
print("Canal agente shape:", canal_agente.shape)

# Extraer MFCC del caller (que es lo que nos interesa clasificar)
mfccs = librosa.feature.mfcc(y=canal_caller, sr=sr, n_mfcc=13)
print("MFCC shape:", mfccs.shape)

# Nuestro "embedding casero": mean + std
mean = np.mean(mfccs, axis=1)
std = np.std(mfccs, axis=1)
features = np.concatenate([mean, std])

print("Features shape (deberia ser 26):", features.shape)
print("Primeros 5 valores de features:", features[:5])