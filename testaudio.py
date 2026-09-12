import soundfile as sf

# Usa el mismo anon_id que viste en tu manifest (primera fila)
audio, sr = sf.read("audio/call_0181ce113ebe.wav")

print("Sample rate:", sr)
print("Shape del audio:", audio.shape)
print("Duracion (segundos):", len(audio) / sr)