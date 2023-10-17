import numpy as np
import sounddevice as sd
import wave
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram

# Parámetros de grabación
sample_rate = 48000  # Frecuencia de muestreo en Hz
duration = 10  # Duración de la grabación en segundos

# Grabación de la señal de audio original
print("Grabando audio original...")
audio_input = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1)
sd.wait()

# Guardar audio original
with wave.open('original_audio.wav', 'w') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)  # 16 bits
    wf.setframerate(sample_rate)
    wf.writeframes((audio_input.flatten() * 32767).astype(np.int16).tobytes())
print("Grabación de audio original completada.")

# Aplicar la transformada de Fourier
audio_fft = np.fft.fft(audio_input.flatten())
frequencies = np.fft.fftfreq(len(audio_fft), d=1/sample_rate)

# Identificar las frecuencias de ruido (ajustar según sea necesario)
noise_freq_indices = np.where((frequencies > 500) & (frequencies < 5000))[0]

# Ajustar ganancias para resaltar la voz y atenuar el ruido
audio_fft[~np.isin(np.arange(len(frequencies)), noise_freq_indices)] *= 2.5

# Atenuación más fuerte en las frecuencias de ruido
audio_fft[noise_freq_indices] *= 5.7

# Aplicar la transformada inversa de Fourier
filtered_audio = np.real(np.fft.ifft(audio_fft))

# Guardar audio filtrado
print("Guardando audio filtrado...")
with wave.open('filtered_audio.wav', 'w') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)  # 16 bits
    wf.setframerate(sample_rate)
    wf.writeframes((filtered_audio * 32767).astype(np.int16).tobytes())
print("Guardado de audio filtrado completado.")

# Crear espectrogramas
_, _, Sxx_original = spectrogram(audio_input.flatten(), fs=sample_rate)
_, _, Sxx_filtrado = spectrogram(filtered_audio, fs=sample_rate)

# Mostrar espectrogramas
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.title('Espectrograma del Audio Original')
plt.specgram(audio_input.flatten(), Fs=sample_rate, cmap='viridis')
plt.xlabel('Tiempo (s)')
plt.ylabel('Frecuencia (Hz)')

plt.subplot(3, 1, 2)
plt.title('Espectrograma del Audio Filtrado')
plt.specgram(filtered_audio, Fs=sample_rate, cmap='viridis')
plt.xlabel('Tiempo (s)')
plt.ylabel('Frecuencia (Hz)')

plt.subplot(3, 1, 3)
plt.title('Diferencia de Espectrogramas')
plt.imshow(np.log(Sxx_original) - np.log(Sxx_filtrado), cmap='viridis', aspect='auto', origin='lower')
plt.colorbar(label='Log(Espectrograma Original) - Log(Espectrograma Filtrado)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Frecuencia (Hz)')

plt.tight_layout()
plt.show()