import numpy as np
import matplotlib.pyplot as plt

fs = 1000          # frequência de amostragem (Hz)
T = 1              # duração do sinal (s)
t = np.arange(0, T, 1/fs)

# Sinais
x = 3 * np.cos(2 * np.pi * 10 * t)
xr = x + 5

N = len(x)
freqs = np.fft.fftfreq(N, 1/fs)

# ============================================================
# (a) Sinal x(t) no tempo e na frequência
# ============================================================

# --- x(t) no tempo ---
plt.figure(figsize=(10,4))
plt.plot(t, x)
plt.title("Sinal x(t) no tempo")
plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude")
plt.grid()
plt.show()

# --- x(t) na frequência ---
X = np.fft.fft(x)
X_mag = np.abs(X) / N

plt.figure(figsize=(10,4))
plt.stem(freqs, X_mag)
plt.xlim(-50, 50)
plt.title("Espectro de x(t)")
plt.xlabel("Frequência (Hz)")
plt.ylabel("|X(f)|")
plt.grid()
plt.show()

# ========================= FIM DO ITEM (a) ===================


# ============================================================
# (b) Sinal xr(t) no tempo e na frequência
# ============================================================

# --- xr(t) no tempo ---
plt.figure(figsize=(10,4))
plt.plot(t, xr)
plt.title("Sinal xr(t) no tempo (com DC offset)")
plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude")
plt.grid()
plt.show()

# --- xr(t) na frequência ---
Xr = np.fft.fft(xr)
Xr_mag = np.abs(Xr) / N

plt.figure(figsize=(10,4))
plt.stem(freqs, Xr_mag)
plt.xlim(-50, 50)
plt.title("Espectro de xr(t)")
plt.xlabel("Frequência (Hz)")
plt.ylabel("|Xr(f)|")
plt.grid()
plt.show()

# ========================= FIM DO ITEM (b) ===================


# ============================================================
# (c) Remoção do DC offset e comparação
# ============================================================

# --- Remoção do DC pela subtração da média ---
media_xr = np.sum(xr) / len(xr)
xr_sem_dc = xr - media_xr

# --- Comparação no tempo ---
plt.figure(figsize=(10,4))
plt.plot(t, x, label="x(t)")
plt.plot(t, xr_sem_dc, '--', label="xr(t) sem DC")
plt.title("Comparação no tempo")
plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.show()

# --- Comparação na frequência ---
Xr_sem_dc = np.fft.fft(xr_sem_dc)
Xr_sem_dc_mag = np.abs(Xr_sem_dc) / N

plt.figure(figsize=(10,4))
plt.stem(freqs, X_mag, label="|X(f)|")
plt.stem(freqs, Xr_sem_dc_mag, linefmt='r--', markerfmt='ro',
         label="|Xr(f) sem DC|")
plt.xlim(-50, 50)
plt.title("Comparação no domínio da frequência")
plt.xlabel("Frequência (Hz)")
plt.ylabel("Magnitude")
plt.legend()
plt.grid()
plt.show()

# ============================================================
# Outras maneiras de remover o DC offset
# ============================================================

#  Metodo 1: Filtro passa-alta simples ##
alpha = 0.99
y = np.zeros_like(xr)

for n in range(1, len(xr)):
    y[n] = xr[n] - xr[n-1] + alpha * y[n-1]

plt.figure(figsize=(10,4))
plt.plot(t, xr, label="xr(t) com DC", alpha=0.6)
plt.plot(t, y, label="xr(t) após passa-altas")
plt.title("Remoção de DC — Filtro passa-altas")
plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.show()

Y = np.fft.fft(y)
Y_mag = np.abs(Y) / N

plt.figure(figsize=(10,4))
plt.stem(freqs, Y_mag)
plt.xlim(-50, 50)
plt.title("Espectro após filtro passa-altas")
plt.xlabel("Frequência (Hz)")
plt.ylabel("|Y(f)|")
plt.grid()
plt.show()

# --- Metodo 2: Subtração da média móvel ---
janela = 100
media_movel = np.convolve(xr, np.ones(janela)/janela, mode='same')
xr_filtro = xr - media_movel

plt.figure(figsize=(10,4))
plt.plot(t, xr, label="xr(t) com DC", alpha=0.6)
plt.plot(t, xr_filtro, label="xr(t) - média móvel")
plt.title("Remoção de DC — Média móvel")
plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.show()

Xf = np.fft.fft(xr_filtro)
Xf_mag = np.abs(Xf) / N

plt.figure(figsize=(10,4))
plt.stem(freqs, Xf_mag)
plt.xlim(-50, 50)
plt.title("Espectro após subtração da média móvel")
plt.xlabel("Frequência (Hz)")
plt.ylabel("|X(f)|")
plt.grid()
plt.show()


