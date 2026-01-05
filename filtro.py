import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import sympy as sp

num = [10000]
den = [1, 10000]

system = signal.TransferFunction(num, den)

w, mag, phase = signal.bode(system)

s = sp.Symbol('s')
num_poly = sum(coef * s**i for i, coef in enumerate(num[::-1]))
den_poly = sum(coef * s**i for i, coef in enumerate(den[::-1]))
G = num_poly / den_poly
print("G(s) =", G)

plt.figure(figsize=(8, 6))

plt.subplot(2, 1, 1)
plt.semilogx(w, mag)
plt.title("Diagrama de Bode - SciPy")
plt.ylabel("Magnitude (dB)")
plt.grid(True, which="both", ls="--", lw=0.5)

plt.subplot(2, 1, 2)
plt.semilogx(w, phase)
plt.xlabel("Frequência (rad/s)")
plt.ylabel("Fase (graus)")
plt.grid(True, which="both", ls="--", lw=0.5)

plt.tight_layout()
plt.show()