import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Eq, solve, lambdify, sqrt, Heaviside, apart, diff, simplify
import warnings
warnings.filterwarnings("ignore", message=".*invalid value.*")

# Símbolos globais do SymPy
n_sym = symbols('n', integer=True)
z_sym = symbols('z')

#################### 1. FUNÇÕES BÁSICAS DE SINAIS ###############################

def impulso(indices):
    """
    TEORIA: Impulso unitário δ[n] = 1 se n = 0, 0 caso contrário.
    É o "átomo" de qualquer sinal discreto: x[n] = Σ x[k] δ[n−k].
    Transformada Z: Z{δ[n]} = 1.
    """
    return [1 if n == 0 else 0 for n in indices]

def degrau(indices):
    """
    TEORIA: Degrau unitário u[n] = 1 para n ≥ 0, 0 caso contrário.
    Representa ligação instantânea de um sinal constante.
    Transformada Z: Z{u[n]} = z/(z−1), |z| > 1.
    """
    return [1 if n >= 0 else 0 for n in indices]

def rampa(indices):
    """
    TEORIA: Rampa unitária r[n] = n·u[n].
    Modela crescimento linear. Útil para testar integradores.
    Transformada Z: Z{r[n]} = z/(z−1)², |z| > 1.
    """
    return [n if n >= 0 else 0 for n in indices]

def atraso_avanco(sinal, nd):
    """
    TEORIA: Deslocamento no tempo: y[n] = x[n − nd].
    - nd > 0: atraso (delay)
    - nd < 0: avanço (advance)
    Em sistemas LTI, multiplica H(z) por z^{−nd}.
    """
    if nd > 0:
        return [0] * nd + sinal[:-nd] if nd < len(sinal) else [0] * nd
    elif nd < 0:
        nd = -nd
        return sinal[nd:] + [0] * min(nd, len(sinal))
    else:
        return sinal[:]

def reflexao_temporal(sinal):
    """
    TEORIA: Reflexão no tempo: y[n] = x[−n].
    Útil para análise de simetria e filtros de fase linear.
    Transformada Z: Z{x[−n]} = X(1/z).
    """
    return sinal[::-1]

def escalonamento_amplitude(sinal, escalar):
    """
    TEORIA: y[n] = A·x[n]. Propriedade de homogeneidade em sistemas lineares.
    Afeta apenas a amplitude, não a forma do sinal.
    """
    return [escalar * val for val in sinal]

#################### 2. CONVOLUÇÃO DISCRETA ###############################

def conv(x, h):
    """
    TEORIA: Convolução discreta y[n] = Σ_{k} x[k]·h[n−k].
    É a saída de um sistema LTI com resposta ao impulso h[n] e entrada x[n].
    No domínio Z: Y(z) = X(z)·H(z).
    Complexidade: O(N·M). Aqui implementada de forma direta.
    """
    N, M = len(x), len(h)
    y = [0.0] * (N + M - 1)
    for n in range(len(y)):
        for k in range(N):
            if 0 <= n - k < M:
                y[n] += x[k] * h[n - k]
    return y

#################### 3. EQUAÇÃO DE DIFERENÇA (RESPOSTA NATURAL) ###############################

def eq_diferenca(coef_a, y_cond, n_cond):
    """
    TEORIA: Resolve a equação homogênea (entrada nula):
        y[n] + a₁·y[n−1] + a₂·y[n−2] + ... = 0
    com condições iniciais dadas.

    - Solução geral: y₀[n] = Σ Cⱼ·γⱼⁿ (γⱼ = raízes características)
    - Estabilidade: |γⱼ| < 1 para todo j (polos dentro do círculo unitário)

    Parâmetros:
        coef_a: lista [a₁, a₂, ...] (coeficientes após y[n])
        y_cond: valores y[n] nas posições n_cond
        n_cond: índices das condições iniciais (ex: [-2, -1] ou [0, 1])

    Retorna: string com raízes, constantes, estabilidade e expressão simbólica.
    Plota y₀[n] para n ∈ [−10, 10].
    """
    from sympy import I
    C1, C2 = symbols('C1 C2')
    ordem = len(coef_a)

    if ordem == 2:
        i1, i2 = n_cond
        y1, y2 = y_cond
        a1, a2 = coef_a

        # Equação característica: γ² + a1·γ + a2 = 0
        delta = a1**2 - 4*a2
        if delta >= 0:
            g1 = float((-a1 + np.sqrt(delta)) / 2)
            g2 = float((-a1 - np.sqrt(delta)) / 2)
            gamas = [g1, g2]
            if delta > 0:
                yn = C1 * g1**n_sym + C2 * g2**n_sym
                eq1 = Eq(C1 * g1**i1 + C2 * g2**i1, y1)
                eq2 = Eq(C1 * g1**i2 + C2 * g2**i2, y2)
            else:  # delta == 0
                yn = (C1 + C2 * n_sym) * g1**n_sym
                eq1 = Eq((C1 + C2 * i1) * g1**i1, y1)
                eq2 = Eq((C1 + C2 * i2) * g1**i2, y2)
        else:
            # Raízes complexas conjugadas
            real_part = -a1 / 2
            imag_part = np.sqrt(-delta) / 2
            g1 = complex(real_part, imag_part)
            g2 = complex(real_part, -imag_part)
            gamas = [g1, g2]
            yn = C1 * g1**n_sym + C2 * g2**n_sym
            eq1 = Eq(C1 * g1**i1 + C2 * g2**i1, y1)
            eq2 = Eq(C1 * g1**i2 + C2 * g2**i2, y2)

        sol = solve((eq1, eq2), (C1, C2))
        if not sol:
            raise ValueError("Sistema de equações sem solução única.")
        yn_eq = yn.subs(sol)
        C_vals = [complex(sol[C1]), complex(sol[C2])]

    elif ordem == 1:
        i1 = n_cond[0]
        y1 = y_cond[0]
        a1 = coef_a[0]
        g = float(-a1)
        gamas = [g]
        yn = C1 * g**n_sym
        eq1 = Eq(C1 * g**i1, y1)
        sol = solve(eq1, C1)
        yn_eq = yn.subs(C1, sol[0])
        C_vals = [float(sol[0])]
    else:
        raise NotImplementedError("Somente ordens 1 e 2 suportadas.")

    # Avaliação numérica para plotagem
    y_func = lambdify(n_sym, yn_eq, modules=["numpy", {"I": 1j}])
    n_vals = np.arange(-10, 11)
    y_vals = np.real(y_func(n_vals))

    # Estabilidade
    estavel = all(abs(g) < 1 for g in gamas)
    estab_str = "Estável" if estavel else "Instável"

    # Plotagem
    plt.figure(figsize=(8, 4))
    plt.stem(n_vals, y_vals, basefmt=" ")
    plt.title(f"Resposta à Entrada Nula $y_0[n]$\nSistema {estab_str}")
    plt.xlabel("n")
    plt.ylabel("$y_0[n]$")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    return (f"Gamas (raízes): {gamas}, "
            f"Constantes (C): {C_vals}, "
            f"Estabilidade: {estab_str}, "
            f"Expressão: $y_0[n] = {yn_eq}$")

#################### 4. ANÁLISE NO DOMÍNIO DA FREQUÊNCIA ###############################

def H_em_omega(b, a, omega):
    """
    TEORIA: Avaliação da função de transferência na circunferência unitária:
        H(e^{jω}) = B(e^{jω}) / A(e^{jω})
    onde B(z) = Σ b_k z^{-k}, A(z) = 1 + Σ a_k z^{-k}
    """
    z = np.exp(1j * omega)
    num = sum(b[k] * z**(-k) for k in range(len(b)))
    den = 1.0 + sum(a[k] * z**(-k) for k in range(len(a)))
    return num / den

def plot_resposta_frequencia(b, a, num_points=512):
    """
    Plota magnitude e fase de H(e^{jω}) para ω ∈ [0, π].
    - Magnitude: ganho do sistema em cada frequência
    - Fase: atraso de fase introduzido
    """
    w = np.linspace(0, np.pi, num_points)
    H_vals = np.array([H_em_omega(b, a, wi) for wi in w])
    mag = np.abs(H_vals)
    phase = np.angle(H_vals)
    phase_unwrapped = np.unwrap(phase)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(w / np.pi, mag, 'b')
    plt.title("Magnitude de $H(e^{j\\omega})$")
    plt.xlabel("$\\omega / \\pi$")
    plt.ylabel("$|H(e^{j\\omega})|$")
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(1, 2, 2)
    plt.plot(w / np.pi, np.degrees(phase_unwrapped), 'r')
    plt.title("Fase de $H(e^{j\\omega})$")
    plt.xlabel("$\\omega / \\pi$")
    plt.ylabel("Fase (graus)")
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()
    return w, mag, phase_unwrapped

def resposta_senoide(b, a, omega, n_samples=30):
    """
    TEORIA: Para x[n] = cos(ω₀n), a saída em regime permanente é:
        y[n] = |H(e^{jω₀})| · cos(ω₀n + ∠H(e^{jω₀}))
    Válido para sistemas estáveis.
    """
    H_val = H_em_omega(b, a, omega)
    mag = np.abs(H_val)
    phase = np.angle(H_val)

    n_vals = np.arange(n_samples)
    x = np.cos(omega * n_vals)
    y = mag * np.cos(omega * n_vals + phase)

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.stem(n_vals, x, basefmt=" ")
    plt.title(f"Entrada: $x[n] = \\cos({omega:.3f} n)$")
    plt.ylabel("$x[n]$")
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(2, 1, 2)
    plt.stem(n_vals, y, basefmt=" ")
    plt.title(f"Saída: $y[n] = {mag:.4f} \\cdot \\cos({omega:.3f} n + {phase:.4f})$")
    plt.xlabel("n")
    plt.ylabel("$y[n]$")
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    print(f"\nResposta em ω = {omega:.3f} rad:")
    print(f"  |H| = {mag:.4f}")
    print(f"  ∠H = {phase:.4f} rad ({np.degrees(phase):.2f}°)")
    return n_vals, x, y, mag, phase

#################### 5. RESPOSTA AO IMPULSO (NUMÉRICA) ###############################

def obter_h_n(b, a, n_max=20):
    """
    TEORIA: h[n] é a saída do sistema quando x[n] = δ[n].
    Pode ser obtida por:
        - Convolução inversa
        - Recursão direta (usado aqui)
        - Transformada Z inversa
    """
    x = [1] + [0] * (n_max - 1)  # impulso
    h = [0.0] * n_max
    for n in range(n_max):
        acc = sum(b[k] * x[n - k] for k in range(len(b)) if n - k >= 0)
        acc -= sum(a[k] * h[n - k] for k in range(len(a)) if n - k > 0 and n - k < n_max)
        h[n] = acc
    return h

#################### 6. FUNÇÕES AUXILIARES ###############################

def soma_geometrica(r, n):
    """
    TEORIA: Soma finita S = Σ_{k=0}^{n-1} r^k = (1 − rⁿ)/(1 − r) se r ≠ 1.
    Aplicações: juros compostos, soma de exponenciais em LTI.
    """
    if abs(r - 1) < 1e-12:
        return float(n)
    return float((1 - r**n) / (1 - r))

#################### 7. EXEMPLOS COMPLETOS ###############################

if __name__ == "__main__":
    print(" Processamento Digital de Sinais – Toolkit Completo")
    print("=" * 60)

    # --- Sinais elementares ---
    n_test = [-2, -1, 0, 1, 2, 3]
    print("\n1. Sinais Elementares:")
    print(f"  n = {n_test}")
    print(f"  δ[n] = {impulso(n_test)}")
    print(f"  u[n] = {degrau(n_test)}")
    print(f"  r[n] = {rampa(n_test)}")

    # --- Convolução ---
    print("\n2. Convolução:")
    x = impulso(n_test)
    h = degrau(n_test)
    y = conv(x, h)
    print(f"  δ[n] * u[n] = {y}")

    # --- Equação de diferença ---
    print("\n3. Equação de Diferença (Resposta Natural):")
    try:
        resultado = eq_diferenca([-0.6, -0.16], [-25/4, 0], [-2, -1])
        print("  Resultado:", resultado)
    except Exception as e:
        print("  Erro:", e)

    # --- Sistema LTI: Questão 3 ---
    print("\n4. Análise de Sistema LTI (Questão 3):")
    b = [0, 0.038, 0.034]  # coeficientes de x[n], x[n-1], x[n-2]
    a = [1.63, -0.70]       # coeficientes de y[n-1], y[n-2] (note: y[n] = -a1 y[n-1] - a2 y[n-2] + ...)

    # Resposta ao impulso
    h_n = obter_h_n(b, a, n_max=15)
    plt.figure()
    plt.stem(range(len(h_n)), h_n, basefmt=" ")
    plt.title("Resposta ao Impulso $h[n]$")
    plt.xlabel("n")
    plt.ylabel("$h[n]$")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # Resposta em frequência
    print("  Plotando resposta em frequência...")
    plot_resposta_frequencia(b[1:], a)  # b[1:] pois b[0]=0

    # Resposta a senoide em ω = π/4
    resposta_senoide(b[1:], a, np.pi/4, n_samples=20)

    # --- Juros compostos (aplicação prática) ---
    print("\n5. Aplicação: Juros Compostos")
    r = 0.005  # 0.5% ao mês
    y0 = 300
    n_meses = 5
    y2b = y0 * (1 + r)**n_meses
    y2c = y0 * (1 + r)**n_meses + 100 * soma_geometrica(1 + r, n_meses)
    print(f"  Saldo após {n_meses} meses (sem depósitos): R$ {y2b:.2f}")
    print(f"  Saldo com depósitos mensais de R$100: R$ {y2c:.2f}")

    # --- Senoides e convolução ---
    print("\n6. Convolução com Senoides:")
    n_sen = np.arange(0, 10)
    x1 = np.cos(2 * np.pi * n_sen).tolist()      # = [1,1,1,...]
    h1 = (3 * np.cos(3 * np.pi * n_sen)).tolist() # = 3*(-1)^n
    y1 = conv(x1, h1)
    print(f"  x[n] = cos(2πn) → {x1[:5]}")
    print(f"  h[n] = 3·cos(3πn) → {h1[:5]}")
    print(f"  y[n] = x*h → {y1[:8]}")

    print("\n Todos os exemplos foram executados com sucesso!")