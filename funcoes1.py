import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import symbols, Eq, solve, lambdify, Heaviside, diff, Poly

# Símbolos globais do SymPy (evita redefinições e melhora legibilidade)
n = symbols('n', integer=True)
z = symbols('z')
C1, C2 = symbols('C1 C2')


#################### SEÇÃO 1: FUNÇÕES BÁSICAS DE SINAIS ####################
def impulso(indices):
    """
    TEORIA DIDÁTICA:
    O impulso unitário δ[n] é o 'átomo' dos sinais discretos: δ[n] = 1 se n=0, 0 caso contrário.
    Propriedades chave:
    - δ[n] = ∑ δ[n-k] δ[k] (propriedade sifting: integra/amostra em k=0).
    - Transformada Z: Z{δ[n]} = 1.
    - Útil para: encontrar resposta ao impulso h[n] de sistemas LTI (saída para δ[n]).
    - Em convolução: x[n] * δ[n] = x[n] (identidade).

    ENTRADA: indices (lista de inteiros n, ex: [-2, -1, 0, 1, 2]).
    SAÍDA: Lista com valores de δ[n] para cada n.
    """
    return [1 if idx == 0 else 0 for idx in indices]


def degrau(indices):
    """
    TEORIA DIDÁTICA:
    O degrau unitário u[n] modela uma 'liga' em n=0: u[n] = 1 se n ≥ 0, 0 caso contrário.
    É a integral discreta do impulso: u[n] = ∑_{k=-∞}^n δ[k].
    Transformada Z: Z{u[n]} = z / (z - 1), para |z| > 1 (ROC fora do polo em z=1).
    Aplicações: Teste de ganho DC (H(0)) em filtros; steady-state para entradas constantes.

    ENTRADA/SAÍDA: Similar a impulso.
    """
    return [1 if idx >= 0 else 0 for idx in indices]


def rampa(indices):
    """
    TEORIA DIDÁTICA:
    A rampa unitária r[n] = n * u[n]: crescimento linear após n=0.
    É a integral do degrau: r[n] = ∑_{k=0}^n u[k] = (n+1) u[n].
    Transformada Z: Z{r[n]} = z / (z - 1)^2.
    Útil para: rampas em controle (aceleração); convolução com integradores dá quadrática.

    ENTRADA/SAÍDA: Similar.
    """
    return [idx if idx >= 0 else 0 for idx in indices]


def atraso_avanco(sinal, nd):
    """
    TEORIA DIDÁTICA:
    Deslocamento temporal: y[n] = x[n - nd].
    - nd > 0: Atraso (adição de zeros à esquerda).
    - nd < 0: Avanço (adição de zeros à direita, assumindo causalidade).
    Em sistemas LTI: Invariância temporal → H(z) multiplica por z^{-nd}.
    Preserva energia, mas altera fase linearmente.

    ENTRADA: sinal (lista), nd (int).
    SAÍDA: Sinal deslocado (mesmo comprimento, zeros se necessário).
    """
    len_s = len(sinal)
    if nd > 0:  # Atraso
        return [0] * min(nd, len_s) + sinal[:len_s - nd]
    elif nd < 0:  # Avanço
        nd_abs = -nd
        return sinal[nd_abs:] + [0] * min(nd_abs, len_s)
    return sinal[:]


def reflexao_temporal(sinal):
    """
    TEORIA DIDÁTICA:
    Reflexão: y[n] = x[-n], inverte a sequência no tempo.
    Relaciona-se à simetria: útil em filtros FIR de fase linear (h[n] = h[N-1-n]).
    Transformada Z: Z{y[n]} = X(1/z).
    Aplicação: Análise de palíndromos em design de filtros.

    ENTRADA/SAÍDA: Lista invertida.
    """
    return sinal[::-1]


def escalonamento_amplitude(sinal, escalar):
    """
    TEORIA DIDÁTICA:
    Escalonamento linear: y[n] = A * x[n], propriedade de homogeneidade em LTI.
    Afeta ganho: |H(ω)| multiplica por |A|; preserva forma e fase.
    Útil para normalização ou amplificação.

    ENTRADA: sinal (lista), escalar (float).
    SAÍDA: Lista escalonada.
    """
    return [val * escalar for val in sinal]


#################### SEÇÃO 2: CONVOLUÇÃO DISCRETA ####################
def conv(x, h):
    """
    TEORIA DIDÁTICA:
    Convolução linear: y[n] = ∑_{k=-∞}^∞ x[k] * h[n - k].
    Para finitos e causais (x[k]=0, h[m]=0 para k<0,m<0): y[n] = ∑_{k=0}^n x[k] h[n-k].
    Propriedades: Comutativa, associativa, distributiva; no Z: Y(z) = X(z) H(z).
    Interpretação: 'Sobreposição' de entrada x com resposta h deslocada.
    Complexidade: O(N^2) aqui (direta); use FFT para N grande.

    ENTRADA: x, h (listas causais).
    SAÍDA: y (lista de len(x) + len(h) - 1).
    """
    N, M = len(x), len(h)
    y = [0.0] * (N + M - 1)
    for n in range(N + M - 1):
        for k in range(max(0, n - M + 1), min(n + 1, N)):
            y[n] += x[k] * h[n - k]
    return [float(val) for val in y]


#################### SEÇÃO 3: SOLUÇÃO DE EQUAÇÕES DE DIFERENÇA ####################
def eq_diferenca(coef_a, cond_y, indices):
    """
    TEORIA DIDÁTICA:
    Equação de Diferença Linear Constante (LCCDE): y[n] + ∑ a_k y[n-k] = ∑ b_m x[n-m].
    Solução homogênea (x=0): y0[n] = ∑ C_j * r_j^n, onde r_j raízes da eq característica
    r^ord + ∑ a_k r^{ord-k} = 0.
    - Raízes reais distintas: C1 r1^n + C2 r2^n.
    - Repetida: (C1 + C2 n) r^n.
    - Complexa: Forma polar (amplitude decaimento + oscilação).
    Constantes C via condições iniciais y[i] = yi.
    Estabilidade BIBO: Todas |r_j| < 1 (polos dentro círculo unitário).
    Plot: y0[n] de n=-10 a 10 para ver transientes.

    ENTRADA: coef_a (lista [a1, a2] para ordem 2; nota: eq é y[n] + a1 y[n-1] + a2 y[n-2]=0),
             cond_y (lista [y_i1, y_i2]), indices (lista [i1, i2]).
    SAÍDA: String com raízes, C's, estabilidade e expr y[n]; plota stem plot.
    Suporte: Ordem 1/2; raízes reais/repetidas/complexas.
    """
    tam = len(coef_a)
    delta = None
    yn_eq = None
    gamas = []
    C_vals = []
    n_vals = np.arange(-10, 11)

    if tam == 2:
        i1, i2 = indices
        y1, y2 = cond_y
        a1, a2 = coef_a
        a1_s, a2_s = sp.S(a1), sp.S(a2)
        delta = a1_s ** 2 - 4 * a2_s
        if delta > 0:  # Reais distintas
            g1 = float((-a1_s + sp.sqrt(delta)) / 2)
            g2 = float((-a1_s - sp.sqrt(delta)) / 2)
            yn = C1 * g1 ** n + C2 * g2 ** n
            eq1 = Eq(C1 * g1 ** i1 + C2 * g2 ** i1, y1)
            eq2 = Eq(C1 * g1 ** i2 + C2 * g2 ** i2, y2)
            sol = solve((eq1, eq2), (C1, C2))
            yn_eq = yn.subs(sol)
            gamas = [g1, g2]
            C_vals = [float(sol[C1]), float(sol[C2])]
        elif delta == 0:  # Repetida
            g = float(-a1_s / 2)
            yn = (C1 + C2 * n) * g ** n
            eq1 = Eq((C1 + C2 * i1) * g ** i1, y1)
            eq2 = Eq((C1 + C2 * i2) * g ** i2, y2)
            sol = solve((eq1, eq2), (C1, C2))
            yn_eq = yn.subs(sol)
            gamas = [g]
            C_vals = [float(sol[C1]), float(sol[C2])]
        else:  # Complexas (conjugadas)
            g1_s = (-a1_s + sp.sqrt(delta)) / 2
            g2_s = (-a1_s - sp.sqrt(delta)) / 2
            yn = C1 * g1_s ** n + C2 * g2_s ** n
            eq1 = Eq(C1 * g1_s ** i1 + C2 * g2_s ** i1, y1)
            eq2 = Eq(C1 * g1_s ** i2 + C2 * g2_s ** i2, y2)
            sol = solve((eq1, eq2), (C1, C2))
            yn_eq = yn.subs(sol)
            gamas = [complex(g1_s), complex(g2_s)]
            C_vals = [complex(sol[C1]), complex(sol[C2])]
    else:  # Ordem 1
        i1, y1 = indices[0], cond_y[0]
        g = float(-coef_a[0])
        yn = C1 * g ** n
        eq1 = Eq(C1 * g ** i1, y1)
        sol = solve(eq1, C1)
        yn_eq = yn.subs(C1, sol[0])
        gamas = [g]
        C_vals = [float(sol[0])]

    # Avaliação numérica e estabilidade
    y_func = lambdify(n, yn_eq, modules="numpy")
    y_vals_temp = y_func(n_vals)
    if np.isscalar(y_vals_temp):
        y_vals = np.full(len(n_vals), float(y_vals_temp))
    else:
        y_vals = np.real(np.array(y_vals_temp))  # Real para complexos
    estavel = all(abs(g) < 1 for g in gamas)
    est_str = "ESTÁVEL" if estavel else "INSTÁVEL"

    plt.figure(figsize=(8, 5))
    plt.stem(n_vals, y_vals, basefmt=" ")
    plt.title(f"Resposta Natural y₀[n]\n(Sistema {est_str} - Polos: {gamas})")
    plt.xlabel("n (tempo discreto)")
    plt.ylabel("y₀[n]")
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color='k', linewidth=0.5)
    plt.show()

    return (f"Raízes (gamas): {gamas}\n"
            f"Constantes (C): {C_vals}\n"
            f"Estabilidade: {est_str}\n"
            f"Expressão: y[n] = {yn_eq}")


#################### SEÇÃO 4: ANÁLISE EM DOMÍNIO Z E FREQUÊNCIA ####################

def H_em_omega(b, a, omega):
    """
    TEORIA DIDÁTICA:
    Função transferência em frequência: H(e^{jω}) = B(e^{jω}) / A(e^{jω}),
    onde B(z) = ∑ b_k z^{-k}, A(z) = 1 + ∑ a_k z^{-k} (normalizado).
    ω ∈ [0, π] (frequência normalizada, fs/2 = π).
    |H(ω)|: Ganho (linear ou dB); ∠H(ω): Atraso de fase (rad ou °).
    Para steady-state senoidal x[n]=cos(ω n): y[n] ≈ |H| cos(ω n + ∠H).

    ENTRADA: b, a (listas coefs num/den, low to high power ex [b0,b1] para b0 + b1 z^{-1}),
             omega (float ou array).
    SAÍDA: Complex H(e^{jω}).
    """
    if np.isscalar(omega):
        omega = np.array([omega])
    H_vals = []
    for w in omega:
        z = np.exp(1j * w)
        num = sum(b[k] * z ** (-k) for k in range(len(b)))
        den = sum(a[k] * z ** (-k) for k in range(len(a)))
        H_vals.append(num / den)
    return np.array(H_vals) if len(omega) > 1 else complex(H_vals[0])


def plot_resposta_frequencia(b, a, num_points=1024):
    """
    TEORIA DIDÁTICA:
    Diagrama de Bode discreto: Magnitude |H(e^{jω})| e fase ∠H(e^{jω}) vs ω.
    - Magnitude linear: Ganho absoluto.
    - Fase: Deslocamento (unwrap para contínua).
    Revela: Tipo de filtro (passa-baixa se |H(0)| alto, |H(π)| baixo).
    Para design: Especifique cutoff ω_c onde |H|=1/√2.

    ENTRADA: b, a (coefs); num_points (resolução plot).
    SAÍDA: w, mag, phase (arrays); plota subplots.
    """
    w = np.linspace(0, np.pi, num_points)
    H_w = H_em_omega(b, a, w)
    mag = np.abs(H_w)
    phase = np.unwrap(np.angle(H_w))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(w / np.pi, mag, 'b-', linewidth=2)
    ax1.set_title('Magnitude |H(e^{jω})|')
    ax1.set_xlabel('Frequência normalizada (ω / π)')
    ax1.set_ylabel('|H(ω)|')
    ax1.grid(True, alpha=0.3)

    ax2.plot(w / np.pi, np.degrees(phase), 'r-', linewidth=2)
    ax2.set_title('Fase ∠H(e^{jω})')
    ax2.set_xlabel('Frequência normalizada (ω / π)')
    ax2.set_ylabel('Fase (graus)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return w, mag, phase


def fracoes_parciais(b, a):
    """
    TEORIA DIDÁTICA:
    Decomposição: H(z) = ∑ R_k / (1 - p_k z^{-1}) + K (termo direto para FIR).
    Inversa Z: h[n] = ∑ R_k p_k^n u[n] (causal).
    R_k = [B(z) / A'(z)]_{z=p_k} (resíduo em polo p_k).
    Útil para: h[n] analítica (sem simulação); análise de modos (decaimento |p_k|<1).
    Polos: Raízes de A(z)=0; zeros de B(z)=0.

    ENTRADA: b, a (listas low to high).
    SAÍDA: R, P, K; plota polos no plano Z (círculo unitário).
    Nota: Usa SymPy para simbólico; approx numérico se falhar.
    """
    # Polinômios em potências positivas, coefs high to low (já são para a e b padded)
    # a = [a0=1, a1, a2] para A(z) = a0 z^2 + a1 z + a2
    # b = [b0, b1, b2] para B(z) = b0 z^2 + b1 z + b2 (pad se necessário)
    if len(b) < len(a):
        b = [0] * (len(a) - len(b)) + b  # Pad high degrees
    den_poly = Poly(a, z)
    num_poly = Poly(b, z)
    H = num_poly / den_poly

    try:
        # Apart simbólico
        H_apart = sp.apart(H, z)
        print(f"H(z) = {H_apart.simplify()}")
    except:
        print("Decomposição simbólica falhou; usando numérica.")
        H_apart = H

    # Polos numéricos
    polos = den_poly.nroots()
    polos = [complex(p) for p in polos]

    # Resíduos
    R = []
    den_diff = den_poly.diff(z)
    for p in polos:
        res_p = num_poly.eval(p) / den_diff.eval(p)
        R.append(complex(res_p))

    K = []  # Termo direto (se deg_b >= deg_a)
    if len(b) > len(a):
        K = [b[0] / a[0]]  # Leading coef ratio

    # PLOT: Polos no plano Z
    plt.figure(figsize=(8, 8))
    circle = plt.Circle((0, 0), 1, fill=False, color='black', linestyle='--', alpha=0.5)
    plt.gca().add_patch(circle)
    plt.scatter([np.real(p) for p in polos], [np.imag(p) for p in polos],
                marker='x', color='red', s=100, label='Polos')
    plt.axhline(0, color='k', linewidth=0.5)
    plt.axvline(0, color='k', linewidth=0.5)
    plt.xlabel('Parte Real')
    plt.ylabel('Parte Imaginária')
    plt.title('Diagrama de Polos no Plano Z\n(Círculo Unitário: |z|=1 para estabilidade)')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.legend()
    plt.show()

    print(f"Resíduos R: {R}")
    print(f"Polos P: {polos}")
    print(f"Termo direto K: {K}")

    return R, polos, K


def resposta_senoide(b, a, omega, n_samples=50):
    """
    TEORIA DIDÁTICA:
    Steady-state para entrada senoidal x[n] = cos(ω n + φ): y[n] = |H(ω)| cos(ω n + φ + ∠H(ω)).
    Ignora transientes (n grande); assume sistema estável.
    Aplicação: Análise de filtros em áudio (ganho/fase em freq específicas).

    ENTRADA: b, a (coefs); omega (rad/amostra); n_samples (plot).
    SAÍDA: n_vals, x, y, |H|, ∠H; plota stems de x e y.
    """
    H_w = H_em_omega(b, a, omega)
    mag_H = np.abs(H_w)
    phase_H = np.angle(H_w)

    print(f"H(e^{{j {omega:.3f}}}) = {H_w:.4f}")
    print(f"|H(ω)| = {mag_H:.4f}")
    print(f"∠H(ω) = {phase_H:.4f} rad ({np.degrees(phase_H):.2f}°)")

    n_vals = np.arange(n_samples)
    x = np.cos(omega * n_vals)  # φ=0
    y = mag_H * np.cos(omega * n_vals + phase_H)

    # PLOT
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
    ax1.stem(n_vals, x, basefmt=" ")
    ax1.set_title(f'Entrada: x[n] = cos({omega:.3f} n)')
    ax1.set_xlabel('n')
    ax1.set_ylabel('x[n]')
    ax1.grid(True)

    ax2.stem(n_vals, y, basefmt=" ")
    ax2.set_title(f'Saída Steady-State: y[n] = {mag_H:.4f} cos({omega:.3f} n + {phase_H:.4f} rad)')
    ax2.set_xlabel('n')
    ax2.set_ylabel('y[n]')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    return n_vals, x, y, mag_H, phase_H


def obter_h_n(b, a, n_max=20):
    """
    TEORIA DIDÁTICA:
    Resposta ao impulso h[n] = Z^{-1}{H(z)}, simulada recursivamente via LCCDE.
    Para IIR: h[n] = ∑ b_k δ[n-k] - ∑ a_k h[n-k] (zero-state).
    Útil quando inversa analítica é complexa; truncate em n_max para approx.

    ENTRADA: b, a (coefs); n_max.
    SAÍDA: h (array); plota stem de h[n].
    """
    # Normaliza a[0]=1
    b = np.array(b) / a[0]
    a_rest = np.array(a[1:]) / a[0]
    h = np.zeros(n_max)
    for nn in range(n_max):
        # Contribuição da entrada (impulso em 0)
        h_contrib_x = b[nn] if nn < len(b) else 0.0
        # Contribuição feedback: - sum a_rest[k] * h[nn - (k+1)]
        h_contrib_y = 0.0
        for k in range(len(a_rest)):
            if nn - (k + 1) >= 0:
                h_contrib_y -= a_rest[k] * h[nn - (k + 1)]
        h[nn] = h_contrib_x + h_contrib_y

    # PLOT
    plt.figure(figsize=(10, 4))
    plt.stem(range(n_max), h, basefmt=" ")
    plt.title('Resposta ao Impulso h[n]')
    plt.xlabel('n')
    plt.ylabel('h[n]')
    plt.grid(True)
    plt.show()

    return h


def resposta_estado_nulo(b, a, x, n_max=50):
    """
    TEORIA DIDÁTICA:
    Resposta zero-state y_zs[n] = x[n] * h[n] (convolução).
    Parte forçada da solução total; ignora ICs (estado nulo).
    Para degrau: Revela soma h[n] (ganho DC).

    ENTRADA: b, a; x (sinal entrada); n_max (truncar y).
    SAÍDA: y_zs; plota stem.
    """
    h = obter_h_n(b, a, len(x) + n_max)
    y_zs = conv(x, h)[:n_max]

    # PLOT
    plt.figure(figsize=(10, 4))
    plt.stem(range(len(y_zs)), y_zs, basefmt=" ")
    plt.title('Resposta Zero-State y_zs[n]')
    plt.xlabel('n')
    plt.ylabel('y_zs[n]')
    plt.grid(True)
    plt.show()

    return y_zs


# ============================================================
# Função 1 — Amostragem: x(t) → x[n]
# ============================================================
def amostragem_sinal(x_func, T, t_max, fs_fft=1000, plotar=True):
    """
    x_func : função x(t)
    T      : período de amostragem (s)
    t_max  : tempo máximo para simulação
    fs_fft : frequência de amostragem para discretização contínua
    plotar : se True, gera gráficos

    Retorna:
        n  : vetor de índices inteiros
        x_n: sinal discreto amostrado
    """
    # Sinal contínuo para visualização
    t = np.linspace(-t_max, t_max, int(2 * fs_fft * t_max))
    x_t = x_func(t)

    # Amostragem
    n = np.arange(-int(t_max / T), int(t_max / T) + 1)
    t_n = n * T
    x_n = x_func(t_n)

    # Trem de impulsos teórico
    delta_train = np.zeros_like(t)
    for tn in t_n:
        idx = np.argmin(np.abs(t - tn))
        delta_train[idx] = 1
    x_s_t = x_t * delta_train * (1 / T)

    # FFT contínua
    X_f = np.fft.fftshift(np.fft.fft(x_t))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(t), d=(t[1] - t[0])))

    # FFT discreta
    Xd = np.fft.fftshift(np.fft.fft(x_n, 1024))
    freqd = np.fft.fftshift(np.fft.fftfreq(len(Xd), d=T))

    if plotar:
        plt.figure(figsize=(12, 10))

        plt.subplot(2, 2, 1)
        plt.plot(t, x_t, label='x(t)')
        plt.stem(t_n, x_n, linefmt='r-', markerfmt='ro', basefmt=' ', label='Amostras')
        plt.title('Sinal contínuo e amostrado')
        plt.xlabel('t (s)')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.plot(freqs, np.abs(X_f) / max(np.abs(X_f)))
        plt.title('Espectro |X(f)|')
        plt.xlabel('Frequência (Hz)')
        plt.grid(True)

        plt.subplot(2, 2, 3)
        plt.stem(n, x_n, basefmt=' ')
        plt.title('Sinal discreto x[n]')
        plt.xlabel('n')
        plt.grid(True)

        plt.subplot(2, 2, 4)
        plt.plot(freqd, np.abs(Xd) / max(np.abs(Xd)))
        plt.title('Espectro |X_d(f)| (mostra réplicas)')
        plt.xlabel('Frequência (Hz)')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    return n, x_n


# ============================================================
# Função 2 — Reconstrução: x[n] → x(t)
# ============================================================
def reconstruir_sinal(x_n, T, t_max, plotar=True, x_func_original=None):
    """
    Reconstrói o sinal contínuo a partir das amostras x[n].

    x_n : vetor de amostras x[n]
    T   : período de amostragem (s)
    t_max : tempo máximo de reconstrução (s)
    plotar : se True, mostra gráfico comparando original e reconstruído
    x_func_original : função x(t) original (opcional, para comparar)

    Retorna:
        t : vetor de tempo contínuo
        x_t_recon : sinal reconstruído
    """
    n = np.arange(len(x_n)) - len(x_n) // 2
    t = np.linspace(-t_max, t_max, 2000)
    x_t_recon = np.zeros_like(t)

    for i, val in enumerate(x_n):
        x_t_recon += val * np.sinc((t - n[i] * T) / T)

    if plotar:
        plt.figure(figsize=(10, 5))
        if x_func_original:
            x_t_original = x_func_original(t)
            plt.plot(t, x_t_original, 'b', label='x(t) original', linewidth=2)
        plt.plot(t, x_t_recon, 'r--', label='x(t) reconstruído (a partir de x[n])')
        plt.title('Comparação entre sinal original e reconstruído')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        plt.show()

    return t, x_t_recon


if __name__ == "__main__":
    """print("=== EXEMPLO 1: SINAIS BÁSICOS ===")
    n_test = list(range(-3, 5))
    delta = impulso(n_test)
    u = degrau(n_test)
    r = rampa(n_test)
    atraso2 = atraso_avanco(delta, 2)
    avanco2 = atraso_avanco(delta, -2)
    reflex = reflexao_temporal(r)
    esc2 = escalonamento_amplitude(r, 2)

    print(f"n: {n_test}")
    print(f"δ[n]: {delta}")
    print(f"u[n]: {u}")
    print(f"r[n]: {r}")
    print(f"δ[n-2] (atraso): {atraso2}")
    print(f"δ[n+2] (avanço): {avanco2}")
    print(f"r[-n] (reflexão): {reflex}")
    print(f"2 * r[n]: {esc2}")

    print("\n=== EXEMPLO 2: CONVOLUÇÕES BÁSICAS ===")
    print(f"r[n] * u[n]: {conv(r, u)[:8]}")  # Rampa convoluída com degrau
    print(f"δ[n] * r[n]: {conv(delta, r)}")  # Deve ser r[n]
    print(f"δ[n] * u[n]: {conv(delta, u)}")  # u[n]

    print("\n=== EXEMPLO 3: CONVOLUÇÃO COM EXPONENCIAIS ===")
    n_exp = np.arange(0, 11)
    x_exp = (0.8 ** n_exp).tolist()
    h_exp = (0.3 ** n_exp).tolist()
    y_exp = conv(x_exp, h_exp)
    print(f"x[n] = 0.8^n u[n] (primeiros 5): {x_exp[:5]}")
    print(f"h[n] = 0.3^n u[n] (primeiros 5): {h_exp[:5]}")
    print(f"y[n] = x * h (primeiros 10): {[f'{v:.3f}' for v in y_exp[:10]]}")

    print("\n=== EXEMPLO 4: CONVOLUÇÃO COM SENOIDES ===")
    n_sen = np.arange(0, 20)

    # Ex 4.1: Senoides constantes/periódicas
    x_cos2pi = np.cos(2 * np.pi * n_sen).tolist()  # cos(2π n) = 1 (DC)
    h_cos3pi = (3 * np.cos(3 * np.pi * n_sen)).tolist()  # 3 cos(π n) = 3 (-1)^n
    y_cos1 = conv(x_cos2pi, h_cos3pi)
    print(f"x[n]=cos(2π n)=1: {x_cos2pi[:5]}")
    print(f"h[n]=3 cos(3π n): {h_cos3pi[:5]}")
    print(f"y[n] (primeiros 10): {[f'{v:.3f}' for v in y_cos1[:10]]}")

    # Plot para Ex 4.1
    fig, axs = plt.subplots(3, 1, figsize=(12, 10))
    axs[0].stem(n_sen, x_cos2pi, basefmt=" ")
    axs[0].set_title('x[n] = cos(2π n)')
    axs[0].grid(True)
    axs[1].stem(n_sen, h_cos3pi, basefmt=" ")
    axs[1].set_title('h[n] = 3 cos(3π n)')
    axs[1].grid(True)
    axs[2].stem(range(len(y_cos1)), y_cos1, basefmt=" ")
    axs[2].set_title('y[n] = x[n] * h[n]')
    axs[2].grid(True)
    plt.tight_layout()
    plt.show()

    # Ex 4.2: Senoides de baixa freq
    x_sen_pi4 = np.sin(np.pi * n_sen / 4).tolist()
    h_sen_pi2 = (2 * np.sin(np.pi * n_sen / 2)).tolist()  # 2 sin(π n /2) = alterna
    y_sen2 = conv(x_sen_pi4, h_sen_pi2)
    print(f"\nx[n]=sin(π n /4): {[f'{v:.3f}' for v in x_sen_pi4[:5]]}")
    print(f"h[n]=2 sin(π n /2): {[f'{v:.3f}' for v in h_sen_pi2[:5]]}")
    print(f"y[n] (primeiros 10): {[f'{v:.3f}' for v in y_sen2[:10]]}")

    # Plot similar omitido por brevidade; adicione se quiser.

    # Ex 4.3: Soma de senoides
    x_mista = (np.cos(np.pi * n_sen / 3) + 0.5 * np.sin(np.pi * n_sen / 6)).tolist()
    y_mista = conv(x_mista, h_exp)  # Com exponencial
    print(f"\nx[n]=cos(π n /3) + 0.5 sin(π n /6): {[f'{v:.3f}' for v in x_mista[:5]]}")
    print(f"y[n] (primeiros 10): {[f'{v:.3f}' for v in y_mista[:10]]}")

    print("\n=== EXEMPLO 5: EQUAÇÕES DE DIFERENÇA (Resposta Natural) ===")
    # Ex 5.1: Ordem 2, reais distintas (da Questão exemplo)
    res1 = eq_diferenca([-0.6, -0.16], [-25 / 4, 0], [-2, -1])
    print(f"Ex 5.1: {res1}")

    # Ex 5.2: Ordem 2, complexas (filtro Q3)
    res2 = eq_diferenca([-1.63, 0.70], [0, 0], [0, 1])  # ICs zero approx
    print(f"\nEx 5.2: {res2}")

    # Ex 5.3: Ordem 1
    res3 = eq_diferenca([-0.5], [2], [0])
    print(f"\nEx 5.3: {res3}")

    # Ex 5.4: Repetida (delta=0)
    res4 = eq_diferenca([2, 1], [1, 0], [0, 1])  # a1=2, a2=1 → g=-1 repetido
    print(f"\nEx 5.4 (repetida): {res4}")

    print("\n=== EXEMPLO 6: QUESTÃO 2 - JUROS COMPOSTOS (Série Geométrica) ===")
    r = 1.005  # Taxa mensal
    y0 = 300.0
    n_meses = 5
    # 2b: Apenas capital
    y2b = y0 * (r ** n_meses)
    # 2c: + Depósitos mensais de 100
    soma_dep = 100 * ((r ** n_meses - 1) / (r - 1))  # S = 100 * sum r^k, k=0 a 4
    y2c = y0 * (r ** n_meses) + soma_dep
    print(f"2b: y[5] (apenas capital) = R$ {y2b:.2f}")
    print(f"2c: y[5] (com depósitos) = R$ {y2c:.2f}")

    print("\n=== EXEMPLO 7: QUESTÃO 3 - FILTRO IIR ===")
    b_q3 = [0, 0.038, 0.034]  # Num: 0.038 z^{-1} + 0.034 z^{-2}
    a_q3 = [1, -1.63, 0.70]  # Den: 1 - 1.63 z^{-1} + 0.70 z^{-2}

    # 7.1: h[n]
    h_q3 = obter_h_n(b_q3, a_q3, 20)
    print(f"h[n] (primeiros 10): {[f'{v:.5f}' for v in h_q3[:10]]}")

    # 7.2: Resposta em frequência
    print("\nPlotando |H(ω)| e ∠H(ω)...")
    w, mag, phase = plot_resposta_frequencia(b_q3, a_q3)

    # 7.3: Em ω=π/4
    H_pi4 = H_em_omega(b_q3, a_q3, np.pi / 4)
    print(f"|H(e^j π/4)| = {np.abs(H_pi4):.4f}")
    print(f"∠H(e^j π/4) = {np.angle(H_pi4):.4f} rad ({np.degrees(np.angle(H_pi4)):.2f}°)")
    print(f"y[n] ≈ {np.abs(H_pi4):.4f} cos( (π/4) n + {np.degrees(np.angle(H_pi4)):.2f}° )")

    # 7.4: Frações parciais
    print("\nFrações parciais de H(z):")
    R, P, K = fracoes_parciais(b_q3, a_q3)

    # 7.5: Resposta a senoide em π/4
    print("\nResposta steady-state a cos(π n /4):")
    resposta_senoide(b_q3, a_q3, np.pi / 4, 30)

    # 7.6: Zero-state para degrau
    n_degrau = list(range(30))
    x_degrau = degrau(n_degrau)
    y_zs = resposta_estado_nulo(b_q3, a_q3, x_degrau, 30)
    print(f"y_zs[n] para degrau (primeiros 10): {[f'{v:.4f}' for v in y_zs[:10]]}")

    print("\n TODOS OS EXEMPLOS EXECUTADOS! Verifique plots para visualização.")"""

    # Sinal original
    f0 = 10  # Hz
    x_func = lambda t: np.cos(2 * np.pi * f0 * t)

    # Parâmetros de amostragem
    T = 0.03  # período de amostragem (fs ≈ 33 Hz)
    t_max = 0.5

    print("=== Etapa 1: Amostragem x(t) → x[n] ===")
    n, x_n = amostragem_sinal(x_func, T, t_max)

    print("=== Etapa 2: Reconstrução x[n] → x(t) ===")
    t, x_t_rec = reconstruir_sinal(x_n, T, t_max, x_func_original=x_func)
