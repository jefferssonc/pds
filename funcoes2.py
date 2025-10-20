import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import symbols, Eq, solve, lambdify, diff, cos, sin, re, im, I
import scipy.signal as signal

# Configuração de plotagem
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

# Símbolos globais para análise simbólica
n_sym = symbols('n', integer=True)
z_sym = symbols('z')
C1, C2 = symbols('C1 C2')


##############################
# 1. SINAIS BÁSICOS DISCRETOS #
##############################

def impulso(indices):
    """TEORIA: Impulso Unitário δ[n] = 1 se n = 0, 0 caso contrário"""
    return [1 if n == 0 else 0 for n in indices]


def degrau(indices):
    """TEORIA: Degrau Unitário u[n] = 1 se n ≥ 0, 0 caso contrário"""
    return [1 if n >= 0 else 0 for n in indices]


def rampa(indices):
    """TEORIA: Rampa Unitária r[n] = n·u[n] = n se n ≥ 0, 0 caso contrário"""
    return [n if n >= 0 else 0 for n in indices]


def exponencial(indices, a):
    """TEORIA: Exponencial Discreta x[n] = a^n · u[n]"""
    return [a ** n if n >= 0 else 0 for n in indices]


def senoide_discreta(indices, freq, amplitude=1, fase=0):
    """TEORIA: Senoide Discreta x[n] = A·cos(ωn + φ)"""
    return [amplitude * np.cos(freq * n + fase) for n in indices]


################################
# 2. OPERAÇÕES EM SINAIS #
################################

def deslocamento_temporal(sinal, k):
    """TEORIA: Deslocamento Temporal y[n] = x[n - k]"""
    if k > 0:
        return [0] * k + sinal[:-k] if k < len(sinal) else [0] * len(sinal)
    elif k < 0:
        k = abs(k)
        return sinal[k:] + [0] * min(k, len(sinal))
    else:
        return sinal.copy()


def reflexao_temporal(sinal):
    """TEORIA: Reflexão Temporal y[n] = x[-n]"""
    return sinal[::-1]


def escalonamento_amplitude(sinal, alpha):
    """TEORIA: Escalonamento de Amplitude y[n] = α·x[n]"""
    return [alpha * x for x in sinal]


def soma_sinais(sinal1, sinal2):
    """TEORIA: Soma de Sinais y[n] = x₁[n] + x₂[n]"""
    return [x1 + x2 for x1, x2 in zip(sinal1, sinal2)]


##########################
# 3. CONVOLUÇÃO DISCRETA #
##########################

def convolucao_discreta(x, h):
    """
    TEORIA: Convolução Discreta y[n] = Σ x[k]·h[n-k]
    """
    N, M = len(x), len(h)
    y = [0.0] * (N + M - 1)

    for n in range(len(y)):
        soma = 0.0
        for k in range(max(0, n - M + 1), min(n + 1, N)):
            soma += x[k] * h[n - k]
        y[n] = soma

    return y


################################
# 4. EQUAÇÕES DE DIFERENÇA #
################################

def resolver_equacao_diferenca(coef_a, condicoes_iniciais, indices_condicoes):
    """
    TEORIA: Resolve equação de diferença homogênea
    Forma: y[n] + a₁y[n-1] + ... + aₖy[n-k] = 0
    """
    n = n_sym

    if len(coef_a) == 2:  # Segunda ordem
        a1, a2 = coef_a
        i1, i2 = indices_condicoes
        y1, y2 = condicoes_iniciais

        # Equação característica: r² + a₁r + a₂ = 0
        delta = a1 ** 2 - 4 * a2

        if delta > 0:  # Raízes reais distintas
            r1 = (-a1 + np.sqrt(delta)) / 2
            r2 = (-a1 - np.sqrt(delta)) / 2
            sol_geral = C1 * r1 ** n + C2 * r2 ** n

        elif delta == 0:  # Raízes reais iguais
            r1 = -a1 / 2
            sol_geral = (C1 + C2 * n) * r1 ** n

        else:  # Raízes complexas conjugadas - CORREÇÃO AQUI
            real_part = -a1 / 2
            imag_part = np.sqrt(-delta) / 2
            r = np.sqrt(real_part ** 2 + imag_part ** 2)
            theta = np.arctan2(imag_part, real_part)

            # Usar expressão simbólica correta para raízes complexas
            sol_geral = r ** n * (C1 * sp.cos(theta * n) + C2 * sp.sin(theta * n))

        # Resolver sistema para condições iniciais
        eq1 = Eq(sol_geral.subs(n, i1), y1)
        eq2 = Eq(sol_geral.subs(n, i2), y2)
        sol_constantes = solve((eq1, eq2), (C1, C2))
        sol_final = sol_geral.subs(sol_constantes)

        # Avaliar estabilidade
        raizes = [r1, r2] if delta >= 0 else [complex(real_part, imag_part), complex(real_part, -imag_part)]
        estavel = all(abs(r) < 1 for r in raizes)

    else:  # Primeira ordem
        a1 = coef_a[0]
        i1 = indices_condicoes[0]
        y1 = condicoes_iniciais[0]

        r1 = -a1
        sol_geral = C1 * r1 ** n
        eq1 = Eq(sol_geral.subs(n, i1), y1)
        sol_constantes = solve(eq1, C1)
        sol_final = sol_geral.subs(C1, sol_constantes[0])

        raizes = [r1]
        estavel = abs(r1) < 1

    # Converter para função numérica
    try:
        y_func = lambdify(n, sol_final, ['numpy', 'sympy'])
        n_vals = np.arange(-5, 16)
        y_vals = np.array([float(y_func(n_val)) for n_val in n_vals], dtype=float)
    except:
        # Fallback para avaliação simbólica
        n_vals = np.arange(-5, 16)
        y_vals = np.array([float(sol_final.subs(n, n_val)) for n_val in n_vals], dtype=float)

    # Plotagem
    plt.figure(figsize=(12, 4))

    # Resposta temporal
    plt.subplot(1, 2, 1)
    plt.stem(n_vals, y_vals, basefmt="C0-", linefmt="C0-", markerfmt="C0o")
    plt.title(f'Resposta Natural\nSistema {"Estável" if estavel else "Instável"}')
    plt.xlabel('n')
    plt.ylabel('y[n]')
    plt.grid(True, alpha=0.3)

    # Diagrama de polos
    plt.subplot(1, 2, 2)
    circle = plt.Circle((0, 0), 1, fill=False, color='red', linestyle='--', alpha=0.7)
    plt.gca().add_patch(circle)

    for raiz in raizes:
        if isinstance(raiz, complex):
            plt.plot(raiz.real, raiz.imag, 'rx', markersize=10, markeredgewidth=2)
        else:
            plt.plot(raiz, 0, 'rx', markersize=10, markeredgewidth=2)

    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.xlabel('Parte Real')
    plt.ylabel('Parte Imaginária')
    plt.title('Diagrama de Polos')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)

    plt.tight_layout()
    plt.show()

    return {
        'solucao': sol_final,
        'raizes': raizes,
        'estavel': estavel,
        'constantes': sol_constantes
    }


#####################################
# 5. ANÁLISE NO DOMÍNIO DA FREQUÊNCIA #
#####################################

def resposta_frequencia(b_coef, a_coef, omega=None):
    """TEORIA: Resposta em Frequência H(e^{jω}) = Σ bₖe^{-jωk} / Σ aₖe^{-jωk}"""
    if omega is None:
        omega = np.linspace(0, np.pi, 1000)

    H_mag = np.zeros_like(omega)
    H_phase = np.zeros_like(omega)

    for i, w in enumerate(omega):
        num = sum(b * np.exp(-1j * w * k) for k, b in enumerate(b_coef))
        den = sum(a * np.exp(-1j * w * k) for k, a in enumerate(a_coef))
        H = num / den

        H_mag[i] = np.abs(H)
        H_phase[i] = np.angle(H)

    H_phase = np.unwrap(H_phase)

    # Plotagem
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

    # Magnitude linear
    ax1.plot(omega / np.pi, H_mag)
    ax1.set_title('Resposta em Magnitude (Linear)')
    ax1.set_xlabel('Frequência (×π rad/amostra)')
    ax1.set_ylabel('|H(ω)|')
    ax1.grid(True, alpha=0.3)

    # Magnitude em dB
    ax2.plot(omega / np.pi, 20 * np.log10(H_mag + 1e-10))
    ax2.set_title('Resposta em Magnitude (dB)')
    ax2.set_xlabel('Frequência (×π rad/amostra)')
    ax2.set_ylabel('|H(ω)| (dB)')
    ax2.grid(True, alpha=0.3)

    # Fase
    ax3.plot(omega / np.pi, np.degrees(H_phase))
    ax3.set_title('Resposta em Fase')
    ax3.set_xlabel('Frequência (×π rad/amostra)')
    ax3.set_ylabel('∠H(ω) (graus)')
    ax3.grid(True, alpha=0.3)

    # Diagrama de polos e zeros
    zeros = np.roots(b_coef)
    polos = np.roots(a_coef)

    circle = plt.Circle((0, 0), 1, fill=False, color='red', linestyle='--', alpha=0.7)
    ax4.add_patch(circle)

    ax4.plot(np.real(zeros), np.imag(zeros), 'bo', markersize=8, label='Zeros')
    ax4.plot(np.real(polos), np.imag(polos), 'rx', markersize=8, markeredgewidth=2, label='Polos')

    ax4.axhline(0, color='black', linewidth=0.5)
    ax4.axvline(0, color='black', linewidth=0.5)
    ax4.set_xlabel('Parte Real')
    ax4.set_ylabel('Parte Imaginária')
    ax4.set_title('Diagrama de Polos e Zeros')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axis('equal')
    ax4.set_xlim(-1.5, 1.5)
    ax4.set_ylim(-1.5, 1.5)

    plt.tight_layout()
    plt.show()

    return {
        'frequencias': omega,
        'magnitude': H_mag,
        'fase': H_phase,
        'polos': polos,
        'zeros': zeros
    }


def resposta_ao_impulso(b_coef, a_coef, n_amostras=50):
    """TEORIA: Resposta ao Impulso h[n] = TZ⁻¹{H(z)}"""
    t, h_n = signal.dimpulse((b_coef, a_coef, 1), n=n_amostras)
    h_n = h_n[0].flatten()

    plt.figure(figsize=(10, 4))
    plt.stem(range(len(h_n)), h_n, basefmt="C0-", linefmt="C0-", markerfmt="C0o")
    plt.title('Resposta ao Impulso $h[n]$')
    plt.xlabel('n')
    plt.ylabel('h[n]')
    plt.grid(True, alpha=0.3)
    plt.show()

    return h_n


########################
# 6. TRANSFORMADA Z #
########################

def analise_transformada_z(b_coef, a_coef):
    """TEORIA: Análise da Transformada Z H(z) = B(z)/A(z)"""
    R, P, K = signal.residuez(b_coef, a_coef)

    print("=" * 50)
    print("ANÁLISE DA TRANSFORMADA Z")
    print("=" * 50)
    print(f"Polinômio do numerador B(z): {b_coef}")
    print(f"Polinômio do denominador A(z): {a_coef}")
    print("\nDECOMPOSIÇÃO EM FRAÇÕES PARCIAIS:")
    print(f"Resíduos (R): {R}")
    print(f"Polos (P): {P}")
    print(f"Termo direto (K): {K}")

    estavel = all(abs(p) < 1 for p in P)
    print(f"\nESTABILIDADE: {'ESTÁVEL' if estavel else 'INSTÁVEL'}")
    print(f"Todos os polos estão {'DENTRO' if estavel else 'FORA'} do círculo unitário")

    z = sp.symbols('z')
    H_z = sum(b * z ** (-k) for k, b in enumerate(b_coef)) / sum(a * z ** (-k) for k, a in enumerate(a_coef))
    print(f"\nEXPRESSÃO DE H(z):")
    print(f"H(z) = {H_z}")

    return {
        'residuos': R,
        'polos': P,
        'termo_direto': K,
        'estavel': estavel,
        'H_z': H_z
    }


################################
# 7. EXEMPLOS PRÁTICOS #
################################

def exemplos_sinais_basicos():
    """Exemplos demonstrativos dos sinais básicos"""
    print("=" * 60)
    print("EXEMPLOS DE SINAIS BÁSICOS")
    print("=" * 60)

    n = np.arange(-5, 11)

    # Gerar sinais básicos
    delta = impulso(n)
    u = degrau(n)
    r = rampa(n)
    exp = exponencial(n, 0.8)
    sen = senoide_discreta(n, np.pi / 4, amplitude=2)

    # Plotagem
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))

    sinais = [delta, u, r, exp, sen]
    titulos = ['Impulso Unitário δ[n]', 'Degrau Unitário u[n]', 'Rampa Unitária r[n]',
               'Exponencial (0.8)^n·u[n]', 'Senóide 2·cos(πn/4)']

    for i, (sinal, titulo) in enumerate(zip(sinais, titulos)):
        ax = axs[i // 3, i % 3]
        ax.stem(n, sinal, basefmt="C0-", linefmt="C0-", markerfmt="C0o")
        ax.set_title(titulo)
        ax.set_xlabel('n')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-5, 10)

    axs[1, 2].remove()
    plt.tight_layout()
    plt.show()

    return {'n': n, 'sinais': sinais, 'titulos': titulos}


def exemplos_convolucao():
    """Exemplos demonstrativos de convolução"""
    print("\n" + "=" * 60)
    print("EXEMPLOS DE CONVOLUÇÃO")
    print("=" * 60)

    # Exemplo 1: Convolução de sinais básicos
    x1 = degrau(list(range(5)))
    h1 = impulso(list(range(5)))
    y1 = convolucao_discreta(x1, h1)

    print("Exemplo 1: u[n] * δ[n] = u[n]")
    print(f"x[n] (degrau): {x1}")
    print(f"h[n] (impulso): {h1}")
    print(f"y[n] (convolução): {[round(val, 2) for val in y1]}")

    # Exemplo 2: Convolução com exponenciais
    x2 = exponencial(list(range(6)), 0.7)
    h2 = exponencial(list(range(6)), 0.5)
    y2 = convolucao_discreta(x2, h2)

    print("\nExemplo 2: (0.7)^n * (0.5)^n")
    print(f"x[n]: {[round(val, 3) for val in x2]}")
    print(f"h[n]: {[round(val, 3) for val in h2]}")
    print(f"y[n]: {[round(val, 3) for val in y2[:8]]}")

    # Exemplo 3: Convolução com senoides
    x3 = senoide_discreta(list(range(8)), np.pi / 6)
    h3 = senoide_discreta(list(range(8)), np.pi / 3)
    y3 = convolucao_discreta(x3, h3)

    print("\nExemplo 3: cos(πn/6) * cos(πn/3)")
    print(f"x[n]: {[round(val, 3) for val in x3]}")
    print(f"h[n]: {[round(val, 3) for val in h3]}")
    print(f"y[n]: {[round(val, 3) for val in y3[:10]]}")

    # Plotagem comparativa
    fig, axs = plt.subplots(3, 1, figsize=(12, 10))

    exemplos = [
        (x1, h1, y1, "u[n] * δ[n]"),
        (x2, h2, y2, "(0.7)^n * (0.5)^n"),
        (x3, h3, y3, "cos(πn/6) * cos(πn/3)")
    ]

    for i, (x, h, y, titulo) in enumerate(exemplos):
        ax = axs[i]
        n_x = range(len(x))
        n_h = range(len(h))
        n_y = range(len(y))

        ax.stem(n_x, x, linefmt='C0-', markerfmt='C0o', basefmt='C0-', label='x[n]')
        ax.stem([ni + 0.2 for ni in n_h], h, linefmt='C1-', markerfmt='C1s', basefmt='C1-', label='h[n]')
        ax.stem([ni + 0.4 for ni in n_y], y, linefmt='C2-', markerfmt='C2^', basefmt='C2-', label='y[n]')

        ax.set_title(f'Convolução: {titulo}')
        ax.set_xlabel('n')
        ax.set_ylabel('Amplitude')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return {'exemplos': exemplos}


def exemplo_sistema_completo():
    """Exemplo completo de análise de sistema LTI"""
    print("\n" + "=" * 60)
    print("ANÁLISE COMPLETA DE SISTEMA LTI")
    print("=" * 60)

    # Sistema: Filtro passa-baixa
    b = [0.1, 0.2, 0.1]
    a = [1, -1.2, 0.5]

    print("Sistema analisado:")
    print(f"H(z) = (0.1 + 0.2z⁻¹ + 0.1z⁻²) / (1 - 1.2z⁻¹ + 0.5z⁻²)")

    # 1. Resposta ao impulso
    print("\n1. RESPOSTA AO IMPULSO:")
    h_n = resposta_ao_impulso(b, a, 20)
    print(f"h[0] a h[4]: {[round(val, 4) for val in h_n[:5]]}")

    # 2. Análise da Transformada Z
    print("\n2. ANÁLISE DA TRANSFORMADA Z:")
    analise_z = analise_transformada_z(b, a)

    # 3. Resposta em frequência
    print("\n3. RESPOSTA EM FREQUÊNCIA:")
    analise_freq = resposta_frequencia(b, a)

    # 4. Resposta a uma senóide
    print("\n4. RESPOSTA A SENÓIDE:")
    n = np.arange(0, 30)
    x_sen = senoide_discreta(n, np.pi / 4)

    y_sen = convolucao_discreta(x_sen, h_n)
    y_sen = y_sen[:len(n)]

    H_omega = analise_freq['magnitude'][len(analise_freq['frequencias']) // 4]
    fase_omega = analise_freq['fase'][len(analise_freq['frequencias']) // 4]

    print(f"Entrada: x[n] = cos(πn/4)")
    print(f"|H(π/4)| = {H_omega:.4f}")
    print(f"∠H(π/4) = {np.degrees(fase_omega):.2f}°")
    print(f"Saída teórica: y[n] = {H_omega:.4f}·cos(πn/4 + {np.degrees(fase_omega):.2f}°)")

    # Plotagem da resposta à senóide
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.stem(n, x_sen, basefmt="C0-", linefmt="C0-", markerfmt="C0o", label='x[n] = cos(πn/4)')
    plt.title('Sinal de Entrada')
    plt.xlabel('n')
    plt.ylabel('x[n]')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.stem(n, y_sen, basefmt="C1-", linefmt="C1-", markerfmt="C1s", label='y[n] (saída)')

    y_teorico = H_omega * np.cos(np.pi / 4 * n + fase_omega)
    plt.plot(n, y_teorico, 'C2--', linewidth=2, label='Resposta teórica (regime)')

    plt.title('Resposta do Sistema')
    plt.xlabel('n')
    plt.ylabel('y[n]')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()

    return {
        'sistema': (b, a),
        'resposta_impulso': h_n,
        'analise_z': analise_z,
        'analise_freq': analise_freq,
        'resposta_senoide': (x_sen, y_sen, y_teorico)
    }


def exemplo_aplicacao_juros():
    """Exemplo aplicado: Sistema de juros compostos"""
    print("\n" + "=" * 60)
    print("APLICAÇÃO: SISTEMA DE JUROS COMPOSTOS")
    print("=" * 60)

    r = 0.005
    y0 = 300

    print(f"Taxa de juros: {r * 100:.1f}% ao mês")
    print(f"Saldo inicial: R$ {y0:.2f}")

    n_meses = 12
    n = np.arange(n_meses + 1)

    # Cenário (a): Apenas saldo inicial
    y_a = [y0 * (1 + r) ** k for k in n]

    # Cenário (b): Depósitos mensais de R$ 100
    deposito_mensal = 100
    y_b = [0] * len(n)
    y_b[0] = y0

    for k in range(1, len(n)):
        y_b[k] = (1 + r) * y_b[k - 1] + deposito_mensal

    # Plotagem
    plt.figure(figsize=(12, 6))

    plt.plot(n, y_a, 'C0-o', linewidth=2, markersize=6, label='Apenas saldo inicial')
    plt.plot(n, y_b, 'C1-s', linewidth=2, markersize=6, label=f'Saldo + depósito R$ {deposito_mensal}')

    plt.title('Evolução do Saldo - Sistema de Juros Compostos')
    plt.xlabel('Mês (n)')
    plt.ylabel('Saldo (R$)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    print("\nEvolução do Saldo:")
    print("Mês | Apenas Saldo | Saldo + Depósito")
    print("-" * 40)
    for k in [0, 3, 6, 9, 12]:
        print(f"{k:3d} | R$ {y_a[k]:8.2f} | R$ {y_b[k]:8.2f}")

    plt.tight_layout()
    plt.show()

    return {
        'taxa_juros': r,
        'saldo_inicial': y0,
        'deposito_mensal': deposito_mensal,
        'cenarios': {
            'apenas_saldo': y_a,
            'com_depositos': y_b
        }
    }


########################
# FUNÇÃO PRINCIPAL #
########################

def main():
    """FUNÇÃO PRINCIPAL - DEMONSTRAÇÃO COMPLETA DE PDS"""
    print("PROCESSAMENTO DIGITAL DE SINAIS - DEMONSTRAÇÃO COMPLETA")
    print("=" * 60)
    print("Autor: Sistema de Processamento Digital")
    print("Data: 2024")
    print("=" * 60)

    try:
        # 1. Sinais básicos
        exemplos_sinais_basicos()

        # 2. Operações em sinais
        print("\n" + "=" * 60)
        print("OPERADORES EM SINAIS")
        print("=" * 60)

        n = np.arange(-3, 7)
        x = [1, 2, 3, 4, 5, 4, 3, 2, 1, 0]

        print(f"Sinal original x[n]: {x}")
        print(f"Deslocamento x[n-2]: {deslocamento_temporal(x, 2)}")
        print(f"Reflexão x[-n]: {reflexao_temporal(x)}")
        print(f"Escalonamento 2·x[n]: {escalonamento_amplitude(x, 2)}")

        # 3. Convolução
        exemplos_convolucao()

        # 4. Equações de diferença - COM TRATAMENTO DE ERRO
        print("\n" + "=" * 60)
        print("EQUAÇÕES DE DIFERENÇA")
        print("=" * 60)

        print("Exemplo 1: Sistema de 1ª ordem")
        print("Equação: y[n] - 0.8y[n-1] = 0, y[0] = 1")
        try:
            sol1 = resolver_equacao_diferenca([-0.8], [1], [0])
        except Exception as e:
            print(f"Erro no exemplo 1: {e}")
            print("Continuando com próximos exemplos...")

        print("\nExemplo 2: Sistema de 2ª ordem (oscilatório)")
        print("Equação: y[n] - 1.5y[n-1] + 0.9y[n-2] = 0, y[0] = 1, y[1] = 1.5")
        try:
            sol2 = resolver_equacao_diferenca([-1.5, 0.9], [1, 1.5], [0, 1])
        except Exception as e:
            print(f"Erro no exemplo 2: {e}")
            print("Continuando com próximos exemplos...")

        # 5. Sistema LTI completo
        exemplo_sistema_completo()

        # 6. Aplicação prática
        exemplo_aplicacao_juros()



    except Exception as e:
        print(f"\nErro durante a execução: {e}")
        print("Alguns exemplos podem não ter sido executados completamente.")


# Executar a demonstração
if __name__ == "__main__":
    main()