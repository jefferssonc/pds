import matplotlib.pyplot as plt
import sympy
from sympy import symbols, solve, Eq, lambdify
import numpy as np

#################### FUNÇÕES ###############################

def impulso(sinal):
    return [1 if s == 0 else 0 for s in sinal]

def rampa(sinal):
    return [s if s>=0 else 0 for s in sinal]

def degrau(sinal):
    return [1 if s>=0 else 0 for s in sinal]

def atraso_avanco(sinal, nd):
    if nd>0:
        y = [0]*nd+sinal
    else:
        y = sinal[-nd:]+[0]*(-nd)
    return y[:len(sinal)]

def reflexao_temporal(sinal):
    return sinal[::-1]

def escalonamento_amplitude(sinal, escalar):
    return [s * escalar for s in sinal]


def conv(x, h):
    N = len(x)
    M = len(h)
    y = [0]*(N+M-1)
    for n in range(len(y)):
        for k in range (N):
            if 0 <= n-k < M:
                y[n] += x[k] * h[n-k]

    #return y
    return [float(val) for val in y]


def eq_diferenca(coeficientes, condicoes, indices):
    #gamma = symbols('gamma')
    n = symbols('n', integer=True)
    C1, C2 = symbols('C1 C2')


    tam = len(coeficientes)

    if tam==2:
        i1, i2 = indices
        yi1, yi2 = condicoes
        a1, a2 = coeficientes
        #poli = gamma**2+yi1*gamma+yi2
        delta = a1**2-4*a2
        g1 = float((-a1 + sympy.sqrt(delta)) / 2)
        g2 = float((-a1 - sympy.sqrt(delta)) / 2)
        if delta>0:
            yn = C1 * g1**n + C2 * g2**n
            eq1 = Eq(C1 * g1 ** i1 + C2 * g2 ** i1, yi1)
            eq2 = Eq(C1 * g1 ** i2 + C2 * g2 ** i2, yi2)
            sol = solve((eq1, eq2), (C1, C2))
            yn_eq = yn.subs(sol)
        elif delta==0:
            #Gamas iguais, então tanto faz usar g1 ou g2
            yn = (C1+C2*n)*g1**n
            eq1 = Eq(C1 * g1**i1 + C2 * i1 * g1**i1, yi1)
            eq2 = Eq(C1 * g1 ** i2 + C2 * i2 * g1 ** i2, yi2)
            sol = solve((eq1, eq2), (C1, C2))
            yn_eq = yn.subs(sol)

        gamas = [float(g1), float(g2)]
        C_vals = [float(sol[C1]), float(sol[C2])]
    else:
        i1 = indices[0]
        yi1 = condicoes[0]
        g = float(-coeficientes[0])
        yn = C1 * g**n
        eq1=Eq(C1*g**i1, yi1)
        sol=solve(eq1,C1)
        yn_eq = yn.subs(C1,sol[0])

        gamas = [float(g)]
        C_vals = [float(sol[0])]


    #Caso complexo, implementar depois

    # Converter expressão simbólica para função numérica
    y_func = lambdify(n, yn_eq, modules="numpy")
    # Gera vetor de índices para plotar
    n_vals = np.arange(-10, 11)
    y_vals = y_func(n_vals)
    # Verifica estabilidade: todas raízes dentro do círculo unitário
    estabilidade = all(abs(g) < 1 for g in gamas)
    estabilidade_str = "Estável" if estabilidade else "Instável"
    # Plotagem
    plt.stem(n_vals, y_vals)
    plt.title("Resposta à Entrada Nula $y_0[n]$\nSistema " + estabilidade_str)
    plt.xlabel("n")
    plt.ylabel("y[n]")
    plt.grid(True)
    plt.show()
    # Retorna resultados
    return  (f"Gamas (raízes): {gamas}, "
             f"Constantes (C): {C_vals}, "
             f"Estabilidade: {estabilidade_str}, "
             f"Expressão y[n]: {yn_eq}")


"""n = [-3,-2,-1,0,1,2,3,4]
x1 = impulso(n)
x2 = rampa(n)
x3 = degrau(n)
x4 = reflexao_temporal(n)
x5 = escalonamento_amplitude(n, 2)
print(f"Impulso: {x1} ")
print(f"Rampa: {x2} ")
print(f"Degrau: {x3}")
print(f"Atraso/Avanço: {atraso_avanco(x1, -2)}")
print(f"Atraso/Avanço: {atraso_avanco(x1, 2)}")
print(f"Reflexão temporal: {x4}")
print(f"Escalonamento amplitude: {x5}")
print(f"Convolução de rampa + degrau: {conv(x2,x3)}")
print(f"Convolução de impulso + rampa: {conv(x1,x2)}")
print(f"Convolução de impulso + degrau: {conv(x1,x3)}")"""

n_tam = np.arange(0,11)

x1 = 0.8 ** n_tam
h1 = 0.3 ** n_tam
print(x1)
print(h1)

# print(eq_diferenca([-0.6,-0.16],[25,0],[-2,-1]))
# print(eq_diferenca([3,2],[2,1],[0,1]))
# print(eq_diferenca([-4,4],[1,4],[0,1]))
# print(eq_diferenca([-0.5], [2], [0]))
# print(eq_diferenca([-0.5], [2], [-1]))

print(f"Convolução de x1 e h1: {conv(x1,h1)}")
print(f"Convolução de x1 e h1: {conv([0,1,6,7,8,9,10,-11],[69,45,66,-2,3,0])}")
print(eq_diferenca([-0.6,-0.16],[(-25/4),0],[-2,-1]))




