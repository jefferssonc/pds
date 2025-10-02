import matplotlib.pyplot as plt

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


n = [-3,-2,-1,0,1,2,3,4]
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

