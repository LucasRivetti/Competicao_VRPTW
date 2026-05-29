from random import *
from bitstring import BitArray
import matplotlib.pyplot as plt

# Configurações
TAM_POP_INI = 50
NUM_ITE_MAX = 100
num_ite = 0
PENALIDADE = 2000
melhores_fitness = []
historico_real = []


def f_obj(x1, x2):
    return (x1 - 10)**3 + (x2 - 20)**3

def fitness(x1, x2):
    valor = f_obj(x1, x2)
    v = 0
    g1 = -((x1 - 5)**2) - ((x2 - 5)**2) + 100
    g2 = (x1 - 6)**2 + (x2 - 5)**2 - 82.81
    if g1 > 0: v += g1
    if g2 > 0: v += g2
    return valor + (PENALIDADE * v)

def combinacao(pai1, pai2):
    pc = len(pai1) // 2
    g1 = pai1[:pc]
    g2 = pai2[pc:]
    return g1 + g2

def mutacao(individuo, taxa=0.1):
    if random() < taxa:
        pos = randint(0, len(individuo) - 1)
        individuo.invert(pos)
    return individuo

def conversao_d_to_b(candidato):
    return BitArray(float=float(candidato), length=32)

def conversao_b_to_d(candidato):
    return candidato.float

def torneio(vec_x1, vec_x2, func_obj):
    reprodutores_x1 = []
    reprodutores_x2 = []
    for _ in range(TAM_POP_INI):
        # Sorteia dois indivíduos quaisquer da população
        a, b = randint(0, TAM_POP_INI-1), randint(0, TAM_POP_INI-1)
        # O melhor entre eles vai para a lista de reprodutores
        if func_obj[a] < func_obj[b]:
            reprodutores_x1.append(vec_x1[a])
            reprodutores_x2.append(vec_x2[a])
        else:
            reprodutores_x1.append(vec_x1[b])
            reprodutores_x2.append(vec_x2[b])
    return reprodutores_x1, reprodutores_x2

# --- Inicialização ---
vec_x1 = [uniform(13, 100) for _ in range(TAM_POP_INI)]
vec_x2 = [uniform(0, 100) for _ in range(TAM_POP_INI)]
func_obj = [fitness(x1, x2) for x1, x2 in zip(vec_x1, vec_x2)]

# --- Loop Principal ---
while(num_ite < NUM_ITE_MAX):
    repro_x1, repro_x2 = torneio(vec_x1, vec_x2, func_obj)
    
    filhos_x1 = []
    filhos_x2 = []

    # Crossover para X1
    for j in range(0, len(repro_x1) - 1, 2):
        p1, p2 = conversao_d_to_b(repro_x1[j]), conversao_d_to_b(repro_x1[j+1])
        filhos_x1.append(mutacao(combinacao(p1, p2)))
        filhos_x1.append(mutacao(combinacao(p2, p1)))

    # Crossover para X2
    for k in range(0, len(repro_x2) - 1, 2):
        p1, p2 = conversao_d_to_b(repro_x2[k]), conversao_d_to_b(repro_x2[k+1])
        filhos_x2.append(mutacao(combinacao(p1, p2)))
        filhos_x2.append(mutacao(combinacao(p2, p1)))

    vec_x1 = [max(13, min(100, conversao_b_to_d(x))) for x in filhos_x1]
    vec_x2 = [max(0, min(100, conversao_b_to_d(x))) for x in filhos_x2]


    # Recalcula fitness para a nova geração
    func_obj = [fitness(x1, x2) for x1, x2 in zip(vec_x1, vec_x2)]
    
    if func_obj:
        idx = func_obj.index(min(func_obj))
        melhores_fitness.append(min(func_obj))
        historico_real.append(f_obj(vec_x1[idx], vec_x2[idx]))
        print(f"Geração {num_ite}: Melhor = {min(func_obj)}, x1 = {vec_x1[idx]}, x2 = {vec_x2[idx]}")
    
    num_ite += 1

# Plota a convergência
plt.figure(figsize=(10, 6))
plt.plot(melhores_fitness, label='Fitness Total (c/ Penalidade)')
plt.plot(historico_real, label='Valor Real f(x)', linestyle='--')
plt.title('Convergência do Algoritmo Genético')
plt.xlabel('Geração')
plt.ylabel('Valor de Fitness')
plt.grid(True)
plt.legend()
plt.show()