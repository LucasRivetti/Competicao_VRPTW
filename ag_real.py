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

def combinacao(pai1, pai2, alpha = 0.5):
    d = abs(pai1 - pai2)
    menor = min(pai1, pai2)
    maior = max(pai1, pai2)
    filho_1 = uniform(menor - alpha * d, maior + alpha * d)
    filho_2 = uniform(menor - alpha * d, maior + alpha * d)
    return filho_1, filho_2

def mutacao(individuo, bot_in, top_in, taxa=0.1):
    if random() < taxa:
        individuo = uniform(bot_in, top_in)
    return individuo


def roleta(vec_x1, vec_x2, func_obj):
    reprodutores_x1 = []
    reprodutores_x2 = []
    roleta = []
    soma = 0
    for i in func_obj:
        if i == 0:
            soma += 1/(abs(i) + 0.001)
        else:
            soma += 1/abs(i)

    for j in func_obj:
        if j == 0:
            x = (1/(abs(i) + 0.001))/soma
            roleta.append(x)
        else:
            x = (1/abs(j))/soma
            roleta.append(x)

    
    for _ in range(TAM_POP_INI):
        soma_ind = 0.0
        r = uniform(0, 1)
        for i in range(len(roleta)):
            soma_ind += roleta[i]
            if soma_ind >= r:
                reprodutores_x1.append(vec_x1[i])
                reprodutores_x2.append(vec_x2[i])
                break


    return reprodutores_x1, reprodutores_x2

# Inicialização
vec_x1 = [uniform(13, 100) for _ in range(TAM_POP_INI)]
vec_x2 = [uniform(0, 100) for _ in range(TAM_POP_INI)]
func_obj = [fitness(x1, x2) for x1, x2 in zip(vec_x1, vec_x2)]

# Loop Principal
while(num_ite < NUM_ITE_MAX):
    repro_x1, repro_x2 = roleta(vec_x1, vec_x2, func_obj)
    
    filhos_x1 = []
    filhos_x2 = []

    idx_elite = func_obj.index(min(func_obj))
    elite_x1 = vec_x1[idx_elite]
    elite_x2 = vec_x2[idx_elite]

    # Crossover para X1
    for j in range(0, len(repro_x1) - 1, 2):
        # Cruza X1 e X2 dos mesmos pais ao mesmo tempo
        f1_x1, f2_x1 = combinacao(repro_x1[j], repro_x1[j+1])
        f1_x2, f2_x2 = combinacao(repro_x2[j], repro_x2[j+1])
        
        # Aplica mutação e adiciona à nova população
        filhos_x1.append(mutacao(f1_x1, 13, 100))
        filhos_x1.append(mutacao(f2_x1, 13, 100))
        filhos_x2.append(mutacao(f1_x2, 0, 100))
        filhos_x2.append(mutacao(f2_x2, 0, 100))

    vec_x1 = [max(13, min(100, x)) for x in filhos_x1]
    vec_x2 = [max(0, min(100, x)) for x in filhos_x2]

    vec_x1[0] = elite_x1
    vec_x2[0] = elite_x2


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
