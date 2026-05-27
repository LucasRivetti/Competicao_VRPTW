import random
from bitstring import BitArray
import matplotlib.pyplot as plt

# Configurações
TAM_POP_INI = 50
NUM_ITE_MAX = 100
num_ite = 0
melhores_fitness = []
historico_real = []
LIM_INF = -10
LIM_SUP = 10


def fitness(x1, x2):
    return (x1 + 2*x2 -7)**2 + (2*x1 + x2 - 5)**2


def combinacao(pai1, pai2, alpha = 0.5):
    d = abs(pai1 - pai2)
    menor = min(pai1, pai2)
    maior = max(pai1, pai2)
    filho_1 = random.uniform(menor - alpha * d, maior + alpha * d)
    return filho_1

def vetor_mutante(indice, populacao1, populacao2, F= 0.6):
    intervalo = list(range(len(populacao1)))
    intervalo.pop(indice)
    indices = random.sample(intervalo, 3)
    vm1 = populacao1[indices[0]] + F * (populacao1[indices[1]] - populacao1[indices[2]])
    vm2 = populacao2[indices[0]] + F * (populacao2[indices[1]] - populacao2[indices[2]])

    return vm1, vm2
    

def selecao(populacao1, populacao2):
    populacao1_final = []
    populacao2_final = []
    func_obj = []
    for i in range(len(populacao1)):
        vm1, vm2 = vetor_mutante(i, populacao1, populacao2)
        U1 = combinacao(populacao1[i], vm1)
        U2 = combinacao(populacao2[i], vm2)
        U1 = max(LIM_INF, min(LIM_SUP, U1))
        U2 = max(LIM_INF, min(LIM_SUP, U2))
        fitp = fitness(populacao1[i], populacao2[i])
        fitU = fitness(U1, U2)
        if fitp > fitU:
            populacao1_final.append(U1)
            populacao2_final.append(U2)
            func_obj.append(fitU)
        else:
            populacao1_final.append(populacao1[i])
            populacao2_final.append(populacao2[i])
            func_obj.append(fitp)

    return populacao1_final, populacao2_final, func_obj


    


# Inicialização
vec_x1 = [random.uniform(LIM_INF, LIM_SUP) for _ in range(TAM_POP_INI)]
vec_x2 = [random.uniform(LIM_INF, LIM_SUP) for _ in range(TAM_POP_INI)]
func_obj = [fitness(x1, x2) for x1, x2 in zip(vec_x1, vec_x2)]

# Loop Principal
while(num_ite < NUM_ITE_MAX):
    vec_x1, vec_x2, func_obj = selecao(vec_x1, vec_x2)

    
    if func_obj:
        idx = func_obj.index(min(func_obj))
        melhores_fitness.append(min(func_obj))
        print(f"Geração {num_ite}: Melhor = {min(func_obj)}, x1 = {vec_x1[idx]}, x2 = {vec_x2[idx]}")
    
    num_ite += 1

# Plota a convergência
plt.figure(figsize=(10, 6))
plt.plot(melhores_fitness, label='Fitness Total')
plt.title('Convergência do Algoritmo Genético')
plt.xlabel('Geração')
plt.ylabel('Valor de Fitness')
plt.grid(True)
plt.legend()
plt.show()
