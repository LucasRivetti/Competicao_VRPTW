from random import *
from bitstring import BitArray
import matplotlib.pyplot as plt
import numpy as np
# Configurações

TAM_POP_INI = 50
NUM_ITE_MAX = 100
num_ite = 0
PENALIDADE = 2000
melhores_fitness = []
historico_real = []


def f_obj(x1, x2):
    return (x1 + 2*x2 -7)**2 + (2*x1 + x2 - 5)**2

def atualiza_pos(pai, pb, gb, w, v0, c1 = 2.05, c2 = 2.05):
    r1 = uniform(0.0, 1.0)
    r2 = uniform(0.0, 1.0)
    v1 = w*v0 + c1*r1*(pb - pai) + c2*r2*(gb - pai)
    nova_pos = pai + v1
    return nova_pos, v1


# Inicialização
pb_x1 = []
pb_x2 = []
vec_x1 = [uniform(-10, 10) for _ in range(TAM_POP_INI)]
vec_x2 = [uniform(-10, 10) for _ in range(TAM_POP_INI)]
pb_x1 = list(vec_x1)
pb_x2 = list(vec_x2)

vel_x1 = [uniform(-10, 10) for _ in range(TAM_POP_INI)]
vel_x2 = [uniform(-10, 10) for _ in range(TAM_POP_INI)]

func_obj = [f_obj(x1, x2) for x1, x2 in zip(vec_x1, vec_x2)]

passo = -1*(0.5/NUM_ITE_MAX)
w = np.arange(0.9, 0.4, passo)
print(w)
# w = [0.9] * NUM_ITE_MAX

idx_gb = func_obj.index(min(func_obj))
gb_x1 = vec_x1[idx_gb]
gb_x2 = vec_x2[idx_gb]

# Loop Principal
while(num_ite < NUM_ITE_MAX):
    
    for i in range(TAM_POP_INI):
        nova_pos_x1, nova_vel_x1 = atualiza_pos(vec_x1[i], pb_x1[i], gb_x1, w[num_ite], vel_x1[i] )
        nova_pos_x2, nova_vel_x2 = atualiza_pos(vec_x2[i], pb_x2[i], gb_x2, w[num_ite], vel_x2[i] )

        if f_obj(nova_pos_x1, nova_pos_x2) < f_obj(vec_x1[i], vec_x2[i]):
            pb_x1[i] = nova_pos_x1
            pb_x2[i] = nova_pos_x2
        vec_x1[i] = max(-10, min(10, nova_pos_x1))
        vec_x2[i] = max(-10, min(10, nova_pos_x2))
        vel_x1[i] = nova_vel_x1
        vel_x2[i] = nova_vel_x2


    func_obj = [f_obj(x1, x2) for x1, x2 in zip(vec_x1, vec_x2)]

    idx_gb = func_obj.index(min(func_obj))
    cand_gb_x1 = vec_x1[idx_gb]
    cand_gb_x2 = vec_x2[idx_gb]

    if f_obj(cand_gb_x1, cand_gb_x2) < f_obj(gb_x1, gb_x2):
        gb_x1 = cand_gb_x1
        gb_x2 = cand_gb_x2
        menor_global = f_obj(gb_x1, gb_x2)
        
    
        melhores_fitness.append(menor_global)
        historico_real.append(menor_global)
        print(f"Geração {num_ite}: Melhor = {menor_global}, x1 = {gb_x1}, x2 = {gb_x2}")
    
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