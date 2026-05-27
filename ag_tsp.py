import random
import matplotlib.pyplot as plt
import numpy as np
import math


# Configurações
TAM_POP_INI = 50
NUM_ITE_MAX = 100
num_ite = 0
PENALIDADE = 200
melhores_fitness = []
historico_real = []

def carregar_vrptw(caminho_arquivo):
    """Lê o arquivo do VRPTW"""
    clientes = {}
    capacidade = 0 
    
    with open(caminho_arquivo, 'r') as f:
        linhas = f.readlines()
    
    for linha in linhas:
        partes = linha.split()

        if len(partes) == 2 and partes[0].isdigit() and partes[1].isdigit():
            capacidade = int(partes[1])

        elif len(partes) >= 7:
            try:
                valores = list(map(int, partes[:7]))
                id_no = valores[0]
                clientes[id_no] = {
                    'x': valores[1],
                    'y': valores[2],
                    'demanda': valores[3],
                    'ready_time': valores[4],
                    'due_date': valores[5],
                    'service_time': valores[6]
                }
            except ValueError:
                continue

    n = len(clientes)
    matriz_adj = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = clientes[i]['x'] - clientes[j]['x']
                dy = clientes[i]['y'] - clientes[j]['y']
                matriz_adj[i][j] = math.sqrt(dx**2 + dy**2)
                
    return clientes, matriz_adj, capacidade, n

def calcula_dist(x, y, matriz):
    return matriz[x][y]

def f_obj(valores):
    return np.sum(valores)

def fitness(solucao, matriz):
    
    distancias = []
    for i in range(len(solucao)):
        if i == (len(solucao) - 1):
            distancia = calcula_dist(solucao[i], solucao[0], matriz)
        else:
            distancia = calcula_dist(solucao[i], solucao[i+1], matriz)
        distancias.append(distancia)
    
    return f_obj(distancias)

def OX(pai1, pai2):
    tam = len(pai1)
    pos1, pos2 = sorted(random.sample(range(tam), 2))

    filho_1 = [None] * tam 
    filho_2 = [None] * tam 
    
    filho_1[pos1: pos2 +1] = pai2[pos1: pos2 +1]
    filho_2[pos1: pos2 +1] = pai1[pos1: pos2 +1]

    pos_atual = (pos2 + 1) % tam
    pos_pai = (pos2 + 1) % tam
    
    while None in filho_1:
        if pai1[pos_pai] not in filho_1:
            filho_1[pos_atual] = pai1[pos_pai]
            pos_atual = (pos_atual + 1) % tam
        pos_pai = (pos_pai + 1) % tam

    pos_atual = (pos2 + 1) % tam
    pos_pai = (pos2 + 1) % tam
    
    while None in filho_2:
        if pai2[pos_pai] not in filho_2:
            filho_2[pos_atual] = pai2[pos_pai]
            pos_atual = (pos_atual + 1) % tam
        pos_pai = (pos_pai + 1) % tam

    return filho_1, filho_2
    

def mutacao(individuo, taxa=0.1):
    r = random.uniform(0, 1)
    tam = len(individuo)
    if r <= taxa:
        pos1, pos2 = sorted(random.sample(range(tam), 2))
        individuo[pos1], individuo[pos2] = individuo[pos2], individuo[pos1]
    return individuo

def selecao_torneio(populacao, matriz, k=3):
    competidores = random.sample(populacao, k)
    melhor = min(competidores, key=lambda ind: fitness(ind, matriz))
    return melhor
    

def solucoes_iniciais(n):
    solucoes = []
    for _ in range(TAM_POP_INI):
        solucao = [x for x in range(n)]
        random.shuffle(solucao)
        solucoes.append(solucao)
    return solucoes

def carregar_grafo(caminho_arquivo):
    with open(caminho_arquivo, 'r') as f:
        # Filtra linhas que não são comentários e junta tudo em uma string
        conteudo = " ".join([linha for linha in f if not linha.startswith('#')])
        
    valores = list(map(int, conteudo.split()))
    
    n = int(len(valores) ** 0.5)
    
    matriz_adj = []
    for i in range(0, len(valores), n):
        matriz_adj.append(valores[i : i + n])
        
    return matriz_adj, n


matriz, num_cidades = carregar_grafo('lau15_dist.txt')

reprodutores = solucoes_iniciais(num_cidades)

# Loop Principal
populacao = solucoes_iniciais(num_cidades)

while num_ite < NUM_ITE_MAX:
    novos_filhos = []
    
    while len(novos_filhos) < TAM_POP_INI:
        pai1 = selecao_torneio(populacao, matriz)
        pai2 = selecao_torneio(populacao, matriz)
        
        f1, f2 = OX(pai1, pai2)
        
        f1 = mutacao(f1, taxa=0.1)
        f2 = mutacao(f2, taxa=0.1)
        
        novos_filhos.append(f1)
        if len(novos_filhos) < TAM_POP_INI:
            novos_filhos.append(f2)
    
    populacao = novos_filhos
    
    fits_geracao = [fitness(ind, matriz) for ind in populacao]
    melhor_da_geracao = min(fits_geracao)
    idx_melhor = fits_geracao.index(melhor_da_geracao)
    
    melhores_fitness.append(melhor_da_geracao)
    historico_real.append(melhor_da_geracao)
    
    if num_ite % 10 == 0:
        print(f"Geração {num_ite}: Menor Distância = {melhor_da_geracao}")
    
    num_ite += 1

# --- Gráfico ---
plt.figure(figsize=(10, 6))
plt.plot(melhores_fitness, label='Menor Distância (Fitness)')
plt.title('Convergência: Problema do Caixeiro Viajante')
plt.xlabel('Geração')
plt.ylabel('Distância Total')
plt.grid(True)
plt.legend()
plt.show()