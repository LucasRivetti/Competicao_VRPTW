import random
import numpy as np
import matplotlib.pyplot as plt

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

def retorna_dist(x, y, matriz):
    return matriz[x][y]

def retorna_fer(x, y, matriz):
    return matriz[x][y]


def f_obj(valores):
    return np.sum(valores)

def fitness(solucao, matriz):
    
    distancias = []
    for i in range(len(solucao)):
        if i == (len(solucao) - 1):
            distancia = retorna_dist(solucao[i], solucao[0], matriz)
        else:
            distancia = retorna_dist(solucao[i], solucao[i+1], matriz)
        distancias.append(distancia)
    
    return f_obj(distancias)


def calcula_p_trans(n, candidatos, vertice_atual, distancias, matriz_fer, alpha = 0.5, beta=0.5):
    probabilidades=[]
    divisor = 0
    for x in candidatos:
        divisor += (retorna_fer(vertice_atual, x, matriz_fer) ** alpha) * ((1/retorna_dist(vertice_atual, x, distancias)) ** beta)

    for i in range(n):
        if i in candidatos:
            p = (retorna_fer(vertice_atual, i, matriz_fer) ** alpha) * ((1/retorna_dist(vertice_atual, i, distancias)) ** beta) / divisor
            probabilidades.append(p)
        else: probabilidades.append(0.0)

    

    return probabilidades

def descobre_prox_vertice(probabilidades, candidatos):
    r = random.random()
    soma_acumulada = 0
    for i in candidatos:
        soma_acumulada += probabilidades[i]
        if r <= soma_acumulada:
            return i
    return candidatos[-1]

def atualiza_matriz_feromonios(matriz_fer, solucoes, matriz_dist, rho=0.5, Q=1.0):
    n = len(matriz_fer)
    for i in range(n):
        for j in range(n):
            if i != j:
                matriz_fer[i][j] *= (1 - rho)

    for solucao in solucoes:
        L_k = fitness(solucao, matriz_dist)
        delta_tau = Q / L_k 
        for i in range(len(solucao)):
            origem = solucao[i]
            destino = solucao[(i + 1) % n]
            matriz_fer[origem][destino] += delta_tau
            matriz_fer[destino][origem] += delta_tau

def executar_aco(matriz_dist, n_formigas, iteracoes, alpha=0.5, beta=0.5, rho=0.5):
    n_cidades = len(matriz_dist)
    matriz_fer = matriz_feromonios_inicial(n_cidades)
    
    melhor_solucao = None
    melhor_distancia = float('inf')
    
    historico_convergencia = []

    for t in range(iteracoes):
        todas_solucoes = []
        
        for k in range(n_formigas):
            v_atual = random.randint(0, n_cidades - 1)
            solucao = [v_atual]
            candidatos = [c for c in range(n_cidades) if c != v_atual]
            
            while candidatos:
                probs = calcula_p_trans(n_cidades, candidatos, v_atual, matriz_dist, matriz_fer, alpha, beta)
                v_proximo = descobre_prox_vertice(probs, candidatos)
                
                solucao.append(v_proximo)
                candidatos.remove(v_proximo)
                v_atual = v_proximo
            
            dist_atual = fitness(solucao, matriz_dist)
            if dist_atual < melhor_distancia:
                melhor_distancia = dist_atual
                melhor_solucao = solucao[:]
            
            todas_solucoes.append(solucao)
        
        atualiza_matriz_feromonios(matriz_fer, todas_solucoes, matriz_dist, rho)
        
        historico_convergencia.append(melhor_distancia)
        
        if (t + 1) % 10 == 0:
            print(f"Iteração {t+1}: Melhor Distância = {melhor_distancia}")

    # Geração do gráfico de convergência
    plt.figure(figsize=(10, 6))
    plt.plot(historico_convergencia, label='Melhor Distância Encontrada')
    plt.title('Curva de Convergência - Ant Colony Optimization (ACO)')
    plt.xlabel('Iteração')
    plt.ylabel('Distância (Função Objetivo)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()

    return melhor_solucao, melhor_distancia


def constroi_solucoes(n):
    solucoes=[]
    for i in range(n):
        solucao=[]
        candidatos=[x for x in range(n)]
        v_atual = i
        candidatos.pop(v_atual)
        solucao.append(v_atual)
        for _ in range(n-1):
            v_atual = descobre_prox_vertice(candidatos, v_atual)
            candidatos.pop(v_atual)
            solucao.append(v_atual)
        solucoes.append(solucao)

    return(solucoes)
        
            


def matriz_feromonios_inicial(n):
    matriz_fer = []
    for i in range(n):
        fer = []
        for j in range(n):
            if j == i:
                fer.append(float('inf'))
            else:
                fer.append(random.uniform(0, 1))
        matriz_fer.append(fer)

    for i in range(n):
        for j in range(n):
            if j < i:
                matriz_fer[i][j] = matriz_fer[j][i]

    return matriz_fer
            


d, n = carregar_grafo('lau15_dist.txt')
f = matriz_feromonios_inicial(n)


melhor_s, melhor_d = executar_aco(d, n_formigas=n, iteracoes=100)

print(f"\nResultado Final:\nDistância: {melhor_d}\nCaminho: {melhor_s}")