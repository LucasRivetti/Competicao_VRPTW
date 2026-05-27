import random
import numpy as np
import matplotlib.pyplot as plt


TAMANHO_P = 20
GERACOES = 100


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


def f_obj(valores):
    return np.sum(valores)


def fitness(solucao, matriz):
    distancias = []

    for i in range(len(solucao)):
        if i == (len(solucao) - 1):
            distancia = retorna_dist(solucao[i], solucao[0], matriz)
        else:
            distancia = retorna_dist(solucao[i], solucao[i + 1], matriz)

        distancias.append(distancia)
    
    return f_obj(distancias)


def criar_dicionario(n):
    dicionario = {}

    for indice in range(n):
        cidade = indice + 1
        dicionario[indice] = cidade

    return dicionario


def trocar_indices(solucao, dicionario):
    rota = []

    for indice in solucao:
        rota.append(dicionario[indice])

    rota.append(dicionario[solucao[0]])

    return rota


def geracao_inicial(n):
    populacao = []

    for _ in range(TAMANHO_P):
        anticorpo = [x for x in range(n)]
        random.shuffle(anticorpo)
        populacao.append(anticorpo)

    return populacao


def selecao(populacao, matriz_adj, n_selecionados):
    fitness_calculo = []

    for anticorpo in populacao:
        valor = fitness(anticorpo, matriz_adj)
        fitness_calculo.append((valor, anticorpo))

    fitness_calculo.sort()

    selecionados = []

    for i in range(n_selecionados):
        selecionados.append(fitness_calculo[i][1])

    return selecionados


def clonagem(selecionados, beta):
    clones = []

    for i in range(len(selecionados)):
        posicao = i + 1

        Nc = round(beta * TAMANHO_P / posicao)

        for _ in range(Nc):
            clones.append(selecionados[i].copy())

    return clones


def mutacao(solucao):
    nova_solucao = solucao.copy()

    pos1 = random.randint(0, len(nova_solucao) - 1)
    pos2 = random.randint(0, len(nova_solucao) - 1)

    aux = nova_solucao[pos1]
    nova_solucao[pos1] = nova_solucao[pos2]
    nova_solucao[pos2] = aux

    return nova_solucao


def hipermutacao(clones):
    clones_mutados = []

    for clone in clones:
        clone_mutado = mutacao(clone)
        clones_mutados.append(clone_mutado)

    return clones_mutados


def nova_populacao(populacao, clones_mutados, matriz_adj, n, d):
    todos = populacao + clones_mutados

    avaliados = []

    for anticorpo in todos:
        valor = fitness(anticorpo, matriz_adj)
        avaliados.append((valor, anticorpo))

    avaliados.sort()

    populacao_nova = []

    for i in range(TAMANHO_P - d):
        populacao_nova.append(avaliados[i][1])

    for _ in range(d):
        novo = [x for x in range(n)]
        random.shuffle(novo)
        populacao_nova.append(novo)

    return populacao_nova


def clonalg(caminho_arquivo, n_selecionados, d, beta):
    matriz_adj, n = carregar_grafo(caminho_arquivo)

    dicionario = criar_dicionario(n)

    populacao = geracao_inicial(n)

    melhor_solucao = None
    melhor_distancia = float("inf")

    historico = []

    for geracao in range(GERACOES):
        selecionados = selecao(populacao, matriz_adj, n_selecionados)

        clones = clonagem(selecionados, beta)

        clones_mutados = hipermutacao(clones)

        populacao = nova_populacao(
            populacao,
            clones_mutados,
            matriz_adj,
            n,
            d
        )

        melhor_atual = selecao(populacao, matriz_adj, 1)[0]
        distancia_atual = fitness(melhor_atual, matriz_adj)

        if distancia_atual < melhor_distancia:
            melhor_distancia = distancia_atual
            melhor_solucao = melhor_atual.copy()

        historico.append(melhor_distancia)

        print("Geração:", geracao + 1, "| Melhor distância:", melhor_distancia)

    melhor_rota = trocar_indices(melhor_solucao, dicionario)

    return melhor_solucao, melhor_rota, melhor_distancia, historico

caminho_arquivo = "LAU15.txt"

valores_n = [5, 10, 15]
valores_d = [2, 4, 6]
valores_beta = [1, 2, 3]

resultados = []

for n_selecionados in valores_n:
    for d in valores_d:
        for beta in valores_beta:

            print()
            print("======================================")
            print("Executando teste:")
            print("n selecionados =", n_selecionados)
            print("d =", d)
            print("beta =", beta)
            print("======================================")

            melhor_solucao, melhor_rota, melhor_distancia, historico = clonalg(
                caminho_arquivo,
                n_selecionados=n_selecionados,
                d=d,
                beta=beta
            )

            resultados.append([
                n_selecionados,
                d,
                beta,
                melhor_distancia,
                melhor_rota,
                historico
            ])

print()
print("========== RESUMO FINAL ==========")

for resultado in resultados:
    print()
    print("n selecionados:", resultado[0])
    print("d:", resultado[1])
    print("beta:", resultado[2])
    print("melhor distância:", resultado[3])
    print("melhor rota:", resultado[4])

melhor_resultado = resultados[0]

for resultado in resultados:
    if resultado[3] < melhor_resultado[3]:
        melhor_resultado = resultado

print()
print("========== MELHOR CONFIGURAÇÃO ==========")
print("n selecionados:", melhor_resultado[0])
print("d:", melhor_resultado[1])
print("beta:", melhor_resultado[2])
print("melhor distância:", melhor_resultado[3])
print("melhor rota:", melhor_resultado[4])

melhor_historico = melhor_resultado[5]

plt.plot(melhor_historico)
plt.xlabel("Geração")
plt.ylabel("Melhor distância")
plt.title("Evolução do melhor teste fatorial")
plt.show()