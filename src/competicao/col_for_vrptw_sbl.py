"""
col_for_vrptw.py — Colônia de Formigas (ACO) adaptada para o VRPTW.

ORIGEM: src/enxame/col_for.py (ACO clássico para o TSP: feromônio,
probabilidade de transição tau^alfa * eta^beta, roleta e evaporação).

O QUE FOI MANTIDO do algoritmo original:
  - a construção de soluções por formigas: cada formiga monta um caminho
    cliente a cliente escolhendo o próximo vértice por roleta sobre
    tau^alfa * eta^beta ('calcula_p_trans' + 'descobre_prox_vertice');
  - a matriz de feromônio com inicialização aleatória simétrica
    ('matriz_feromonios_inicial');
  - a atualização do feromônio: evaporação (1-rho) + depósito Q/L de cada
    formiga ('atualiza_matriz_feromonios');
  - o laço principal de 'executar_aco' (iterações x formigas).

O QUE FOI ADAPTADO (e por quê):
  - cada formiga constrói um TOUR GIGANTE (permutação dos clientes), que o
    decodificador comum divide em rotas viáveis (capacidade + janelas) —
    a mesma representação dos demais algoritmos. A formiga parte do
    depósito (vértice 0), então o feromônio do depósito também é usado;
  - VISIBILIDADE espaço-tempo: eta(i,j) = 1 / (d(i,j) * due_date_j) em vez
    de 1/d(i,j). É a mesma heurística "distância x urgência da janela" da
    população inicial do AG original — clientes com janela mais cedo ficam
    mais atraentes, o que reduz violações de janela na decodificação;
  - o depósito de feromônio passa a percorrer as ARESTAS DAS ROTAS
    DECODIFICADAS (incluindo idas/voltas ao depósito) e usa L = objetivo
    (veículos e distância), pois é isso que a competição avalia. No TSP
    original o depósito era no ciclo completo com L = distância;
  - depósito elitista: a melhor solução global reforça suas arestas a cada
    iteração (pequena adição clássica do ACO, melhora a convergência);
  - cálculo da probabilidade vetorizado com numpy — necessário porque a
    instância maior tem 400 clientes (o laço puro em Python do original
    não terminaria dentro do tempo limite da competição);
  - melhor solução global escolhida pelas REGRAS DE DEB;
  - critério de parada por tempo + busca local final + saída no formato
    exigido pela competição.
"""

import os
import random
import sys
import time

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from competicao.vrptw_comum import (INSTANCIAS_PADRAO, busca_local, caminho_instancia,
                         carregar_vrptw, decodificar_e_avaliar, eh_melhor_deb,
                         escrever_resultado, imprimir_resumo,
                         verificar_solucao)

# --------------------------- Parâmetros do ACO ------------------------------
AUTORES = "Lucas Rivetti, Ian Nunes"        # ajustar com os nomes do grupo
N_FORMIGAS = 20                  # formigas por iteração
NUM_ITE_MAX = 1000            # teto de iterações (para por tempo)
TEMPO_MAX_SEG = 60              # tempo máximo por instância
ALPHA = 1.0                      # peso do feromônio
BETA = 2.0                       # peso da visibilidade (heurística)
RHO = 0.10                       # taxa de evaporação
Q = 1000000.0                    # constante de depósito (escala do objetivo)
PESO_ELITISTA = 5.0              # reforço da melhor solução global
FRACAO_BUSCA_LOCAL = 0.25        # fração final do tempo p/ busca local


def matriz_feromonios_inicial(n):
    """Matriz de feromônio inicial aleatória e simétrica (como no original),
       agora em numpy. A diagonal não é usada (vértice atual fica fora dos
       candidatos), então recebe 1 apenas para evitar valores especiais."""
    matriz = np.random.uniform(0.0, 1.0, size=(n, n))
    matriz = (matriz + matriz.T) / 2.0      # simetriza
    np.fill_diagonal(matriz, 1.0)
    return matriz


def calcula_p_trans(candidatos, vertice_atual, matriz_fer, visibilidade,
                    alpha=ALPHA, beta=BETA):
    """Probabilidades de transição do ACO — mesma fórmula do original
       (tau^alfa * eta^beta normalizado), vetorizada com numpy."""
    tau = matriz_fer[vertice_atual, candidatos] ** alpha
    eta = visibilidade[vertice_atual, candidatos] ** beta
    pesos = tau * eta
    soma = pesos.sum()
    if soma <= 0.0:                    # degenerado: escolhe uniforme
        return np.full(len(candidatos), 1.0 / len(candidatos))
    return pesos / soma


def descobre_prox_vertice(probabilidades, candidatos):
    """Roleta sobre as probabilidades (mesmo papel da função original)."""
    r = random.random()
    soma_acumulada = 0.0
    for k in range(len(candidatos)):
        soma_acumulada += probabilidades[k]
        if r <= soma_acumulada:
            return candidatos[k]
    return candidatos[-1]


def constroi_tour(n, matriz_fer, visibilidade):
    """Uma formiga constrói o tour gigante: parte do depósito (0) e escolhe
       cliente a cliente pela regra de transição até visitar todos."""
    candidatos = list(range(1, n))
    v_atual = 0                        # depósito
    solucao = []
    while candidatos:
        probs = calcula_p_trans(candidatos, v_atual, matriz_fer, visibilidade)
        v_proximo = descobre_prox_vertice(probs, candidatos)
        solucao.append(v_proximo)
        candidatos.remove(v_proximo)
        v_atual = v_proximo
    return solucao


def arestas_das_rotas(rotas):
    """Arestas reais percorridas pela solução decodificada,
       incluindo as idas e voltas ao depósito."""
    arestas = []
    for rota in rotas:
        pos = 0
        for c in rota:
            arestas.append((pos, c))
            pos = c
        arestas.append((pos, 0))
    return arestas


def atualiza_matriz_feromonios(matriz_fer, solucoes_avaliadas, melhor_global,
                               rho=RHO, q=Q):
    """Evaporação + depósito (mesma estrutura do original): cada formiga
       deposita q/L nas arestas das suas ROTAS DECODIFICADAS, onde L é o
       objetivo (veículos*peso + distância). A melhor solução global
       deposita um reforço elitista adicional."""
    matriz_fer *= (1.0 - rho)          # evaporação em todas as arestas

    for rotas, aval in solucoes_avaliadas:
        delta = q / aval["objetivo"]
        for (i, j) in arestas_das_rotas(rotas):
            matriz_fer[i, j] += delta
            matriz_fer[j, i] += delta  # matriz simétrica, como no original

    # Depósito elitista da melhor solução encontrada até agora
    if melhor_global is not None:
        rotas, aval = melhor_global
        delta = PESO_ELITISTA * q / aval["objetivo"]
        for (i, j) in arestas_das_rotas(rotas):
            matriz_fer[i, j] += delta
            matriz_fer[j, i] += delta


def executar_aco(nome_instancia, tempo_max=TEMPO_MAX_SEG):
    """Executa o ACO completo em uma instância e grava o resultado."""
    caminho = caminho_instancia(nome_instancia)
    clientes, matriz, capacidade, max_veiculos, n = carregar_vrptw(caminho)

    inicio = time.time()
    tempo_evolucao = tempo_max * (1 - FRACAO_BUSCA_LOCAL)

    matriz_np = np.array(matriz)
    matriz_fer = matriz_feromonios_inicial(n)

    # Visibilidade espaço-tempo: 1 / (distância * urgência da janela).
    # O +0.1 e o +1 evitam divisão por zero (distância nula / janelas em 0).
    due = np.array([clientes[j]["due_date"] for j in range(n)])
    visibilidade = 1.0 / ((matriz_np + 0.1) * (due[None, :] + 1.0))
    np.fill_diagonal(visibilidade, 0.0)

    melhor_solucao = None              # tour gigante da melhor solução
    melhor_rotas_aval = None           # (rotas, aval) p/ depósito elitista
    melhor_iteracao = 0

    iteracao = 0
    while iteracao < NUM_ITE_MAX: # and time.time() - inicio < tempo_evolucao:
        solucoes_avaliadas = []
        for _ in range(N_FORMIGAS):
            solucao = constroi_tour(n, matriz_fer, visibilidade)
            rotas, aval = decodificar_e_avaliar(solucao, matriz, clientes,
                                                capacidade, max_veiculos)
            solucoes_avaliadas.append((rotas, aval))

            # Melhor global pelas regras de Deb
            if melhor_rotas_aval is None or eh_melhor_deb(
                    aval, melhor_rotas_aval[1]):
                melhor_solucao = solucao[:]
                melhor_rotas_aval = (rotas, aval)
                melhor_iteracao = iteracao

        atualiza_matriz_feromonios(matriz_fer, solucoes_avaliadas,
                                   melhor_rotas_aval)

        # Polimento memético: a cada 20 iterações, rajada curta de busca
        # local na melhor solução; se melhorar, o depósito elitista passa a
        # reforçar as arestas da solução polida.
        # if iteracao % 20 == 19:
        #     sol_bl, aval_bl, _ = busca_local(
        #         melhor_solucao, melhor_rotas_aval[1], matriz, clientes,
        #         capacidade, max_veiculos, max_tentativas=3000,
        #         tempo_limite=2.0)
        #     if eh_melhor_deb(aval_bl, melhor_rotas_aval[1]):
        #         melhor_solucao = sol_bl
        #         rotas_bl, aval_bl = decodificar_e_avaliar(
        #             sol_bl, matriz, clientes, capacidade, max_veiculos)
        #         melhor_rotas_aval = (rotas_bl, aval_bl)
        #         melhor_iteracao = iteracao

        if iteracao % 10 == 0:
            aval = melhor_rotas_aval[1]
            status = "VIAVEL" if aval["viavel"] else "INVIAVEL"
            print(f"Iteração {iteracao}: veículos = {aval['num_veiculos']}, "
                  f"distância = {aval['distancia']:.2f} [{status}]")
        iteracao += 1

    # --- BUSCA LOCAL na melhor solução (pós-processamento) ---
    # melhor_aval = melhor_rotas_aval[1]
    # tempo_restante = tempo_max - (time.time() - inicio)
    # melhor_solucao, melhor_aval, melhorias = busca_local(
    #     melhor_solucao, melhor_aval, matriz, clientes, capacidade,
    #     max_veiculos, max_tentativas=200000, tempo_limite=tempo_restante)
    # print(f"Busca local: {melhorias} melhorias aplicadas")

    tempo_exec = time.time() - inicio

    # Partição final sempre pelo split ótimo (a evolução pode usar o guloso)
    rotas, melhor_aval = decodificar_e_avaliar(
        melhor_solucao, matriz, clientes, capacidade, max_veiculos,
        metodo="split")
    viavel, problemas = verificar_solucao(
        rotas, matriz, clientes, capacidade, max_veiculos, n)

    parametros = (f"formigas={N_FORMIGAS}, alfa={ALPHA}, beta={BETA}, "
                  f"rho={RHO}, Q={Q:.0f}, elitista={PESO_ELITISTA}, "
                  f"iterações={iteracao}, tempo_max={tempo_max}s")
    imprimir_resumo(nome_instancia, "ACO", parametros, melhor_aval, rotas,
                    melhor_iteracao, tempo_exec, viavel, problemas)
    arquivo = escrever_resultado(nome_instancia, "aco", AUTORES, rotas,
                                 melhor_aval["distancia"], tempo_exec)
    print(f"Resultado salvo em: {arquivo}")
    return melhor_aval, tempo_exec


if __name__ == "__main__":
    # Uso: python col_for_vrptw.py [instancia|todas] [tempo_max_segundos]
    alvo = sys.argv[1] if len(sys.argv) > 1 else "c101"
    tempo = float(sys.argv[2]) if len(sys.argv) > 2 else TEMPO_MAX_SEG
    nomes = INSTANCIAS_PADRAO if alvo == "todas" else [alvo]
    for nome in nomes:
        print(f"\n########## ACO — instância {nome} ##########")
        executar_aco(nome, tempo)
