"""
de_vrptw.py — Evolução Diferencial (DE) adaptada para o VRPTW.

ORIGEM: src/evolucao_diferencial/de_booth.py (DE contínua com vetor
mutante rand/1, recombinação BLX-alfa e seleção gulosa um-a-um).

O QUE FOI MANTIDO do algoritmo original:
  - o vetor mutante DE/rand/1 ('vetor_mutante'):
        vm = x_r1 + F * (x_r2 - x_r3),  com F = 0.6;
  - a recombinação BLX-alfa ('combinacao') entre o indivíduo alvo e o
    vetor mutante, com alfa = 0.5 (era a variante usada no original);
  - a seleção gulosa: o filho substitui o pai somente se for melhor
    (estrutura da função 'selecao' original);
  - o laço principal: a cada geração, toda a população passa por
    mutação diferencial + recombinação + seleção.

O QUE FOI ADAPTADO (e por quê):
  - REPRESENTAÇÃO POR CHAVES ALEATÓRIAS (random keys): a DE opera sobre
    vetores reais; cada indivíduo é um vetor com uma dimensão por cliente
    e a permutação (tour gigante) é obtida por argsort das chaves. Assim
    os operadores aritméticos da DE permanecem INTACTOS;
  - no original havia duas listas separadas (populacao1/populacao2 para
    x1 e x2); aqui cada indivíduo é um único vetor numpy de N dimensões —
    a mesma fórmula é aplicada elemento a elemento;
  - a comparação pai x filho usa as REGRAS DE DEB (viável > inviável)
    em vez de comparar apenas f(x), pois o problema tem restrições;
  - limites das variáveis passam a ser [0,1] (chaves), como o original
    limitava a [-10,10];
  - parte da população inicial é semeada com soluções heurísticas
    convertidas em chaves;
  - critério de parada por tempo + busca local final + saída no formato
    exigido pela competição.
"""

import os
import random
import sys
import time

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from vrptw_comum import (INSTANCIAS_PADRAO, avaliar, busca_local,
                         caminho_instancia, carregar_vrptw, chave_deb,
                         chaves_para_permutacao, decodificar_e_avaliar,
                         eh_melhor_deb, escrever_resultado, imprimir_resumo,
                         permutacao_para_chaves,
                         solucoes_iniciais_heuristicas, verificar_solucao)

# --------------------------- Parâmetros da DE -------------------------------
AUTORES = "Lucas Rivetti, Ian Nunes"        # ajustar com os nomes do grupo
TAM_POP = 60                     # tamanho da população
NUM_GER_MAX = 1000000            # teto de gerações (para por tempo)
TEMPO_MAX_SEG = 120              # tempo máximo por instância
F = 0.6                          # mesmo fator de escala do original
ALPHA_BLX = 0.5                  # mesmo alfa da recombinação original
LIM_INF, LIM_SUP = 0.0, 1.0      # domínio das chaves aleatórias
FRACAO_SEMENTES = 0.5            # fração da população semeada c/ heurísticas
FRACAO_BUSCA_LOCAL = 0.25        # fração final do tempo p/ busca local


def vetor_mutante(indice, populacao, f_escala=F):
    """DE/rand/1 — mesma fórmula do original, agora vetorizada:
       vm = x_r1 + F * (x_r2 - x_r3), com r1, r2, r3 distintos do alvo."""
    intervalo = list(range(len(populacao)))
    intervalo.pop(indice)
    r1, r2, r3 = random.sample(intervalo, 3)
    return populacao[r1] + f_escala * (populacao[r2] - populacao[r3])


def combinacao(pai, vm, alpha=ALPHA_BLX):
    """Recombinação BLX-alfa — mesma fórmula do original, aplicada
       elemento a elemento: sorteia em [min - a*d, max + a*d]."""
    d = np.abs(pai - vm)
    menor = np.minimum(pai, vm)
    maior = np.maximum(pai, vm)
    return np.random.uniform(menor - alpha * d, maior + alpha * d)


def selecao(populacao, avaliacoes, matriz, clientes, capacidade, max_veiculos):
    """Uma geração completa da DE (estrutura da função 'selecao' original):
       para cada indivíduo gera o vetor mutante, recombina, limita ao
       domínio, avalia e mantém o melhor entre pai e filho (regras de Deb)."""
    populacao_final = []
    avaliacoes_finais = []
    for i in range(len(populacao)):
        vm = vetor_mutante(i, populacao)
        u = combinacao(populacao[i], vm)
        u = np.clip(u, LIM_INF, LIM_SUP)

        aval_u = avaliar(chaves_para_permutacao(u), matriz, clientes,
                         capacidade, max_veiculos)

        # Seleção gulosa pai x filho decidida pelas regras de Deb
        if eh_melhor_deb(aval_u, avaliacoes[i]):
            populacao_final.append(u)
            avaliacoes_finais.append(aval_u)
        else:
            populacao_final.append(populacao[i])
            avaliacoes_finais.append(avaliacoes[i])

    return populacao_final, avaliacoes_finais


def executar_de(nome_instancia, tempo_max=TEMPO_MAX_SEG):
    """Executa a DE completa em uma instância e grava o resultado."""
    caminho = caminho_instancia(nome_instancia)
    clientes, matriz, capacidade, max_veiculos, n = carregar_vrptw(caminho)
    dim = n - 1                       # uma chave por cliente

    inicio = time.time()
    tempo_evolucao = tempo_max * (1 - FRACAO_BUSCA_LOCAL)

    # --- População inicial: parte heurística (em chaves), parte aleatória ---
    qtd_sementes = int(TAM_POP * FRACAO_SEMENTES)
    sementes = solucoes_iniciais_heuristicas(n, clientes, matriz, qtd_sementes)
    populacao = [permutacao_para_chaves(p) for p in sementes]
    while len(populacao) < TAM_POP:
        populacao.append(np.random.uniform(LIM_INF, LIM_SUP, size=dim))

    avaliacoes = [avaliar(chaves_para_permutacao(ind), matriz, clientes,
                          capacidade, max_veiculos) for ind in populacao]

    melhor_geracao = 0
    geracao = 0
    idx_melhor = 0
    for i in range(1, TAM_POP):
        if eh_melhor_deb(avaliacoes[i], avaliacoes[idx_melhor]):
            idx_melhor = i
    melhor_aval = avaliacoes[idx_melhor]
    melhor_chaves = populacao[idx_melhor].copy()

    while geracao < NUM_GER_MAX and time.time() - inicio < tempo_evolucao:
        populacao, avaliacoes = selecao(populacao, avaliacoes, matriz,
                                        clientes, capacidade, max_veiculos)

        # Acompanha a melhor solução global (regras de Deb)
        for i in range(TAM_POP):
            if eh_melhor_deb(avaliacoes[i], melhor_aval):
                melhor_aval = avaliacoes[i]
                melhor_chaves = populacao[i].copy()
                melhor_geracao = geracao

        # Polimento memético: a cada 25 gerações, rajada curta de busca
        # local no melhor global; se melhorar, ele volta para a população
        # (em chaves) no lugar do pior indivíduo.
        if geracao % 25 == 24:
            perm_melhor = chaves_para_permutacao(melhor_chaves)
            perm_melhor, aval_bl, _ = busca_local(
                perm_melhor, melhor_aval, matriz, clientes, capacidade,
                max_veiculos, max_tentativas=3000, tempo_limite=2.0)
            if eh_melhor_deb(aval_bl, melhor_aval):
                melhor_aval = aval_bl
                melhor_chaves = permutacao_para_chaves(perm_melhor)
                melhor_geracao = geracao
                pior = max(range(TAM_POP),
                           key=lambda i: chave_deb(avaliacoes[i]))
                populacao[pior] = melhor_chaves.copy()
                avaliacoes[pior] = melhor_aval

        if geracao % 20 == 0:
            status = "VIAVEL" if melhor_aval["viavel"] else "INVIAVEL"
            print(f"Geração {geracao}: veículos = {melhor_aval['num_veiculos']}, "
                  f"distância = {melhor_aval['distancia']:.2f} [{status}]")
        geracao += 1

    # --- BUSCA LOCAL na melhor solução (pós-processamento) ---
    melhor_perm = chaves_para_permutacao(melhor_chaves)
    tempo_restante = tempo_max - (time.time() - inicio)
    melhor_perm, melhor_aval, melhorias = busca_local(
        melhor_perm, melhor_aval, matriz, clientes, capacidade, max_veiculos,
        max_tentativas=200000, tempo_limite=tempo_restante)
    print(f"Busca local: {melhorias} melhorias aplicadas")

    tempo_exec = time.time() - inicio

    # Partição final sempre pelo split ótimo (a evolução pode usar o guloso)
    rotas, melhor_aval = decodificar_e_avaliar(
        melhor_perm, matriz, clientes, capacidade, max_veiculos,
        metodo="split")
    viavel, problemas = verificar_solucao(
        rotas, matriz, clientes, capacidade, max_veiculos, n)

    parametros = (f"pop={TAM_POP}, F={F}, alpha_blx={ALPHA_BLX}, "
                  f"gerações={geracao}, tempo_max={tempo_max}s")
    imprimir_resumo(nome_instancia, "DE", parametros, melhor_aval, rotas,
                    melhor_geracao, tempo_exec, viavel, problemas)
    arquivo = escrever_resultado(nome_instancia, "de", AUTORES, rotas,
                                 melhor_aval["distancia"], tempo_exec)
    print(f"Resultado salvo em: {arquivo}")
    return melhor_aval, tempo_exec


if __name__ == "__main__":
    # Uso: python de_vrptw.py [instancia|todas] [tempo_max_segundos]
    alvo = sys.argv[1] if len(sys.argv) > 1 else "c101"
    tempo = float(sys.argv[2]) if len(sys.argv) > 2 else TEMPO_MAX_SEG
    nomes = INSTANCIAS_PADRAO if alvo == "todas" else [alvo]
    for nome in nomes:
        print(f"\n########## DE — instância {nome} ##########")
        executar_de(nome, tempo)
