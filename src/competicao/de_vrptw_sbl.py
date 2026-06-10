import os
import random
import sys
import time

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from competicao.vrptw_comum import (INSTANCIAS_PADRAO, avaliar,
                         caminho_instancia, carregar_vrptw, chave_deb,
                         chaves_para_permutacao, decodificar_e_avaliar,
                         eh_melhor_deb, escrever_resultado, imprimir_resumo,
                         permutacao_para_chaves,
                         solucoes_iniciais_heuristicas, verificar_solucao)

AUTORES = "Lucas Rivetti, Ian Nunes"
TAM_POP = 100
NUM_GER_MAX = 1000
TEMPO_MAX_SEG = 30
F = 0.6
ALPHA_BLX = 0.5
LIM_INF, LIM_SUP = 0.0, 1.0  # domínio das chaves aleatórias
FRACAO_SEMENTES = 0.5
FRACAO_BUSCA_LOCAL = 0.0
DIVERSIDADE_MIN = 0.05
FRACAO_REINJECAO = 0.30


def vetor_mutante(indice, populacao, f_escala=F):
    """DE/rand/1: vm = x_r1 + F * (x_r2 - x_r3)."""
    intervalo = list(range(len(populacao)))
    intervalo.pop(indice)
    r1, r2, r3 = random.sample(intervalo, 3)
    return populacao[r1] + f_escala * (populacao[r2] - populacao[r3])


def combinacao(pai, vm, alpha=ALPHA_BLX):
    """Recombinação BLX-alfa: sorteia em [min - a*d, max + a*d]."""
    d = np.abs(pai - vm)
    menor = np.minimum(pai, vm)
    maior = np.maximum(pai, vm)
    return np.random.uniform(menor - alpha * d, maior + alpha * d)


def selecao(populacao, avaliacoes, matriz, clientes, capacidade, max_veiculos):
    populacao_final = []
    avaliacoes_finais = []
    for i in range(len(populacao)):
        vm = vetor_mutante(i, populacao)
        u = combinacao(populacao[i], vm)
        u = np.clip(u, LIM_INF, LIM_SUP)

        aval_u = avaliar(chaves_para_permutacao(u), matriz, clientes,
                         capacidade, max_veiculos)

        if eh_melhor_deb(aval_u, avaliacoes[i]):
            populacao_final.append(u)
            avaliacoes_finais.append(aval_u)
        else:
            populacao_final.append(populacao[i])
            avaliacoes_finais.append(avaliacoes[i])

    return populacao_final, avaliacoes_finais


def executar_de(nome_instancia, tempo_max=TEMPO_MAX_SEG):
    caminho = caminho_instancia(nome_instancia)
    clientes, matriz, capacidade, max_veiculos, n = carregar_vrptw(caminho)
    dim = n - 1  # uma chave por cliente

    inicio = time.time()
    tempo_evolucao = tempo_max * (1 - FRACAO_BUSCA_LOCAL)

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

        for i in range(TAM_POP):
            if eh_melhor_deb(avaliacoes[i], melhor_aval):
                melhor_aval = avaliacoes[i]
                melhor_chaves = populacao[i].copy()
                melhor_geracao = geracao

        # Quando o desvio padrão médio cai abaixo de DIVERSIDADE_MIN,
        # F*(x_r2 - x_r3) tende a zero e a DE para de explorar; reinjeta
        # os piores para restaurar amplitude das mutações.
        pop_array = np.array(populacao)
        if float(np.mean(np.std(pop_array, axis=0))) < DIVERSIDADE_MIN:
            qtd = int(TAM_POP * FRACAO_REINJECAO)
            piores = sorted(range(TAM_POP),
                            key=lambda i: chave_deb(avaliacoes[i]),
                            reverse=True)[:qtd]
            for i in piores:
                populacao[i] = np.random.uniform(LIM_INF, LIM_SUP, size=dim)
                avaliacoes[i] = avaliar(chaves_para_permutacao(populacao[i]),
                                        matriz, clientes, capacidade, max_veiculos)

        if geracao % 20 == 0:
            status = "VIAVEL" if melhor_aval["viavel"] else "INVIAVEL"
            print(f"Geração {geracao}: veículos = {melhor_aval['num_veiculos']}, "
                  f"distância = {melhor_aval['distancia']:.2f} [{status}]")
        geracao += 1

    melhor_perm = chaves_para_permutacao(melhor_chaves)

    tempo_exec = time.time() - inicio

    rotas, melhor_aval = decodificar_e_avaliar(
        melhor_perm, matriz, clientes, capacidade, max_veiculos,
        metodo="split")
    viavel, problemas = verificar_solucao(
        rotas, matriz, clientes, capacidade, max_veiculos, n)

    parametros = (f"pop={TAM_POP}, F={F}, alpha_blx={ALPHA_BLX}, "
                  f"div_min={DIVERSIDADE_MIN}, reinjecao={FRACAO_REINJECAO}, "
                  f"gerações={geracao}, tempo_max={tempo_max}s")
    imprimir_resumo(nome_instancia, "DE", parametros, melhor_aval, rotas,
                    melhor_geracao, tempo_exec, viavel, problemas)
    arquivo = escrever_resultado(nome_instancia, "de", AUTORES, rotas,
                                 melhor_aval["distancia"], tempo_exec)
    print(f"Resultado salvo em: {arquivo}")
    return melhor_aval, tempo_exec


if __name__ == "__main__":
    # Uso: python de_vrptw_sbl.py [instancia|todas] [tempo_max_segundos]
    alvo = sys.argv[1] if len(sys.argv) > 1 else "c101"
    tempo = float(sys.argv[2]) if len(sys.argv) > 2 else TEMPO_MAX_SEG
    nomes = INSTANCIAS_PADRAO if alvo == "todas" else [alvo]
    for nome in nomes:
        print(f"\n########## DE — instância {nome} ##########")
        executar_de(nome, tempo)
