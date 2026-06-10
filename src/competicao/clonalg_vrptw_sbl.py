import os
import random
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from competicao.vrptw_comum import (INSTANCIAS_PADRAO, avaliar, busca_local,
                         caminho_instancia, carregar_vrptw, chave_deb,
                         decodificar_e_avaliar, eh_melhor_deb,
                         escrever_resultado, imprimir_resumo,
                         solucoes_iniciais_heuristicas, verificar_solucao)

AUTORES = "Lucas Rivetti, Ian Nunes"
TAMANHO_P = 30
NUM_GER_MAX = 1000
TEMPO_MAX_SEG = 60
N_SELECIONADOS = 10
BETA = 1.0
D_ALEATORIOS = 3
FRACAO_SEMENTES = 0.3
FRACAO_BUSCA_LOCAL = 0.25


def geracao_inicial(n, clientes, matriz):
    qtd_sementes = max(1, int(TAMANHO_P * FRACAO_SEMENTES))
    populacao = solucoes_iniciais_heuristicas(n, clientes, matriz, qtd_sementes)
    while len(populacao) < TAMANHO_P:
        anticorpo = list(range(1, n))
        random.shuffle(anticorpo)
        populacao.append(anticorpo)
    return populacao


def selecao(populacao, avaliacoes, n_selecionados):
    ordem = sorted(range(len(populacao)), key=lambda i: chave_deb(avaliacoes[i]))
    selecionados = [(populacao[i], avaliacoes[i]) for i in ordem[:n_selecionados]]
    return selecionados


def clonagem(selecionados, beta=BETA):
    """Nc = round(beta * TAMANHO_P / posicao); devolve também o rank do pai."""
    clones = []
    for i, (anticorpo, _) in enumerate(selecionados):
        posicao = i + 1
        nc = round(beta * TAMANHO_P / posicao)
        for _ in range(nc):
            clones.append((anticorpo[:], posicao))
    return clones


def mutacao(solucao, num_trocas=1):
    nova_solucao = solucao[:]
    for _ in range(num_trocas):
        pos1 = random.randint(0, len(nova_solucao) - 1)
        pos2 = random.randint(0, len(nova_solucao) - 1)
        nova_solucao[pos1], nova_solucao[pos2] = (nova_solucao[pos2],
                                                  nova_solucao[pos1])
    return nova_solucao


def hipermutacao(clones):
    """Mutação inversamente proporcional à afinidade: rank 1-2 → 1 troca; 3-4 → 2; ..."""
    clones_mutados = []
    for clone, posicao in clones:
        num_trocas = 1 + (posicao - 1) // 2
        clones_mutados.append(mutacao(clone, num_trocas))
    return clones_mutados


def nova_populacao(populacao, avaliacoes, clones_mutados, avals_clones,
                   n, d=D_ALEATORIOS):
    todos = list(zip(populacao, avaliacoes)) + list(zip(clones_mutados,
                                                        avals_clones))
    todos.sort(key=lambda par: chave_deb(par[1]))

    populacao_nova = [anticorpo for anticorpo, _ in todos[:TAMANHO_P - d]]
    avaliacoes_novas = [aval for _, aval in todos[:TAMANHO_P - d]]

    return populacao_nova, avaliacoes_novas


def executar_clonalg(nome_instancia, tempo_max=TEMPO_MAX_SEG):
    caminho = caminho_instancia(nome_instancia)
    clientes, matriz, capacidade, max_veiculos, n = carregar_vrptw(caminho)

    inicio = time.time()
    tempo_evolucao = tempo_max * (1 - FRACAO_BUSCA_LOCAL)

    populacao = geracao_inicial(n, clientes, matriz)
    avaliacoes = [avaliar(a, matriz, clientes, capacidade, max_veiculos)
                  for a in populacao]

    melhor_solucao = None
    melhor_aval = None
    melhor_geracao = 0
    geracao = 0

    while geracao < NUM_GER_MAX:
        selecionados = selecao(populacao, avaliacoes, N_SELECIONADOS)

        if melhor_aval is None or eh_melhor_deb(selecionados[0][1], melhor_aval):
            melhor_solucao = selecionados[0][0][:]
            melhor_aval = selecionados[0][1]
            melhor_geracao = geracao

        clones = clonagem(selecionados)
        clones_mutados = hipermutacao(clones)
        avals_clones = [avaliar(c, matriz, clientes, capacidade, max_veiculos)
                        for c in clones_mutados]

        populacao, avaliacoes = nova_populacao(
            populacao, avaliacoes, clones_mutados, avals_clones, n)
        for _ in range(D_ALEATORIOS):
            novo = list(range(1, n))
            random.shuffle(novo)
            populacao.append(novo)
            avaliacoes.append(avaliar(novo, matriz, clientes, capacidade,
                                      max_veiculos))

        if geracao % 20 == 0:
            status = "VIAVEL" if melhor_aval["viavel"] else "INVIAVEL"
            print(f"Geração {geracao}: veículos = {melhor_aval['num_veiculos']}, "
                  f"distância = {melhor_aval['distancia']:.2f} [{status}]")
        geracao += 1

    tempo_exec = time.time() - inicio

    rotas, melhor_aval = decodificar_e_avaliar(
        melhor_solucao, matriz, clientes, capacidade, max_veiculos,
        metodo="split")
    viavel, problemas = verificar_solucao(
        rotas, matriz, clientes, capacidade, max_veiculos, n)

    parametros = (f"pop={TAMANHO_P}, selecionados={N_SELECIONADOS}, "
                  f"beta={BETA}, d={D_ALEATORIOS}, gerações={geracao}, "
                  f"tempo_max={tempo_max}s")
    imprimir_resumo(nome_instancia, "CLONALG", parametros, melhor_aval, rotas,
                    melhor_geracao, tempo_exec, viavel, problemas)
    arquivo = escrever_resultado(nome_instancia, "clonalg", AUTORES, rotas,
                                 melhor_aval["distancia"], tempo_exec)
    print(f"Resultado salvo em: {arquivo}")
    return melhor_aval, tempo_exec


if __name__ == "__main__":
    # Uso: python clonalg_vrptw_sbl.py [instancia|todas] [tempo_max_segundos]
    alvo = sys.argv[1] if len(sys.argv) > 1 else "c101"
    tempo = float(sys.argv[2]) if len(sys.argv) > 2 else TEMPO_MAX_SEG
    nomes = INSTANCIAS_PADRAO if alvo == "todas" else [alvo]
    for nome in nomes:
        print(f"\n########## CLONALG — instância {nome} ##########")
        executar_clonalg(nome, tempo)
