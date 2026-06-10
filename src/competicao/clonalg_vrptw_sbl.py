"""
clonalg_vrptw.py — CLONALG (Sistema Imunológico Artificial) adaptado
para o VRPTW.

ORIGEM: src/imunologico/clonalg.py (CLONALG para o TSP com anticorpos de
permutação, clonagem proporcional ao rank e hipermutação por troca).

O QUE FOI MANTIDO do algoritmo original:
  - anticorpos representados por PERMUTAÇÕES (mesma representação do TSP,
    agora interpretada como tour gigante e decodificada em rotas);
  - o ciclo do CLONALG: seleção dos melhores -> clonagem proporcional ao
    rank (Nc = round(beta * TAM_POP / posicao)) -> hipermutação ->
    nova população com substituição dos d piores por aleatórios;
  - a mutação por troca de duas posições (swap);
  - a estrutura das funções: selecao, clonagem, mutacao, hipermutacao,
    nova_populacao, clonalg.

O QUE FOI ADAPTADO (e por quê):
  - o fitness deixa de ser a distância do ciclo TSP e passa a ser a
    avaliação comum do VRPTW (tour gigante -> rotas viáveis; objetivo
    lexicográfico veículos/distância);
  - ordenações (seleção e nova população) usam a chave das REGRAS DE DEB:
    soluções viáveis primeiro (por objetivo), inviáveis depois (por
    violação) — o original ordenava só por distância pois não havia
    restrições;
  - HIPERMUTAÇÃO PROPORCIONAL AO RANK: no original todo clone sofria
    exatamente 1 troca; aqui clones de anticorpos pior ranqueados sofrem
    mais trocas (1 troca para o melhor, crescendo com o rank). É o
    princípio clássico do CLONALG (mutação inversamente proporcional à
    afinidade) aplicado à permutação;
  - avaliações calculadas uma única vez por anticorpo e reaproveitadas
    (o original recalculava o fitness em cada ordenação);
  - população inicial parcialmente heurística (permitido no enunciado);
  - critério de parada por tempo + busca local final + saída no formato
    exigido pela competição.
"""

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

# ------------------------- Parâmetros do CLONALG ----------------------------
AUTORES = "Lucas Rivetti, Ian Nunes"        # ajustar com os nomes do grupo
TAMANHO_P = 30                   # tamanho da população de anticorpos
NUM_GER_MAX = 1000000            # teto de gerações (para por tempo)
TEMPO_MAX_SEG = 120              # tempo máximo por instância
N_SELECIONADOS = 10              # anticorpos selecionados para clonagem
BETA = 1.0                       # fator de clonagem (Nc = beta*P/rank)
D_ALEATORIOS = 3                 # piores substituídos por aleatórios
FRACAO_SEMENTES = 0.3            # fração inicial heurística
FRACAO_BUSCA_LOCAL = 0.25        # fração final do tempo p/ busca local


def geracao_inicial(n, clientes, matriz):
    """População inicial: parte heurística (devido às janelas de tempo) e
       parte aleatória, como o shuffle do original."""
    qtd_sementes = max(1, int(TAMANHO_P * FRACAO_SEMENTES))
    populacao = solucoes_iniciais_heuristicas(n, clientes, matriz, qtd_sementes)
    while len(populacao) < TAMANHO_P:
        anticorpo = list(range(1, n))
        random.shuffle(anticorpo)
        populacao.append(anticorpo)
    return populacao


def selecao(populacao, avaliacoes, n_selecionados):
    """Seleciona os n melhores anticorpos ordenando pela chave de Deb
       (viáveis por objetivo, inviáveis por violação) — papel idêntico à
       'selecao' original, que ordenava por distância."""
    ordem = sorted(range(len(populacao)), key=lambda i: chave_deb(avaliacoes[i]))
    selecionados = [(populacao[i], avaliacoes[i]) for i in ordem[:n_selecionados]]
    return selecionados


def clonagem(selecionados, beta=BETA):
    """Clonagem proporcional ao rank — fórmula idêntica à original:
       Nc = round(beta * TAMANHO_P / posicao). Devolve também o rank do
       pai de cada clone, usado na intensidade da hipermutação."""
    clones = []
    for i, (anticorpo, _) in enumerate(selecionados):
        posicao = i + 1
        nc = round(beta * TAMANHO_P / posicao)
        for _ in range(nc):
            clones.append((anticorpo[:], posicao))
    return clones


def mutacao(solucao, num_trocas=1):
    """Mutação por troca (swap) — a mesma do original, agora aplicada
       'num_trocas' vezes para regular a intensidade."""
    nova_solucao = solucao[:]
    for _ in range(num_trocas):
        pos1 = random.randint(0, len(nova_solucao) - 1)
        pos2 = random.randint(0, len(nova_solucao) - 1)
        nova_solucao[pos1], nova_solucao[pos2] = (nova_solucao[pos2],
                                                  nova_solucao[pos1])
    return nova_solucao


def hipermutacao(clones):
    """Hipermutação inversamente proporcional à afinidade: clones do melhor
       anticorpo (rank 1) sofrem 1 troca; ranks piores sofrem mais trocas."""
    clones_mutados = []
    for clone, posicao in clones:
        num_trocas = 1 + (posicao - 1) // 2     # rank 1-2: 1 troca; 3-4: 2; ...
        clones_mutados.append(mutacao(clone, num_trocas))
    return clones_mutados


def nova_populacao(populacao, avaliacoes, clones_mutados, avals_clones,
                   n, d=D_ALEATORIOS):
    """Junta população e clones, mantém os TAMANHO_P - d melhores (ordem de
       Deb) e injeta d anticorpos aleatórios novos — estrutura idêntica à
       'nova_populacao' original (diversidade via recém-chegados)."""
    todos = list(zip(populacao, avaliacoes)) + list(zip(clones_mutados,
                                                        avals_clones))
    todos.sort(key=lambda par: chave_deb(par[1]))

    populacao_nova = [anticorpo for anticorpo, _ in todos[:TAMANHO_P - d]]
    avaliacoes_novas = [aval for _, aval in todos[:TAMANHO_P - d]]

    return populacao_nova, avaliacoes_novas


def executar_clonalg(nome_instancia, tempo_max=TEMPO_MAX_SEG):
    """Executa o CLONALG completo em uma instância e grava o resultado."""
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

    while geracao < NUM_GER_MAX: # and time.time() - inicio < tempo_evolucao:
        # 1. Seleção dos melhores anticorpos (ordem de Deb)
        selecionados = selecao(populacao, avaliacoes, N_SELECIONADOS)

        # Melhor global (o primeiro selecionado é o melhor da população)
        if melhor_aval is None or eh_melhor_deb(selecionados[0][1], melhor_aval):
            melhor_solucao = selecionados[0][0][:]
            melhor_aval = selecionados[0][1]
            melhor_geracao = geracao

        # 2. Clonagem proporcional ao rank
        clones = clonagem(selecionados)

        # 3. Hipermutação (intensidade cresce com o rank do pai)
        clones_mutados = hipermutacao(clones)
        avals_clones = [avaliar(c, matriz, clientes, capacidade, max_veiculos)
                        for c in clones_mutados]

        # 4. Nova população: melhores + d aleatórios novos
        populacao, avaliacoes = nova_populacao(
            populacao, avaliacoes, clones_mutados, avals_clones, n)
        for _ in range(D_ALEATORIOS):
            novo = list(range(1, n))
            random.shuffle(novo)
            populacao.append(novo)
            avaliacoes.append(avaliar(novo, matriz, clientes, capacidade,
                                      max_veiculos))

        # Polimento memético: a cada 50 gerações, rajada curta de busca
        # local no melhor anticorpo, reinjetado no lugar do pior.
        # if geracao % 50 == 49 and melhor_solucao is not None:
        #     melhor_solucao, melhor_aval, melhorias_bl = busca_local(
        #         melhor_solucao, melhor_aval, matriz, clientes, capacidade,
        #         max_veiculos, max_tentativas=3000, tempo_limite=2.0)
        #     if melhorias_bl:
        #         melhor_geracao = geracao
        #     pior = max(range(len(populacao)),
        #                key=lambda i: chave_deb(avaliacoes[i]))
        #     populacao[pior] = melhor_solucao[:]
        #     avaliacoes[pior] = melhor_aval

        if geracao % 20 == 0:
            status = "VIAVEL" if melhor_aval["viavel"] else "INVIAVEL"
            print(f"Geração {geracao}: veículos = {melhor_aval['num_veiculos']}, "
                  f"distância = {melhor_aval['distancia']:.2f} [{status}]")
        geracao += 1

    # --- BUSCA LOCAL na melhor solução (pós-processamento) ---
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
    # Uso: python clonalg_vrptw.py [instancia|todas] [tempo_max_segundos]
    alvo = sys.argv[1] if len(sys.argv) > 1 else "c101"
    tempo = float(sys.argv[2]) if len(sys.argv) > 2 else TEMPO_MAX_SEG
    nomes = INSTANCIAS_PADRAO if alvo == "todas" else [alvo]
    for nome in nomes:
        print(f"\n########## CLONALG — instância {nome} ##########")
        executar_clonalg(nome, tempo)
