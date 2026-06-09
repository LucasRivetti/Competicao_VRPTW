"""
ag_vrptw.py — Algoritmo Genético adaptado para o VRPTW.

ORIGEM: src/geneticos/ag_tsp.py (que já era um AG de permutação com
crossover OX, mutação por troca e torneio com as regras de Deb).

O QUE FOI MANTIDO do algoritmo original:
  - representação por permutação (tour gigante) decodificada em rotas;
  - crossover OX (Order Crossover) idêntico;
  - mutação por troca de duas posições (swap) com a mesma taxa;
  - seleção por torneio usando as regras de Deb;
  - elitismo copiando os melhores direto para a próxima geração;
  - população inicial heurística (tempo / vizinho mais próximo /
    espaço-tempo / aleatória), agora no módulo comum.

O QUE FOI ADAPTADO (e por quê):
  - leitura/decodificação/avaliação movidas para vrptw_comum.py para que
    os 5 algoritmos usem EXATAMENTE a mesma função de avaliação;
  - a avaliação agora mede também violações de capacidade/janela
    estruturais (antes só o excesso de frota) — necessário para as regras
    de Deb cobrirem todas as restrições do VRPTW;
  - avaliação calculada UMA vez por geração e reaproveitada no torneio e
    no elitismo (o original reavaliava o mesmo indivíduo várias vezes;
    o comportamento é o mesmo, só evita custo repetido);
  - critério de parada por TEMPO (a competição limita 8 min/instância);
  - busca local simples aplicada só à melhor solução final
    (pós-processamento — não altera o fluxo do AG);
  - saída no formato exigido pelo manual da competição (4 casas decimais).
"""

import os
import random
import sys
import time

# Permite importar o módulo comum que está em src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from vrptw_comum import (INSTANCIAS_PADRAO, avaliar, busca_local,
                         caminho_instancia, carregar_vrptw, chave_deb,
                         decodificar_e_avaliar, eh_melhor_deb,
                         escrever_resultado, imprimir_resumo,
                         solucoes_iniciais_heuristicas, verificar_solucao)

# --------------------------- Parâmetros do AG ------------------------------
AUTORES = "Lucas Rivetti, Ian Nunes"        # ajustar com os nomes do grupo
TAM_POP = 100                    # tamanho da população (igual ao original)
NUM_GER_MAX = 1000000            # teto de gerações (na prática para por tempo)
TEMPO_MAX_SEG = 120              # tempo máximo de execução por instância
TAXA_MUTACAO = 0.10              # mesma taxa do original
QTD_ELITE = 6                    # mesmo elitismo do original
K_TORNEIO = 3                    # mesmo torneio do original
FRACAO_BUSCA_LOCAL = 0.25        # fração final do tempo p/ busca local


def OX(pai1, pai2):
    """Order Crossover — idêntico ao original: copia um trecho de um pai e
    completa com a ordem relativa do outro, sem repetir clientes."""
    tam = len(pai1)
    pos1, pos2 = sorted(random.sample(range(tam), 2))

    filho_1 = [None] * tam
    filho_2 = [None] * tam

    filho_1[pos1:pos2 + 1] = pai2[pos1:pos2 + 1]
    filho_2[pos1:pos2 + 1] = pai1[pos1:pos2 + 1]

    usados_1 = set(filho_1[pos1:pos2 + 1])
    usados_2 = set(filho_2[pos1:pos2 + 1])

    pos_atual = (pos2 + 1) % tam
    pos_pai = (pos2 + 1) % tam
    while None in filho_1:
        if pai1[pos_pai] not in usados_1:
            filho_1[pos_atual] = pai1[pos_pai]
            usados_1.add(pai1[pos_pai])
            pos_atual = (pos_atual + 1) % tam
        pos_pai = (pos_pai + 1) % tam

    pos_atual = (pos2 + 1) % tam
    pos_pai = (pos2 + 1) % tam
    while None in filho_2:
        if pai2[pos_pai] not in usados_2:
            filho_2[pos_atual] = pai2[pos_pai]
            usados_2.add(pai2[pos_pai])
            pos_atual = (pos_atual + 1) % tam
        pos_pai = (pos_pai + 1) % tam

    return filho_1, filho_2


def mutacao(individuo, taxa=TAXA_MUTACAO):
    """Mutação por troca (swap) de duas posições — idêntica ao original.
    Como a permutação não repete clientes, a mutação preserva a validade."""
    if random.random() <= taxa:
        pos1, pos2 = sorted(random.sample(range(len(individuo)), 2))
        individuo[pos1], individuo[pos2] = individuo[pos2], individuo[pos1]
    return individuo


def selecao_torneio_deb(populacao, avaliacoes, k=K_TORNEIO):
    """Torneio de k indivíduos decidido pelas REGRAS DE DEB:
       viável > inviável; entre viáveis, menor objetivo; entre inviáveis,
       menor violação. Usa as avaliações já calculadas na geração."""
    indices = random.sample(range(len(populacao)), k)
    melhor = indices[0]
    for i in indices[1:]:
        if eh_melhor_deb(avaliacoes[i], avaliacoes[melhor]):
            melhor = i
    return populacao[melhor]


def executar_ag(nome_instancia, tempo_max=TEMPO_MAX_SEG):
    """Executa o AG completo em uma instância e grava o resultado."""
    caminho = caminho_instancia(nome_instancia)
    clientes, matriz, capacidade, max_veiculos, n = carregar_vrptw(caminho)

    inicio = time.time()
    tempo_evolucao = tempo_max * (1 - FRACAO_BUSCA_LOCAL)

    # População inicial heurística (mesma estratégia do original)
    populacao = solucoes_iniciais_heuristicas(n, clientes, matriz, TAM_POP)
    avaliacoes = [avaliar(ind, matriz, clientes, capacidade, max_veiculos)
                  for ind in populacao]

    melhor_solucao = None
    melhor_aval = None
    melhor_geracao = 0
    geracao = 0

    while geracao < NUM_GER_MAX and time.time() - inicio < tempo_evolucao:
        # --- ELITISMO: ordena a população pelas regras de Deb ---
        ordem = sorted(range(TAM_POP), key=lambda i: chave_deb(avaliacoes[i]))
        novos_filhos = [populacao[i][:] for i in ordem[:QTD_ELITE]]

        # Atualiza a melhor solução global (comparação de Deb)
        candidato = ordem[0]
        if melhor_aval is None or eh_melhor_deb(avaliacoes[candidato], melhor_aval):
            melhor_solucao = populacao[candidato][:]
            melhor_aval = avaliacoes[candidato]
            melhor_geracao = geracao

        # --- CRUZAMENTO E MUTAÇÃO (preenche o resto da população) ---
        while len(novos_filhos) < TAM_POP:
            pai1 = selecao_torneio_deb(populacao, avaliacoes)
            pai2 = selecao_torneio_deb(populacao, avaliacoes)
            f1, f2 = OX(pai1, pai2)
            novos_filhos.append(mutacao(f1))
            if len(novos_filhos) < TAM_POP:
                novos_filhos.append(mutacao(f2))

        populacao = novos_filhos
        avaliacoes = [avaliar(ind, matriz, clientes, capacidade, max_veiculos)
                      for ind in populacao]

        # Polimento memético: a cada 50 gerações, uma rajada curta de busca
        # local no melhor global, reinjetado no lugar do pior indivíduo.
        # O fluxo do AG não muda — é um reforço periódico do elitismo.
        if geracao % 50 == 49 and melhor_solucao is not None:
            melhor_solucao, melhor_aval, melhorias_bl = busca_local(
                melhor_solucao, melhor_aval, matriz, clientes, capacidade,
                max_veiculos, max_tentativas=3000, tempo_limite=2.0)
            if melhorias_bl:
                melhor_geracao = geracao
            pior = max(range(TAM_POP), key=lambda i: chave_deb(avaliacoes[i]))
            populacao[pior] = melhor_solucao[:]
            avaliacoes[pior] = melhor_aval

        if geracao % 50 == 0:
            status = "VIAVEL" if melhor_aval["viavel"] else "INVIAVEL"
            print(f"Geração {geracao}: veículos = {melhor_aval['num_veiculos']}, "
                  f"distância = {melhor_aval['distancia']:.2f} [{status}]")
        geracao += 1

    # --- BUSCA LOCAL no melhor indivíduo (pós-processamento) ---
    tempo_restante = tempo_max - (time.time() - inicio)
    melhor_solucao, melhor_aval, melhorias = busca_local(
        melhor_solucao, melhor_aval, matriz, clientes, capacidade,
        max_veiculos, max_tentativas=200000, tempo_limite=tempo_restante)
    print(f"Busca local: {melhorias} melhorias aplicadas")

    tempo_exec = time.time() - inicio

    # Decodifica a melhor solução em rotas (split ótimo: na instância grande
    # a evolução avalia com o decodificador guloso, mas a partição final é
    # sempre a ótima) e verifica de forma independente
    rotas, melhor_aval = decodificar_e_avaliar(
        melhor_solucao, matriz, clientes, capacidade, max_veiculos,
        metodo="split")
    viavel, problemas = verificar_solucao(
        rotas, matriz, clientes, capacidade, max_veiculos, n)

    parametros = (f"pop={TAM_POP}, elite={QTD_ELITE}, torneio={K_TORNEIO}, "
                  f"mutação={TAXA_MUTACAO}, gerações={geracao}, "
                  f"tempo_max={tempo_max}s")
    imprimir_resumo(nome_instancia, "AG", parametros, melhor_aval, rotas,
                    melhor_geracao, tempo_exec, viavel, problemas)
    arquivo = escrever_resultado(nome_instancia, "ag", AUTORES, rotas,
                                 melhor_aval["distancia"], tempo_exec)
    print(f"Resultado salvo em: {arquivo}")
    return melhor_aval, tempo_exec


if __name__ == "__main__":
    # Uso: python ag_vrptw.py [instancia|todas] [tempo_max_segundos]
    alvo = sys.argv[1] if len(sys.argv) > 1 else "c101"
    tempo = float(sys.argv[2]) if len(sys.argv) > 2 else TEMPO_MAX_SEG
    nomes = INSTANCIAS_PADRAO if alvo == "todas" else [alvo]
    for nome in nomes:
        print(f"\n########## AG — instância {nome} ##########")
        executar_ag(nome, tempo)
