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
TAM_ENXAME = 60
NUM_ITE_MAX = 1000000
TEMPO_MAX_SEG = 120
C1 = 2.05
C2 = 2.05
W_INICIAL, W_FINAL = 0.9, 0.4
VEL_MAX = 0.25
FRACAO_SEMENTES = 0.5
FRACAO_BUSCA_LOCAL = 0.0
PACIENCIA_GB = 100
FRACAO_REINICIO = 0.40


def atualiza_pos(particula, pb, gb, w, v0, c1=C1, c2=C2):
    """v = w*v0 + c1*r1*(pb - x) + c2*r2*(gb - x);  x = x + v."""
    r1 = np.random.uniform(0.0, 1.0, size=len(particula))
    r2 = np.random.uniform(0.0, 1.0, size=len(particula))
    v1 = w * v0 + c1 * r1 * (pb - particula) + c2 * r2 * (gb - particula)
    v1 = np.clip(v1, -VEL_MAX, VEL_MAX)
    nova_pos = np.clip(particula + v1, 0.0, 1.0)
    return nova_pos, v1


def avaliar_particula(particula, matriz, clientes, capacidade, max_veiculos):
    perm = chaves_para_permutacao(particula)
    return avaliar(perm, matriz, clientes, capacidade, max_veiculos)


def executar_pso(nome_instancia, tempo_max=TEMPO_MAX_SEG):
    caminho = caminho_instancia(nome_instancia)
    clientes, matriz, capacidade, max_veiculos, n = carregar_vrptw(caminho)
    dim = n - 1  # uma chave por cliente

    inicio = time.time()
    tempo_evolucao = tempo_max * (1 - FRACAO_BUSCA_LOCAL)

    qtd_sementes = int(TAM_ENXAME * FRACAO_SEMENTES)
    sementes = solucoes_iniciais_heuristicas(n, clientes, matriz, qtd_sementes)
    enxame = [permutacao_para_chaves(p) for p in sementes]
    while len(enxame) < TAM_ENXAME:
        enxame.append(np.random.uniform(0.0, 1.0, size=dim))

    velocidades = [np.random.uniform(-VEL_MAX, VEL_MAX, size=dim)
                   for _ in range(TAM_ENXAME)]

    avaliacoes = [avaliar_particula(p, matriz, clientes, capacidade, max_veiculos)
                  for p in enxame]

    pb = [p.copy() for p in enxame]
    pb_aval = list(avaliacoes)

    idx_gb = 0
    for i in range(1, TAM_ENXAME):
        if eh_melhor_deb(pb_aval[i], pb_aval[idx_gb]):
            idx_gb = i
    gb = pb[idx_gb].copy()
    gb_aval = pb_aval[idx_gb]
    melhor_iteracao = 0
    sem_melhora_gb = 0

    iteracao = 0
    while iteracao < NUM_ITE_MAX and time.time() - inicio < tempo_evolucao:
        # inércia decai com o tempo decorrido, não com iterações
        progresso = (time.time() - inicio) / tempo_evolucao
        w = W_INICIAL - (W_INICIAL - W_FINAL) * min(1.0, progresso)

        melhorou_gb = False
        for i in range(TAM_ENXAME):
            nova_pos, nova_vel = atualiza_pos(enxame[i], pb[i], gb, w,
                                              velocidades[i])
            enxame[i] = nova_pos
            velocidades[i] = nova_vel
            aval = avaliar_particula(nova_pos, matriz, clientes, capacidade,
                                     max_veiculos)

            if eh_melhor_deb(aval, pb_aval[i]):
                pb[i] = nova_pos.copy()
                pb_aval[i] = aval
                if eh_melhor_deb(aval, gb_aval):
                    gb = nova_pos.copy()
                    gb_aval = aval
                    melhor_iteracao = iteracao
                    melhorou_gb = True

        # Sem melhora por PACIENCIA_GB iterações: reinicia os piores para
        # injetar diversidade sem descartar soluções já encontradas.
        if melhorou_gb:
            sem_melhora_gb = 0
        else:
            sem_melhora_gb += 1
            if sem_melhora_gb >= PACIENCIA_GB:
                qtd = int(TAM_ENXAME * FRACAO_REINICIO)
                piores = sorted(range(TAM_ENXAME),
                                key=lambda i: chave_deb(pb_aval[i]),
                                reverse=True)[:qtd]
                for i in piores:
                    enxame[i] = np.random.uniform(0.0, 1.0, size=dim)
                    velocidades[i] = np.random.uniform(-VEL_MAX, VEL_MAX, size=dim)
                    aval_novo = avaliar_particula(enxame[i], matriz, clientes,
                                                  capacidade, max_veiculos)
                    pb[i] = enxame[i].copy()
                    pb_aval[i] = aval_novo
                sem_melhora_gb = 0

        if iteracao % 20 == 0:
            status = "VIAVEL" if gb_aval["viavel"] else "INVIAVEL"
            print(f"Iteração {iteracao}: veículos = {gb_aval['num_veiculos']}, "
                  f"distância = {gb_aval['distancia']:.2f} [{status}]")
        iteracao += 1

    melhor_perm = chaves_para_permutacao(gb)

    tempo_exec = time.time() - inicio

    rotas, gb_aval = decodificar_e_avaliar(
        melhor_perm, matriz, clientes, capacidade, max_veiculos,
        metodo="split")
    viavel, problemas = verificar_solucao(
        rotas, matriz, clientes, capacidade, max_veiculos, n)

    parametros = (f"enxame={TAM_ENXAME}, c1={C1}, c2={C2}, "
                  f"w={W_INICIAL}->{W_FINAL}, vel_max={VEL_MAX}, "
                  f"paciencia={PACIENCIA_GB}, reinicio={FRACAO_REINICIO}, "
                  f"iterações={iteracao}, tempo_max={tempo_max}s")
    imprimir_resumo(nome_instancia, "PSO", parametros, gb_aval, rotas,
                    melhor_iteracao, tempo_exec, viavel, problemas)
    arquivo = escrever_resultado(nome_instancia, "pso", AUTORES, rotas,
                                 gb_aval["distancia"], tempo_exec)
    print(f"Resultado salvo em: {arquivo}")
    return gb_aval, tempo_exec


if __name__ == "__main__":
    # Uso: python pso_vrptw_sbl.py [instancia|todas] [tempo_max_segundos]
    alvo = sys.argv[1] if len(sys.argv) > 1 else "c101"
    tempo = float(sys.argv[2]) if len(sys.argv) > 2 else TEMPO_MAX_SEG
    nomes = INSTANCIAS_PADRAO if alvo == "todas" else [alvo]
    for nome in nomes:
        print(f"\n########## PSO — instância {nome} ##########")
        executar_pso(nome, tempo)
