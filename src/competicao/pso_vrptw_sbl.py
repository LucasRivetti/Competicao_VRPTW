"""
pso_vrptw.py — Particle Swarm Optimization adaptado para o VRPTW.

ORIGEM: src/enxame/pso.py (PSO contínuo clássico com inércia decrescente,
melhor pessoal pb e melhor global gb).

O QUE FOI MANTIDO do algoritmo original:
  - a equação de atualização de velocidade/posição ('atualiza_pos'):
        v = w*v0 + c1*r1*(pb - x) + c2*r2*(gb - x);  x = x + v
    com os mesmos c1 = c2 = 2.05;
  - a inércia w decaindo linearmente de 0.9 a 0.4 ao longo da execução;
  - a estrutura pb (melhor pessoal) / gb (melhor global);
  - o laço principal: atualizar cada partícula, avaliar, atualizar pb/gb.

O QUE FOI ADAPTADO (e por quê):
  - REPRESENTAÇÃO POR CHAVES ALEATÓRIAS (random keys): o PSO é um algoritmo
    contínuo e o VRPTW é combinatório. Cada partícula é um vetor real com
    uma dimensão por cliente; a permutação (tour gigante) é obtida ordenando
    as chaves (argsort). Isso permite manter as equações do PSO INTACTAS —
    só a decodificação muda. O tour é decodificado em rotas viáveis pelo
    decodificador comum (capacidade + janelas de tempo);
  - no original cada variável (x1, x2) era atualizada separadamente com seus
    próprios sorteios r1/r2; aqui a mesma fórmula é aplicada por dimensão
    de forma vetorizada (numpy), com r1/r2 sorteados por dimensão;
  - comparações de pb/gb passam a usar as REGRAS DE DEB (viável > inviável)
    em vez de comparar apenas f(x), porque o problema agora tem restrições;
  - posição limitada a [0,1] e velocidade limitada (clamp), como o original
    limitava a posição ao domínio [-10,10];
  - parte do enxame é semeada com as soluções heurísticas convertidas em
    chaves (população inicial mais adequada, permitido no enunciado);
  - critério de parada por tempo + busca local final na melhor solução;
  - saída no formato exigido pela competição.
"""

import os
import random
import sys
import time

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from competicao.vrptw_comum import (INSTANCIAS_PADRAO, avaliar, busca_local,
                         caminho_instancia, carregar_vrptw,
                         chaves_para_permutacao, decodificar_e_avaliar,
                         eh_melhor_deb, escrever_resultado, imprimir_resumo,
                         permutacao_para_chaves,
                         solucoes_iniciais_heuristicas, verificar_solucao)

# --------------------------- Parâmetros do PSO ------------------------------
AUTORES = "Lucas Rivetti, Ian Nunes"        # ajustar com os nomes do grupo
TAM_ENXAME = 60                  # número de partículas
NUM_ITE_MAX = 1000000            # teto de iterações (para por tempo)
TEMPO_MAX_SEG = 120              # tempo máximo por instância
C1 = 2.05                        # mesmo coeficiente cognitivo do original
C2 = 2.05                        # mesmo coeficiente social do original
W_INICIAL, W_FINAL = 0.9, 0.4    # mesma faixa de inércia do original
VEL_MAX = 0.25                   # limite de velocidade no espaço das chaves
FRACAO_SEMENTES = 0.5            # fração do enxame semeada com heurísticas
FRACAO_BUSCA_LOCAL = 0.25        # fração final do tempo p/ busca local


def atualiza_pos(particula, pb, gb, w, v0, c1=C1, c2=C2):
    """Mesma equação do PSO original, aplicada por dimensão (vetorizada):
       v1 = w*v0 + c1*r1*(pb - x) + c2*r2*(gb - x);  nova_pos = x + v1."""
    r1 = np.random.uniform(0.0, 1.0, size=len(particula))
    r2 = np.random.uniform(0.0, 1.0, size=len(particula))
    v1 = w * v0 + c1 * r1 * (pb - particula) + c2 * r2 * (gb - particula)
    v1 = np.clip(v1, -VEL_MAX, VEL_MAX)        # limita a velocidade
    nova_pos = np.clip(particula + v1, 0.0, 1.0)  # mantém as chaves em [0,1]
    return nova_pos, v1


def avaliar_particula(particula, matriz, clientes, capacidade, max_veiculos):
    """Decodifica as chaves em permutação e avalia com a função comum."""
    perm = chaves_para_permutacao(particula)
    return avaliar(perm, matriz, clientes, capacidade, max_veiculos)


def executar_pso(nome_instancia, tempo_max=TEMPO_MAX_SEG):
    """Executa o PSO completo em uma instância e grava o resultado."""
    caminho = caminho_instancia(nome_instancia)
    clientes, matriz, capacidade, max_veiculos, n = carregar_vrptw(caminho)
    dim = n - 1                       # uma chave por cliente

    inicio = time.time()
    tempo_evolucao = tempo_max * (1 - FRACAO_BUSCA_LOCAL)

    # --- Inicialização do enxame ---
    # Parte das partículas nasce das soluções heurísticas (convertidas em
    # chaves); o resto nasce aleatório em [0,1], como no PSO original.
    qtd_sementes = int(TAM_ENXAME * FRACAO_SEMENTES)
    sementes = solucoes_iniciais_heuristicas(n, clientes, matriz, qtd_sementes)
    enxame = [permutacao_para_chaves(p) for p in sementes]
    while len(enxame) < TAM_ENXAME:
        enxame.append(np.random.uniform(0.0, 1.0, size=dim))

    velocidades = [np.random.uniform(-VEL_MAX, VEL_MAX, size=dim)
                   for _ in range(TAM_ENXAME)]

    avaliacoes = [avaliar_particula(p, matriz, clientes, capacidade, max_veiculos)
                  for p in enxame]

    # pb = melhor posição pessoal; gb = melhor posição global (como no original)
    pb = [p.copy() for p in enxame]
    pb_aval = list(avaliacoes)

    idx_gb = 0
    for i in range(1, TAM_ENXAME):
        if eh_melhor_deb(pb_aval[i], pb_aval[idx_gb]):
            idx_gb = i
    gb = pb[idx_gb].copy()
    gb_aval = pb_aval[idx_gb]
    melhor_iteracao = 0

    iteracao = 0
    while iteracao < NUM_ITE_MAX: # and time.time() - inicio < tempo_evolucao:
        # Inércia decai linearmente com o TEMPO decorrido (mesma ideia do
        # original, que decaía com as iterações; aqui a parada é por tempo).
        progresso = (time.time() - inicio) / tempo_evolucao
        w = W_INICIAL - (W_INICIAL - W_FINAL) * min(1.0, progresso)

        for i in range(TAM_ENXAME):
            nova_pos, nova_vel = atualiza_pos(enxame[i], pb[i], gb, w,
                                              velocidades[i])
            enxame[i] = nova_pos
            velocidades[i] = nova_vel
            aval = avaliar_particula(nova_pos, matriz, clientes, capacidade,
                                     max_veiculos)

            # Atualização do melhor pessoal pelas regras de Deb
            if eh_melhor_deb(aval, pb_aval[i]):
                pb[i] = nova_pos.copy()
                pb_aval[i] = aval
                # Atualização do melhor global pelas regras de Deb
                if eh_melhor_deb(aval, gb_aval):
                    gb = nova_pos.copy()
                    gb_aval = aval
                    melhor_iteracao = iteracao

        # Polimento memético: a cada 25 iterações, rajada curta de busca
        # local no melhor global; se melhorar, o gb (convertido de volta
        # para chaves) passa a atrair o enxame para a solução polida.
        # if iteracao % 25 == 24:
        #     perm_gb = chaves_para_permutacao(gb)
        #     perm_gb, aval_bl, _ = busca_local(
        #         perm_gb, gb_aval, matriz, clientes, capacidade, max_veiculos,
        #         max_tentativas=3000, tempo_limite=2.0)
        #     if eh_melhor_deb(aval_bl, gb_aval):
        #         gb = permutacao_para_chaves(perm_gb)
        #         gb_aval = aval_bl
        #         melhor_iteracao = iteracao

        if iteracao % 20 == 0:
            status = "VIAVEL" if gb_aval["viavel"] else "INVIAVEL"
            print(f"Iteração {iteracao}: veículos = {gb_aval['num_veiculos']}, "
                  f"distância = {gb_aval['distancia']:.2f} [{status}]")
        iteracao += 1

    # --- BUSCA LOCAL na melhor solução (pós-processamento) ---
    # melhor_perm = chaves_para_permutacao(gb)
    # tempo_restante = tempo_max - (time.time() - inicio)
    # melhor_perm, gb_aval, melhorias = busca_local(
    #     melhor_perm, gb_aval, matriz, clientes, capacidade, max_veiculos,
    #     max_tentativas=200000, tempo_limite=tempo_restante)
    # print(f"Busca local: {melhorias} melhorias aplicadas")

    tempo_exec = time.time() - inicio

    # Partição final sempre pelo split ótimo (a evolução pode usar o guloso)
    rotas, gb_aval = decodificar_e_avaliar(
        melhor_perm, matriz, clientes, capacidade, max_veiculos,
        metodo="split")
    viavel, problemas = verificar_solucao(
        rotas, matriz, clientes, capacidade, max_veiculos, n)

    parametros = (f"enxame={TAM_ENXAME}, c1={C1}, c2={C2}, "
                  f"w={W_INICIAL}->{W_FINAL}, vel_max={VEL_MAX}, "
                  f"iterações={iteracao}, tempo_max={tempo_max}s")
    imprimir_resumo(nome_instancia, "PSO", parametros, gb_aval, rotas,
                    melhor_iteracao, tempo_exec, viavel, problemas)
    arquivo = escrever_resultado(nome_instancia, "pso", AUTORES, rotas,
                                 gb_aval["distancia"], tempo_exec)
    print(f"Resultado salvo em: {arquivo}")
    return gb_aval, tempo_exec


if __name__ == "__main__":
    # Uso: python pso_vrptw.py [instancia|todas] [tempo_max_segundos]
    alvo = sys.argv[1] if len(sys.argv) > 1 else "c101"
    tempo = float(sys.argv[2]) if len(sys.argv) > 2 else TEMPO_MAX_SEG
    nomes = INSTANCIAS_PADRAO if alvo == "todas" else [alvo]
    for nome in nomes:
        print(f"\n########## PSO — instância {nome} ##########")
        executar_pso(nome, tempo)
