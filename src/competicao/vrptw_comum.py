import math
import os
import random
import time

import numpy as np

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

INSTANCIAS_PADRAO = ["c101", "c1_2_1", "r209", "rc208", "rc2_4_9"]

# maior que qualquer distância total possível nas instâncias (até 400 clientes)
PESO_VEICULO = 100000.0


def caminho_instancia(nome):
    if os.path.isfile(nome):
        return nome
    return os.path.join(RAIZ, "dados", "vrptw", nome + ".txt")


def carregar_vrptw(caminho_arquivo):
    """Lê instância no formato Solomon/Gehring-Homberger."""
    clientes = {}
    capacidade = 0
    max_veiculos = 0

    with open(caminho_arquivo, "r") as f:
        linhas = f.readlines()

    for linha in linhas:
        partes = linha.split()

        if len(partes) == 2 and partes[0].isdigit() and partes[1].isdigit():
            max_veiculos = int(partes[0])
            capacidade = int(partes[1])

        elif len(partes) >= 7:
            try:
                valores = list(map(float, partes[:7]))
            except ValueError:
                continue
            id_no = int(valores[0])
            clientes[id_no] = {
                "x": valores[1],
                "y": valores[2],
                "demanda": valores[3],
                "ready_time": valores[4],
                "due_date": valores[5],
                "service_time": valores[6],
            }

    n = len(clientes)

    coords = np.array([[clientes[i]["x"], clientes[i]["y"]] for i in range(n)])
    dif = coords[:, None, :] - coords[None, :, :]
    matriz = np.sqrt((dif ** 2).sum(axis=2)).tolist()

    return clientes, matriz, capacidade, max_veiculos, n


# Cache de listas planas — acesso em lista é mais rápido que dict-de-dicts
# no laço interno do decodificador (ponto mais quente do código).
_cache_dados = []


def _dados_clientes(clientes):
    for ref, dados in _cache_dados:
        if ref is clientes:
            return dados
    n = len(clientes)
    dados = (
        [clientes[i]["demanda"] for i in range(n)],
        [clientes[i]["ready_time"] for i in range(n)],
        [clientes[i]["due_date"] for i in range(n)],
        [clientes[i]["service_time"] for i in range(n)],
    )
    _cache_dados.append((clientes, dados))
    if len(_cache_dados) > 10:
        _cache_dados.pop(0)
    return dados


# Acima deste tamanho usa guloso durante a evolução: split custa O(N x rota)
# e reduz gerações o suficiente para piorar o resultado final.
LIMIAR_SPLIT = 250


def _decodificar_guloso(solucao, matriz, clientes, capacidade, max_veiculos):
    demanda, ready, due, service = _dados_clientes(clientes)
    rotas = []
    rota_atual = []
    carga = 0.0
    tempo = 0.0
    pos = 0

    for c in solucao:
        dist = matriz[pos][c]
        chegada = tempo + dist
        if chegada < ready[c]:
            chegada = ready[c]  # espera permitida

        if rota_atual and (carga + demanda[c] > capacidade
                           or chegada > due[c]):
            rotas.append(rota_atual)
            rota_atual = []
            pos = 0
            carga = 0.0
            tempo = 0.0
            chegada = matriz[0][c]
            if chegada < ready[c]:
                chegada = ready[c]

        rota_atual.append(c)
        carga += demanda[c]
        tempo = chegada + service[c]
        pos = c

    if rota_atual:
        rotas.append(rota_atual)

    aval = avaliar_rotas(rotas, matriz, clientes, capacidade, max_veiculos)
    return rotas, aval


def decodificar_e_avaliar(solucao, matriz, clientes, capacidade, max_veiculos,
                          metodo=None):
    """
    Split ótimo (programação dinâmica) do tour gigante em rotas viáveis.
    metodo="split"|"guloso"|None (automático pelo LIMIAR_SPLIT).
    A viabilidade é monotônica: se i..j viola, i..j' (j'>j) também viola —
    o laço interno para cedo mantendo custo próximo de O(N x tamanho de rota).
    """
    if metodo is None:
        metodo = "split" if len(solucao) <= LIMIAR_SPLIT else "guloso"
    if metodo == "guloso":
        return _decodificar_guloso(solucao, matriz, clientes, capacidade,
                                   max_veiculos)

    demanda, ready, due, service = _dados_clientes(clientes)
    n = len(solucao)
    horizonte = due[0]
    INF = float("inf")
    custo = [(INF, INF, INF)] * (n + 1)
    custo[0] = (0.0, 0, 0.0)
    pred = [0] * (n + 1)

    matriz_0 = matriz[0]
    for i in range(n):
        ci = custo[i]
        if ci[0] == INF:
            continue
        carga = 0.0
        tempo = 0.0
        dist_rota = 0.0
        pos = 0
        j = i
        while j < n:
            c = solucao[j]
            dist_seg = matriz[pos][c]
            chegada = tempo + dist_seg
            if chegada < ready[c]:
                chegada = ready[c]
            carga += demanda[c]

            estourou = carga > capacidade or chegada > due[c]
            if j > i and estourou:
                break

            viol = 0.0
            if estourou:
                if chegada > due[c]:
                    viol += chegada - due[c]
                if carga > capacidade:
                    viol += carga - capacidade

            dist_rota += dist_seg
            tempo = chegada + service[c]
            pos = c

            retorno = tempo + matriz[pos][0]
            if retorno > horizonte:
                if j > i:
                    break
                viol += retorno - horizonte

            cand = (ci[0] + viol, ci[1] + 1, ci[2] + dist_rota + matriz[pos][0])
            if cand < custo[j + 1]:
                custo[j + 1] = cand
                pred[j + 1] = i
            if viol > 0:
                break
            j += 1

    rotas = []
    fim = n
    while fim > 0:
        inicio_rota = pred[fim]
        rotas.append(list(solucao[inicio_rota:fim]))
        fim = inicio_rota
    rotas.reverse()

    aval = avaliar_rotas(rotas, matriz, clientes, capacidade, max_veiculos)
    return rotas, aval


def avaliar_rotas(rotas, matriz, clientes, capacidade, max_veiculos):
    demanda, ready, due, service = _dados_clientes(clientes)
    horizonte = due[0]

    distancia_total = 0.0
    viol_capacidade = 0.0
    viol_janela = 0.0

    for rota in rotas:
        carga = sum(demanda[c] for c in rota)
        if carga > capacidade:
            viol_capacidade += carga - capacidade
        tempo = 0.0
        pos = 0
        for c in rota:
            dist_seg = matriz[pos][c]
            distancia_total += dist_seg
            chegada = tempo + dist_seg
            if chegada < ready[c]:
                chegada = ready[c]
            if chegada > due[c]:
                viol_janela += chegada - due[c]
            tempo = chegada + service[c]
            pos = c
        distancia_total += matriz[pos][0]
        retorno = tempo + matriz[pos][0]
        if retorno > horizonte:
            viol_janela += retorno - horizonte

    num_veiculos = len(rotas)
    viol_veiculos = max(0, num_veiculos - max_veiculos)
    violacao = viol_veiculos + viol_janela + viol_capacidade

    return {
        "objetivo": num_veiculos * PESO_VEICULO + distancia_total,
        "distancia": distancia_total,
        "num_veiculos": num_veiculos,
        "viol_capacidade": viol_capacidade,
        "viol_janela": viol_janela,
        "viol_veiculos": viol_veiculos,
        "violacao": violacao,
        "viavel": violacao == 0,
    }


def avaliar(solucao, matriz, clientes, capacidade, max_veiculos):
    return decodificar_e_avaliar(solucao, matriz, clientes, capacidade, max_veiculos)[1]


def eh_melhor_deb(a, b):
    """Regras de Deb: viável > inviável; entre iguais, menor objetivo/violação."""
    if a["violacao"] == 0 and b["violacao"] == 0:
        return a["objetivo"] < b["objetivo"]
    if a["violacao"] == 0:
        return True
    if b["violacao"] == 0:
        return False
    return a["violacao"] < b["violacao"]


def chave_deb(aval):
    if aval["violacao"] == 0:
        return (0, aval["objetivo"], 0.0)
    return (1, aval["violacao"], aval["objetivo"])


def solucoes_iniciais_heuristicas(n, clientes, matriz, quantidade):
    solucoes = []

    solucao_due = sorted(range(1, n), key=lambda c: clientes[c]["due_date"])
    solucoes.append(solucao_due)
    solucao_ready = sorted(range(1, n), key=lambda c: clientes[c]["ready_time"])
    solucoes.append(solucao_ready)

    for _ in range(max(1, int(quantidade * 0.2))):
        mutante = solucao_due[:]
        for _ in range(5):
            i, j = random.sample(range(len(mutante)), 2)
            mutante[i], mutante[j] = mutante[j], mutante[i]
        solucoes.append(mutante)

    for _ in range(max(1, int(quantidade * 0.3))):
        nao_visitados = list(range(1, n))
        atual = random.choice(nao_visitados)
        rota = [atual]
        nao_visitados.remove(atual)
        while nao_visitados:
            prox = min(nao_visitados, key=lambda c: matriz[atual][c])
            rota.append(prox)
            nao_visitados.remove(prox)
            atual = prox
        solucoes.append(rota)

    for _ in range(max(1, int(quantidade * 0.3))):
        nao_visitados = list(range(1, n))
        atual = random.choice(nao_visitados)
        rota = [atual]
        nao_visitados.remove(atual)
        while nao_visitados:
            prox = min(nao_visitados,
                       key=lambda c: matriz[atual][c] * clientes[c]["due_date"])
            rota.append(prox)
            nao_visitados.remove(prox)
            atual = prox
        solucoes.append(rota)

    while len(solucoes) < quantidade:
        aleatoria = list(range(1, n))
        random.shuffle(aleatoria)
        solucoes.append(aleatoria)

    return solucoes[:quantidade]


def chaves_para_permutacao(chaves):
    return [int(i) + 1 for i in np.argsort(chaves, kind="stable")]


def permutacao_para_chaves(perm):
    n = len(perm)
    chaves = np.empty(n)
    for posicao, cliente in enumerate(perm):
        chaves[cliente - 1] = (posicao + 0.5) / n
    return chaves


def busca_local(solucao, aval, matriz, clientes, capacidade, max_veiculos,
                max_tentativas=20000, tempo_limite=None):
    """Subida de encosta estocástica: swap, or-opt 1, or-opt 2-3, 2-opt."""
    melhor = list(solucao)
    melhor_aval = aval
    n = len(melhor)
    inicio = time.time()
    melhorias = 0

    for _ in range(max_tentativas):
        if tempo_limite is not None and time.time() - inicio > tempo_limite:
            break
        vizinho = melhor[:]
        i, j = sorted(random.sample(range(n), 2))
        sorteio = random.random()
        if sorteio < 0.25:
            vizinho[i], vizinho[j] = vizinho[j], vizinho[i]
        elif sorteio < 0.50:
            cliente = vizinho.pop(i)
            vizinho.insert(j, cliente)
        elif sorteio < 0.75:
            tam_seg = random.randint(2, 3)
            i = random.randint(0, n - tam_seg)
            segmento = vizinho[i:i + tam_seg]
            del vizinho[i:i + tam_seg]
            destino = random.randint(0, len(vizinho))
            vizinho[destino:destino] = segmento
        else:
            vizinho[i:j + 1] = reversed(vizinho[i:j + 1])

        aval_vizinho = avaliar(vizinho, matriz, clientes, capacidade, max_veiculos)
        if eh_melhor_deb(aval_vizinho, melhor_aval):
            melhor = vizinho
            melhor_aval = aval_vizinho
            melhorias += 1

    return melhor, melhor_aval, melhorias


def verificar_solucao(rotas, matriz, clientes, capacidade, max_veiculos, n):
    """Verificação independente do decodificador para a solução final."""
    problemas = []

    atendidos = [c for rota in rotas for c in rota]
    if len(atendidos) != len(set(atendidos)):
        problemas.append("ha clientes repetidos nas rotas")
    faltando = set(range(1, n)) - set(atendidos)
    if faltando:
        problemas.append(f"clientes nao atendidos: {sorted(faltando)}")

    if len(rotas) > max_veiculos:
        problemas.append(f"frota excedida: {len(rotas)} > {max_veiculos}")

    for k, rota in enumerate(rotas, 1):
        carga = sum(clientes[c]["demanda"] for c in rota)
        if carga > capacidade:
            problemas.append(f"rota {k}: capacidade excedida ({carga} > {capacidade})")
        tempo = 0.0
        pos = 0
        for c in rota:
            chegada = tempo + matriz[pos][c]
            if chegada < clientes[c]["ready_time"]:
                chegada = clientes[c]["ready_time"]
            if chegada > clientes[c]["due_date"] + 1e-9:
                problemas.append(
                    f"rota {k}: janela violada no cliente {c} "
                    f"(chegada {chegada:.2f} > due {clientes[c]['due_date']:.2f})")
            tempo = chegada + clientes[c]["service_time"]
            pos = c
        retorno = tempo + matriz[pos][0]
        if retorno > clientes[0]["due_date"] + 1e-9:
            problemas.append(f"rota {k}: retorno ao deposito apos o horizonte")

    return len(problemas) == 0, problemas


def calcular_distancia_rotas(rotas, matriz):
    total = 0.0
    for rota in rotas:
        pos = 0
        for c in rota:
            total += matriz[pos][c]
            pos = c
        total += matriz[pos][0]
    return total


def _ler_resultado_existente(caminho):
    if not os.path.isfile(caminho):
        return None
    veiculos = distancia = None
    try:
        with open(caminho, "r", encoding="utf-8") as f:
            for linha in f:
                if "veículos:" in linha:
                    veiculos = int(linha.split(":")[1])
                elif "Distância total:" in linha:
                    distancia = float(linha.split(":")[1])
    except (ValueError, OSError):
        return None
    if veiculos is None or distancia is None:
        return None
    return veiculos, distancia


def escrever_resultado(nome_instancia, algoritmo, autores, rotas, distancia,
                       tempo_exec):
    """Não sobrescreve se o arquivo existente já contiver solução melhor."""
    pasta = os.path.join(RAIZ, "dados", "resultados")
    os.makedirs(pasta, exist_ok=True)
    caminho = os.path.join(
        pasta, f"{nome_instancia}_resultado_{algoritmo.lower()}.txt")

    existente = _ler_resultado_existente(caminho)
    if existente is not None:
        veic_ant, dist_ant = existente
        if (veic_ant, dist_ant) <= (len(rotas), round(distancia, 4)):
            print(f"Resultado anterior mantido (era melhor ou igual): "
                  f"{veic_ant} veículos / {dist_ant:.4f}")
            return caminho

    linhas = [
        f"======== MELHOR SOLUÇÃO {algoritmo.upper()} ========",
        f"Nome da instância: {nome_instancia}",
        f"Autores: {autores}",
        f"Número de veículos: {len(rotas)}",
        f"Distância total: {distancia:.4f}",
        f"Tempo total: {tempo_exec:.4f}s",
        "Rotas:",
    ]
    for k, rota in enumerate(rotas, 1):
        caminho_rota = " -> ".join(["0"] + [str(c) for c in rota] + ["0"])
        linhas.append(f"Rota {k}: {caminho_rota}")

    with open(caminho, "w", encoding="utf-8") as f:
        f.write("\n".join(linhas) + "\n")

    return caminho


def imprimir_resumo(nome_instancia, algoritmo, parametros, aval, rotas,
                    melhor_iteracao, tempo_exec, viavel, problemas):
    print("\n" + "=" * 50)
    print(f"Instância: {nome_instancia} | Algoritmo: {algoritmo}")
    print(f"Parâmetros: {parametros}")
    print(f"Número de veículos: {aval['num_veiculos']}")
    print(f"Distância total: {aval['distancia']:.4f}")
    print(f"Função de avaliação (veic*{PESO_VEICULO:.0f} + dist): {aval['objetivo']:.4f}")
    print(f"Violação de capacidade: {aval['viol_capacidade']:.4f}")
    print(f"Violação de janela de tempo: {aval['viol_janela']:.4f}")
    print(f"Violação de frota (veículos extras): {aval['viol_veiculos']}")
    print(f"Violação total: {aval['violacao']:.4f}")
    print(f"Melhor iteração/geração: {melhor_iteracao}")
    print(f"Tempo de execução: {tempo_exec:.4f}s")
    print(f"Solução viável (verificação independente): {'SIM' if viavel else 'NAO'}")
    for p in problemas:
        print(f"  PROBLEMA: {p}")
    print("Rotas:")
    for k, rota in enumerate(rotas, 1):
        print(f"  Rota {k}: " + " -> ".join(["0"] + [str(c) for c in rota] + ["0"]))
    print("=" * 50)
