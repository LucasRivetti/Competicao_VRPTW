"""
vrptw_comum.py — Funções comuns do VRPTW usadas pelos 5 algoritmos bioinspirados.

Este módulo concentra tudo o que NÃO faz parte da lógica de cada algoritmo:
    1. leitura das instâncias (formato Solomon / Gehring-Homberger);
    2. matriz de distâncias euclidianas;
    3. decodificação da solução (tour gigante -> rotas);
    4. avaliação (nº de veículos, distância, violações, viabilidade);
    5. comparação de soluções pelas regras de Deb;
    6. heurísticas de população inicial;
    7. busca local simples (pós-processamento, igual para todos);
    8. verificação independente da solução final;
    9. escrita do arquivo de resultados no formato exigido pela competição.

REPRESENTAÇÃO DA SOLUÇÃO (igual para todos os algoritmos)
----------------------------------------------------------
Uma solução é uma permutação dos clientes 1..N-1 (o "tour gigante").
O decodificador percorre a permutação na ordem e vai preenchendo veículos:
quando o próximo cliente violaria a capacidade OU a janela de tempo do
veículo atual, a rota é fechada (volta ao depósito) e um veículo novo é
aberto. Assim, TODA permutação é decodificada em rotas que respeitam
capacidade e janelas de tempo por construção — a única restrição que pode
sobrar violada é o limite de veículos da frota (e casos estruturais raros,
que são medidos e contados como violação).

FUNÇÃO OBJETIVO (prioridades da competição)
-------------------------------------------
    objetivo = num_veiculos * PESO_VEICULO + distancia_total

PESO_VEICULO é maior do que qualquer distância possível nas instâncias,
o que torna a comparação lexicográfica: primeiro minimiza veículos
(critério principal da competição), depois a distância (critério
secundário). As violações NÃO entram no objetivo: elas são tratadas
separadamente pelas REGRAS DE DEB (ver eh_melhor_deb).
"""

import math
import os
import random
import time

import numpy as np

# Raiz do projeto (um nível acima de src/)
RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Instâncias de treino fornecidas com o trabalho
INSTANCIAS_PADRAO = ["c101", "c1_2_1", "r209", "rc208", "rc2_4_9"]

# Peso lexicográfico do número de veículos no objetivo.
# Precisa ser maior que qualquer distância total possível: nas instâncias
# de até 400 clientes a pior distância fica na casa de dezenas de milhares.
PESO_VEICULO = 100000.0


# ---------------------------------------------------------------------------
# 1. LEITURA DA INSTÂNCIA
# ---------------------------------------------------------------------------
def caminho_instancia(nome):
    """Monta o caminho do arquivo da instância a partir do nome (ex.: 'c101')."""
    if os.path.isfile(nome):           # também aceita um caminho completo
        return nome
    return os.path.join(RAIZ, "dados", "vrptw", nome + ".txt")


def carregar_vrptw(caminho_arquivo):
    """
    Lê o arquivo da instância no formato Solomon/Gehring-Homberger.

    O arquivo tem uma linha 'NUMBER CAPACITY' (2 inteiros) com o tamanho da
    frota e a capacidade dos veículos, e depois uma linha por nó com 7 campos:
        id  x  y  demanda  inicio_janela  fim_janela  tempo_servico
    O nó 0 é o depósito (demanda 0, janela = horizonte de operação).

    Retorna: clientes (dict id -> dados), matriz de distâncias, capacidade,
             máximo de veículos e número total de nós (depósito incluído).
    """
    clientes = {}
    capacidade = 0
    max_veiculos = 0

    with open(caminho_arquivo, "r") as f:
        linhas = f.readlines()

    for linha in linhas:
        partes = linha.split()

        # Linha de configuração da frota: exatamente 2 números inteiros
        if len(partes) == 2 and partes[0].isdigit() and partes[1].isdigit():
            max_veiculos = int(partes[0])
            capacidade = int(partes[1])

        # Linha de cliente/depósito: 7 ou mais campos numéricos
        elif len(partes) >= 7:
            try:
                valores = list(map(float, partes[:7]))
            except ValueError:
                continue  # cabeçalhos de texto são ignorados
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

    # Matriz de distâncias euclidianas (exigência da competição), calculada
    # de forma vetorizada com numpy e convertida para listas, pois o acesso
    # elemento a elemento em listas puras é mais rápido nos laços do Python.
    coords = np.array([[clientes[i]["x"], clientes[i]["y"]] for i in range(n)])
    dif = coords[:, None, :] - coords[None, :, :]
    matriz = np.sqrt((dif ** 2).sum(axis=2)).tolist()

    return clientes, matriz, capacidade, max_veiculos, n


# ---------------------------------------------------------------------------
# 2. DECODIFICAÇÃO + AVALIAÇÃO (split ótimo do tour gigante)
# ---------------------------------------------------------------------------
# Cache dos dados dos clientes em listas planas (demanda, ready, due,
# service) — o decodificador é o ponto mais quente do código e listas são
# bem mais rápidas que dict-de-dicts dentro do laço. A referência ao próprio
# dict 'clientes' é guardada para o cache nunca confundir instâncias.
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
    if len(_cache_dados) > 10:          # nunca cresce além das instâncias
        _cache_dados.pop(0)
    return dados


# Acima deste número de clientes a avaliação usa o decodificador GULOSO
# (O(N)) durante a evolução: o split ótimo custa O(N x tamanho de rota) e,
# na instância de 400 clientes, derrubava o número de gerações a ponto de
# piorar o resultado final. O split continua sendo aplicado à solução final
# (metodo="split"), o que reduz veículos sem custo para a evolução.
LIMIAR_SPLIT = 250


def _decodificar_guloso(solucao, matriz, clientes, capacidade, max_veiculos):
    """
    Decodificador guloso O(N) (a versão original do ag_tsp.py): percorre o
    tour e fecha a rota na primeira violação de capacidade ou janela.
    Produz partições piores que o split ótimo, mas é ~10-30x mais barato —
    usado como avaliação durante a evolução nas instâncias grandes.
    """
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
            chegada = ready[c]              # espera permitida

        if rota_atual and (carga + demanda[c] > capacidade
                           or chegada > due[c]):
            rotas.append(rota_atual)        # fecha a rota e abre veículo novo
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
    Decodifica o tour gigante em rotas usando o SPLIT ÓTIMO (algoritmo de
    particionamento de Prins, adaptado às janelas de tempo do VRPTW).

    metodo: "split", "guloso" ou None (automático: split até LIMIAR_SPLIT
    clientes; guloso acima disso, pelo custo — ver comentário do limiar).

    Em vez de fechar a rota gulosamente na primeira violação (versão
    anterior, que fragmentava a frota), uma programação dinâmica escolhe a
    MELHOR partição do tour em rotas consecutivas:

        custo[k] = melhor custo para atender os k primeiros clientes do tour
        custo[j+1] = min sobre i <= j de  custo[i] + custo_rota(i..j)

    onde a rota i..j (servida na ordem do tour) só é considerada se for
    viável: capacidade, janelas de tempo (com espera permitida quando o
    veículo chega cedo) e retorno ao depósito dentro do horizonte. O custo
    é lexicográfico (violação, nº de veículos, distância) — exatamente as
    prioridades da competição. Clientes individualmente inviáveis (não
    existem nas instâncias benchmark) viram rotas unitárias com a violação
    medida, para que as regras de Deb possam penalizá-los.

    A viabilidade de uma rota é monotônica: se i..j viola capacidade ou
    janela, qualquer extensão i..j' (j' > j) também viola — por isso o laço
    interno pode parar cedo ('break'), mantendo o custo próximo de
    O(N x tamanho máximo de rota).

    Retorna (rotas, aval) — mesma interface da versão anterior, então os
    cinco algoritmos não precisam mudar.
    """
    if metodo is None:
        metodo = "split" if len(solucao) <= LIMIAR_SPLIT else "guloso"
    if metodo == "guloso":
        return _decodificar_guloso(solucao, matriz, clientes, capacidade,
                                   max_veiculos)

    demanda, ready, due, service = _dados_clientes(clientes)
    n = len(solucao)
    horizonte = due[0]                 # janela do depósito = fim da operação
    INF = float("inf")
    custo = [(INF, INF, INF)] * (n + 1)   # (violação, veículos, distância)
    custo[0] = (0.0, 0, 0.0)
    pred = [0] * (n + 1)               # início da última rota da partição

    matriz_0 = matriz[0]
    for i in range(n):
        ci = custo[i]
        if ci[0] == INF:
            continue
        # Tenta a rota que começa no cliente solucao[i] e vai esticando
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
                chegada = ready[c]      # chegou cedo: espera abrir a janela
            carga += demanda[c]

            estourou = carga > capacidade or chegada > due[c]
            if j > i and estourou:
                break                   # esticar mais só piora (monotônico)

            viol = 0.0
            if estourou:                # só possível na rota unitária (j == i)
                if chegada > due[c]:
                    viol += chegada - due[c]
                if carga > capacidade:
                    viol += carga - capacidade

            dist_rota += dist_seg
            tempo = chegada + service[c]
            pos = c

            retorno = tempo + matriz[pos][0]
            if retorno > horizonte:     # retorno só atrasa ao esticar a rota
                if j > i:
                    break
                viol += retorno - horizonte

            cand = (ci[0] + viol, ci[1] + 1, ci[2] + dist_rota + matriz[pos][0])
            if cand < custo[j + 1]:
                custo[j + 1] = cand
                pred[j + 1] = i
            if viol > 0:
                break                   # rota com violação não é esticada
            j += 1

    # Reconstrói as rotas da melhor partição seguindo os predecessores
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
    """
    Calcula todas as métricas percorrendo as rotas (fonte única de verdade
    para distância e violações — independente de como as rotas surgiram):
      - distância total (euclidiana, com ida e volta ao depósito);
      - violação de capacidade (excesso de carga por rota);
      - violação de janela (atraso na chegada; espera não é violação) e
        retorno ao depósito após o horizonte;
      - violação de frota (veículos além do limite da instância).
    """
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
    """Atalho quando só interessa a avaliação (descarta as rotas)."""
    return decodificar_e_avaliar(solucao, matriz, clientes, capacidade, max_veiculos)[1]


# ---------------------------------------------------------------------------
# 3. REGRAS DE DEB (tratamento de restrições escolhido para TODOS os algoritmos)
# ---------------------------------------------------------------------------
def eh_melhor_deb(a, b):
    """
    Compara duas avaliações pelas regras de Deb:
      1. entre duas soluções viáveis, vence a de melhor objetivo;
      2. entre uma viável e uma inviável, vence a viável;
      3. entre duas inviáveis, vence a de menor violação total.
    Retorna True se 'a' é melhor que 'b'.
    """
    if a["violacao"] == 0 and b["violacao"] == 0:
        return a["objetivo"] < b["objetivo"]
    if a["violacao"] == 0:
        return True
    if b["violacao"] == 0:
        return False
    return a["violacao"] < b["violacao"]


def chave_deb(aval):
    """
    Chave de ordenação equivalente às regras de Deb (menor = melhor):
    soluções viáveis vêm antes (ordenadas por objetivo) e as inviáveis
    depois (ordenadas por violação e, em empate, por objetivo).
    """
    if aval["violacao"] == 0:
        return (0, aval["objetivo"], 0.0)
    return (1, aval["violacao"], aval["objetivo"])


# ---------------------------------------------------------------------------
# 4. POPULAÇÃO INICIAL HEURÍSTICA (mesma estratégia do ag_tsp.py original)
# ---------------------------------------------------------------------------
def solucoes_iniciais_heuristicas(n, clientes, matriz, quantidade):
    """
    Gera 'quantidade' permutações iniciais combinando quatro famílias
    (estratégia herdada do ag_tsp.py original):
      1. ordenações por TEMPO (due_date e ready_time) + variações mutadas;
      2. vizinho mais próximo (ESPAÇO puro), com início aleatório;
      3. guloso ESPAÇO-TEMPO: minimiza distância * due_date do candidato;
      4. permutações aleatórias para completar a diversidade.
    """
    solucoes = []

    # 1. Ordenações por tempo
    solucao_due = sorted(range(1, n), key=lambda c: clientes[c]["due_date"])
    solucoes.append(solucao_due)
    solucao_ready = sorted(range(1, n), key=lambda c: clientes[c]["ready_time"])
    solucoes.append(solucao_ready)

    # Variações da ordenação por due_date (trocas aleatórias leves)
    for _ in range(max(1, int(quantidade * 0.2))):
        mutante = solucao_due[:]
        for _ in range(5):
            i, j = random.sample(range(len(mutante)), 2)
            mutante[i], mutante[j] = mutante[j], mutante[i]
        solucoes.append(mutante)

    # 2. Vizinho mais próximo (início aleatório)
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

    # 3. Guloso espaço-tempo (distância x urgência da janela)
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

    # 4. Aleatórias até completar
    while len(solucoes) < quantidade:
        aleatoria = list(range(1, n))
        random.shuffle(aleatoria)
        solucoes.append(aleatoria)

    return solucoes[:quantidade]


# ---------------------------------------------------------------------------
# 5. CONVERSÃO PERMUTAÇÃO <-> CHAVES ALEATÓRIAS (para PSO e DE)
# ---------------------------------------------------------------------------
def chaves_para_permutacao(chaves):
    """
    Decodifica um vetor contínuo (random keys) em permutação de clientes:
    a posição de cada cliente no tour é dada pela ordem crescente da sua
    chave. A dimensão k corresponde ao cliente k+1.
    """
    return [int(i) + 1 for i in np.argsort(chaves, kind="stable")]


def permutacao_para_chaves(perm):
    """Operação inversa: gera chaves que reproduzem a permutação dada
    (usada para semear PSO/DE com as soluções iniciais heurísticas)."""
    n = len(perm)
    chaves = np.empty(n)
    for posicao, cliente in enumerate(perm):
        chaves[cliente - 1] = (posicao + 0.5) / n
    return chaves


# ---------------------------------------------------------------------------
# 6. BUSCA LOCAL SIMPLES (pós-processamento, igual para todos os algoritmos)
# ---------------------------------------------------------------------------
def busca_local(solucao, aval, matriz, clientes, capacidade, max_veiculos,
                max_tentativas=20000, tempo_limite=None):
    """
    Subida de encosta estocástica sobre o tour gigante, aplicada apenas à
    MELHOR solução ao final da evolução (não altera o algoritmo original).
    Vizinhanças sorteadas a cada tentativa:
      - troca de dois clientes (swap);
      - realocação de um cliente para outra posição (or-opt de tamanho 1);
      - realocação de um segmento de 2 ou 3 clientes (or-opt maior — move
        pedaços de rota inteiros, o que ajuda a esvaziar/fundir rotas);
      - inversão de um trecho (2-opt no tour gigante).
    Movimentos são aceitos se forem melhores pelas regras de Deb.
    """
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
        if sorteio < 0.25:                       # troca
            vizinho[i], vizinho[j] = vizinho[j], vizinho[i]
        elif sorteio < 0.50:                     # realocação de 1 cliente
            cliente = vizinho.pop(i)
            vizinho.insert(j, cliente)
        elif sorteio < 0.75:                     # realocação de segmento (2-3)
            tam_seg = random.randint(2, 3)
            i = random.randint(0, n - tam_seg)
            segmento = vizinho[i:i + tam_seg]
            del vizinho[i:i + tam_seg]
            destino = random.randint(0, len(vizinho))
            vizinho[destino:destino] = segmento
        else:                                    # inversão de trecho (2-opt)
            vizinho[i:j + 1] = reversed(vizinho[i:j + 1])

        aval_vizinho = avaliar(vizinho, matriz, clientes, capacidade, max_veiculos)
        if eh_melhor_deb(aval_vizinho, melhor_aval):
            melhor = vizinho
            melhor_aval = aval_vizinho
            melhorias += 1

    return melhor, melhor_aval, melhorias


# ---------------------------------------------------------------------------
# 7. VERIFICAÇÃO INDEPENDENTE DA SOLUÇÃO FINAL
# ---------------------------------------------------------------------------
def verificar_solucao(rotas, matriz, clientes, capacidade, max_veiculos, n):
    """
    Reconfere TODAS as restrições do VRPTW de forma independente do
    decodificador (defesa contra bugs — solução inviável é desclassificada):
      - cada cliente atendido exatamente uma vez (sem faltas nem repetições);
      - capacidade respeitada em cada rota;
      - janelas de tempo respeitadas (com espera quando chega cedo);
      - número de rotas dentro do limite da frota.
    Retorna (viavel, lista_de_problemas).
    """
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
        # retorno ao depósito dentro do horizonte (janela do depósito)
        retorno = tempo + matriz[pos][0]
        if retorno > clientes[0]["due_date"] + 1e-9:
            problemas.append(f"rota {k}: retorno ao deposito apos o horizonte")

    return len(problemas) == 0, problemas


# ---------------------------------------------------------------------------
# 8. SAÍDA NO FORMATO DA COMPETIÇÃO
# ---------------------------------------------------------------------------
def calcular_distancia_rotas(rotas, matriz):
    """Distância total das rotas (depósito -> clientes -> depósito)."""
    total = 0.0
    for rota in rotas:
        pos = 0
        for c in rota:
            total += matriz[pos][c]
            pos = c
        total += matriz[pos][0]
    return total


def _ler_resultado_existente(caminho):
    """Lê veículos e distância de um arquivo de resultado já gravado.
       Retorna (veiculos, distancia) ou None se não existir/não parsear."""
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
    """
    Grava o arquivo de resultado no formato exigido pelo manual da
    competição (4 casas decimais, rotas com depósito nas pontas).
    Ex. de nome: dados/resultados/c101_resultado_ag.txt

    Como as regras permitem várias execuções e só a melhor conta, o
    arquivo NÃO é sobrescrito se já contiver uma solução melhor pelos
    critérios da competição (menos veículos; em empate, menor distância).
    """
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
    """Imprime no console o resumo completo pedido no enunciado do trabalho."""
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
