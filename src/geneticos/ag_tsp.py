import os
import random
import matplotlib.pyplot as plt
import numpy as np
import math
import time 

# Raiz do projeto (dois níveis acima de src/geneticos/)
RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# Configurações
TAM_POP_INI = 100
NUM_ITE_MAX = 200
num_ite = 0

def carregar_vrptw(caminho_arquivo):
    """Lê o arquivo do VRPTW e extrai o limite de veículos"""
    clientes = {}
    capacidade = 0 
    max_veiculos = 0 # NOVO
    
    with open(caminho_arquivo, 'r') as f:
        linhas = f.readlines()
    
    for linha in linhas:
        partes = linha.split()

        # Captura os dois valores da linha de configuração
        if len(partes) == 2 and partes[0].isdigit() and partes[1].isdigit():
            max_veiculos = int(partes[0]) # Captura o limite
            capacidade = int(partes[1])

        elif len(partes) >= 7:
            try:
                valores = list(map(int, partes[:7]))
                id_no = valores[0]
                clientes[id_no] = {
                    'x': valores[1],
                    'y': valores[2],
                    'demanda': valores[3],
                    'ready_time': valores[4],
                    'due_date': valores[5],
                    'service_time': valores[6]
                }
            except ValueError:
                continue

    n = len(clientes)
    matriz_adj = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = clientes[i]['x'] - clientes[j]['x']
                dy = clientes[i]['y'] - clientes[j]['y']
                matriz_adj[i][j] = math.sqrt(dx**2 + dy**2)
                
    return clientes, matriz_adj, capacidade, max_veiculos, n # Retorna o novo parâmetro


def fitness_deb(solucao, matriz, clientes, capacidade, max_veiculos):
    distancia_total = 0
    veiculos_usados = 1
    
    carga_atual = 0
    tempo_atual = 0
    posicao_atual = 0
    
    for cliente in solucao:
        dados_cliente = clientes[cliente]
        dist_viagem = matriz[posicao_atual][cliente]
        tempo_chegada = tempo_atual + dist_viagem
        
        if tempo_chegada < dados_cliente['ready_time']:
            tempo_chegada = dados_cliente['ready_time']
            
        violou_capacidade = (carga_atual + dados_cliente['demanda']) > capacidade
        violou_janela = tempo_chegada > dados_cliente['due_date']
        
        if violou_capacidade or violou_janela:
            distancia_total += matriz[posicao_atual][0]
            veiculos_usados += 1
            posicao_atual = 0
            carga_atual = 0
            
            dist_viagem = matriz[posicao_atual][cliente]
            tempo_chegada = dist_viagem
            if tempo_chegada < dados_cliente['ready_time']:
                tempo_chegada = dados_cliente['ready_time']

        distancia_total += dist_viagem
        carga_atual += dados_cliente['demanda']
        tempo_atual = tempo_chegada + dados_cliente['service_time']
        posicao_atual = cliente

    distancia_total += matriz[posicao_atual][0]

    # --- LÓGICA PARA AS REGRAS DE DEB ---
    # Objetivo: Como ainda queremos minimizar a frota MESMO estando dentro do limite,
    # mantemos um peso nos veículos.
    objetivo = distancia_total + (veiculos_usados * 1000)
    
    # Violação: Quantos veículos a solução usou acima do limite oficial?
    violacao = max(0, veiculos_usados - max_veiculos)
    
    return objetivo, violacao

def OX(pai1, pai2):
    tam = len(pai1)
    pos1, pos2 = sorted(random.sample(range(tam), 2))

    filho_1 = [None] * tam 
    filho_2 = [None] * tam 
    
    filho_1[pos1: pos2 +1] = pai2[pos1: pos2 +1]
    filho_2[pos1: pos2 +1] = pai1[pos1: pos2 +1]

    pos_atual = (pos2 + 1) % tam
    pos_pai = (pos2 + 1) % tam
    
    while None in filho_1:
        if pai1[pos_pai] not in filho_1:
            filho_1[pos_atual] = pai1[pos_pai]
            pos_atual = (pos_atual + 1) % tam
        pos_pai = (pos_pai + 1) % tam

    pos_atual = (pos2 + 1) % tam
    pos_pai = (pos2 + 1) % tam
    
    while None in filho_2:
        if pai2[pos_pai] not in filho_2:
            filho_2[pos_atual] = pai2[pos_pai]
            pos_atual = (pos_atual + 1) % tam
        pos_pai = (pos_pai + 1) % tam

    return filho_1, filho_2
    

def mutacao(individuo, taxa=0.15):
    r = random.uniform(0, 1)
    tam = len(individuo)
    if r <= taxa:
        pos1, pos2 = sorted(random.sample(range(tam), 2))
        individuo[pos1], individuo[pos2] = individuo[pos2], individuo[pos1]
    return individuo

def selecao_torneio_deb(populacao, matriz, clientes, capacidade, max_veiculos, k=3):
    competidores = random.sample(populacao, k)
    melhor = competidores[0]
    obj_melhor, viol_melhor = fitness_deb(melhor, matriz, clientes, capacidade, max_veiculos)
    
    for i in range(1, k):
        atual = competidores[i]
        obj_atual, viol_atual = fitness_deb(atual, matriz, clientes, capacidade, max_veiculos)
        
        # Regras de Restrição de Deb:
        if viol_atual == 0 and viol_melhor == 0:
            # 1. Ambos viáveis: Vence quem tem a menor distância
            if obj_atual < obj_melhor:
                melhor, obj_melhor, viol_melhor = atual, obj_atual, viol_atual
                
        elif viol_atual == 0 and viol_melhor > 0:
            # 2. Um viável e outro inviável: O viável vence automaticamente
            melhor, obj_melhor, viol_melhor = atual, obj_atual, viol_atual
            
        elif viol_atual > 0 and viol_melhor > 0:
            # 3. Ambos inviáveis: Vence o que está mais perto de se tornar viável
            if viol_atual < viol_melhor:
                melhor, obj_melhor, viol_melhor = atual, obj_atual, viol_atual
                
    return melhor



def obter_rotas_e_distancia(solucao, matriz, clientes, capacidade):
    rotas = []
    rota_atual = []
    
    carga_atual = 0
    tempo_atual = 0
    posicao_atual = 0  # Depósito
    distancia_total = 0
    
    for cliente in solucao:
        dados_cliente = clientes[cliente]
        dist_viagem = matriz[posicao_atual][cliente]
        tempo_chegada = tempo_atual + dist_viagem
        
        if tempo_chegada < dados_cliente['ready_time']:
            tempo_chegada = dados_cliente['ready_time']
            
        violou_capacidade = (carga_atual + dados_cliente['demanda']) > capacidade
        violou_janela = tempo_chegada > dados_cliente['due_date']
        
        if violou_capacidade or violou_janela:
            # Fecha a rota do veículo atual (volta para o depósito)
            distancia_total += matriz[posicao_atual][0]
            rotas.append(rota_atual)
            
            # Reseta as variáveis para o novo veículo
            rota_atual = []
            posicao_atual = 0
            carga_atual = 0
            tempo_atual = 0
            
            # Recalcula o trajeto do depósito até o cliente atual
            dist_viagem = matriz[posicao_atual][cliente]
            tempo_chegada = dist_viagem
            if tempo_chegada < dados_cliente['ready_time']:
                tempo_chegada = dados_cliente['ready_time']
                
        # Adiciona o cliente na rota ativa
        rota_atual.append(cliente)
        distancia_total += dist_viagem
        carga_atual += dados_cliente['demanda']
        tempo_atual = tempo_chegada + dados_cliente['service_time']
        posicao_atual = cliente
        
    # Salva a última rota pendente
    if rota_atual:
        distancia_total += matriz[posicao_atual][0]
        rotas.append(rota_atual)
        
    return rotas, distancia_total

def solucoes_iniciais_inteligentes(n, clientes, matriz, tam_pop):
    solucoes = []
    
    # 1. Indivíduos baseados em TEMPO PURO (Due Date e Ready Time)
    solucao_due_date = sorted(range(1, n), key=lambda x: clientes[x]['due_date'])
    solucoes.append(solucao_due_date)
    
    solucao_ready_time = sorted(range(1, n), key=lambda x: clientes[x]['ready_time'])
    solucoes.append(solucao_ready_time)
    
    # Pequenas variações de tempo (10% da população)
    for _ in range(int(tam_pop * 0.2)):
        mutante = solucao_due_date.copy()
        for _ in range(5): 
            mutante = mutacao(mutante, taxa=1.0)
        solucoes.append(mutante)

    # 2. Indivíduos baseados em ESPAÇO PURO (Vizinho Mais Próximo)
    # Reduzido para 40% da população
    for _ in range(int(tam_pop * 0.3)):
        nao_visitados = list(range(1, n))
        rota_gulosa = []
        
        atual = random.choice(nao_visitados)
        rota_gulosa.append(atual)
        nao_visitados.remove(atual)
        
        while nao_visitados:
            prox = min(nao_visitados, key=lambda x: matriz[atual][x])
            rota_gulosa.append(prox)
            nao_visitados.remove(prox)
            atual = prox
            
        solucoes.append(rota_gulosa)

    # ====================================================================
    # 3. NOVO: Indivíduos baseados em ESPAÇO-TEMPO (Distância x Due Date)
    # Inserido com 40% de peso na população
    # ====================================================================
    for _ in range(int(tam_pop * 0.3)):
        nao_visitados = list(range(1, n))
        rota_espaco_tempo = []
        
        # Ponto de partida aleatório para gerar rotas com inícios diferentes
        atual = random.choice(nao_visitados)
        rota_espaco_tempo.append(atual)
        nao_visitados.remove(atual)
        
        while nao_visitados:
            # O "Score" minimiza a distância, mas pune quem tem due_date muito alto
            prox = min(nao_visitados, key=lambda x: matriz[atual][x] * clientes[x]['due_date'])
            rota_espaco_tempo.append(prox)
            nao_visitados.remove(prox)
            atual = prox
            
        solucoes.append(rota_espaco_tempo)
        
    # 4. Indivíduos 100% ALEATÓRIOS (para preencher o resto e garantir a diversidade)
    while len(solucoes) < tam_pop:
        aleatoria = list(range(1, n))
        random.shuffle(aleatoria)
        solucoes.append(aleatoria)
        
    return solucoes

caminho_instancia = f'{RAIZ}/dados/vrptw/rc2_4_9.txt'

clientes, matriz, capacidade, max_veiculos, num_nos = carregar_vrptw(caminho_instancia)

tempo_inicio = time.time()
populacao = solucoes_iniciais_inteligentes(num_nos, clientes, matriz, TAM_POP_INI)
num_ite = 0

melhor_solucao_global = None
# Vamos rastrear o objetivo (distância) da melhor solução viável
melhor_objetivo_global = float('inf') 

# Define a taxa de elitismo (ex: manter os 4 melhores - aprox 10% de 50)
QTD_ELITE = 6

while num_ite < NUM_ITE_MAX:
    novos_filhos = []
    
    # --- ELITISMO ---
    # Precisamos ordenar a população. Como Deb usa duas métricas, criamos um score rápido:
    # Soluções inviáveis recebem um valor na casa dos milhões para irem para o fim da lista
    def score_ordenacao(ind):
        obj, viol = fitness_deb(ind, matriz, clientes, capacidade, max_veiculos)
        return obj if viol == 0 else (1000000 + viol * 10000 + obj)
        
    populacao_ordenada = sorted(populacao, key=score_ordenacao)
    
    # Copia a elite diretamente para a próxima geração
    novos_filhos.extend(populacao_ordenada[:QTD_ELITE])
    
    # --- CRUZAMENTO E MUTAÇÃO (preenchendo o resto da população) ---
    while len(novos_filhos) < TAM_POP_INI:
        pai1 = selecao_torneio_deb(populacao, matriz, clientes, capacidade, max_veiculos)
        pai2 = selecao_torneio_deb(populacao, matriz, clientes, capacidade, max_veiculos)
        
        f1, f2 = OX(pai1, pai2)
        
        f1 = mutacao(f1, taxa=0.1)
        f2 = mutacao(f2, taxa=0.1)
        
        novos_filhos.append(f1)
        if len(novos_filhos) < TAM_POP_INI:
            novos_filhos.append(f2)
            
    populacao = novos_filhos
    
    # --- AVALIAÇÃO DA GERAÇÃO ---
    melhor_da_geracao = populacao_ordenada[0] # A elite já nos dá o melhor!
    obj_geracao, viol_geracao = fitness_deb(melhor_da_geracao, matriz, clientes, capacidade, max_veiculos)
    
    # Atualiza a melhor solução global APENAS se for estritamente melhor e viável (ou a menos pior)
    if viol_geracao == 0 and obj_geracao < melhor_objetivo_global:
        melhor_objetivo_global = obj_geracao
        melhor_solucao_global = melhor_da_geracao.copy()
        
    if num_ite % 10 == 0:
        status = "VIÁVEL" if viol_geracao == 0 else f"INVÁVEL (Passou {viol_geracao} veículos)"
        print(f"Geração {num_ite}: Objetivo = {obj_geracao:.2f} [{status}]")
    
    num_ite += 1

tempo_fim = time.time()
tempo_execucao = tempo_fim - tempo_inicio

# Extrai o nome da instância a partir do caminho do arquivo
nome_instancia = os.path.basename(caminho_instancia).split('.')[0]

# Decodifica o melhor indivíduo para pegar as rotas separadas e a distância real
rotas_finais, distancia_real = obter_rotas_e_distancia(
    melhor_solucao_global, matriz, clientes, capacidade
)

# Saída Exatamente no formato desejado
print("\n" + "="*30)
print(f"Nome da instância: {nome_instancia}")
print(f"Número de veículos: {len(rotas_finais)}")
print(f"Distância total: {distancia_real:.2f}")
print("Rotas:")
for i, rota in enumerate(rotas_finais, 1):
    # Converte a lista de inteiros em uma string separada por espaços
    rota_str = " ".join(map(str, rota))
    print(f"Rota {i}: {rota_str}")
print("="*30)